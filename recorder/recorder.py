# takes as input scene and returns a dataframe with the sensor data
import itertools
from sim.scene import Scene, parse_scene
from recorder.utils import prune_spikes, add_visual_capsule, get_geom_speed, draw_trails, get_body_forward_vector, estimate_trail_radius_from_geom, draw_regions, unit_cos, get_body_color
import mujoco
import hydra
import glob
from omegaconf import DictConfig, OmegaConf
import traceback

import os
import imageio
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import copy
import ipdb
st = ipdb.set_trace
import matplotlib.font_manager as fm

import logging, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(
    filename="run.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

from recorder.contact_utils import (get_text, parse_custom_data,
                           process_coefficients_friction, process_coefficients_restitution, 
                           calculate_contact_force, apply_restitution_correction)

try:
    fm.rebuild()
except AttributeError:
    pass 
font_path = "sim/tests/fonts/times.ttf"
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams.update({'font.size': 6})
plt.rcParams['font.family'] = font_prop.get_name()

mjWRAP_CONSTANTS = {
    0: "mjWRAP_NONE",                   # null object
    1: "mjWRAP_JOINT",                   # constant moment arm
    2: "mjWRAP_PULLEY",                  # pulley used to split tendon
    3: "mjWRAP_SITE",                    # pass through site
    4: "mjWRAP_SPHERE",                  # wrap around sphere
    5: "mjWRAP_CYLINDER"                 
}

SCENE_TYPE_TO_CATEGORY_MAP = {
    "pulley": ["BasicPulley", "IntermediatePulley", "BasicInclinedPlaneFriction", "IntermediateInclinedPlaneFriction", "AdvancedInclinedPlaneFriction", "IntermediateHybrid", "AdvancedHybrid", "DifficultPulley"],
    "collision": ["BasicCollision", "IntermediateCollision", "AdvancedCollision"],
    "spring": ["SpringBlockSystems", "DifficultSpringMass"],
    "rotation": ["Rotation", "RigidBodyRotation"],
    "orbital": ["DifficultOrbitalMotion"],
    "em": ["DifficultElectroMagnetic"]
}

# Entity priority configuration: selection priority when multiple entities exist in a scene
ENTITY_PRIORITY_CONFIG = {
    # Entity type priority (lower number = higher priority)
    "MassPrismPlaneEntity": 1,
    "StackedMassPlane": 2, 
    "TwoSideMassPlane": 3,
    "DirectedMass": 4,
    "MassWithMovablePulley": 5,
    "MassWithReversedMovablePulley": 6,
    "MassWithFixedPulley": 7,
    "ConstantForceFixedPulley": 8,
    "FixedPulleyEntity": 9,
    # Collision entities
    "ComplexCollisionPlane": 2,
    "TwoDCollisionPlane": 3,
    # Spring systems  
    "SpringMassPlaneEntity": 2,
    "SpringBlockEntity": 3,
    # Box and other plane entities
    "MassBoxPlaneEntity": 2,
    "MassPrismPulleyPlane": 3,
    # Others
    "default": 10  # Default priority
}

# Scene focus configuration: based on all scene types from scene_generator.py
SCENE_FOCUS_CONFIG = {
    # Scene types that need to focus on specific body
    "focus_on_mass": {
        # Collision scenes - focus on moving masses
        "collision": [
            "BasicCollision", "IntermediateCollision", "AdvancedCollision", 
        ],
        # Spring-mass systems - focus on masses
        "spring_mass": [
            "SpringBlockSystems", "DifficultSpringMass"
        ],
        # Inclined plane with mass - focus on masses
        "inclined_plane": [
            "BasicInclinedPlaneFriction", "IntermediateInclinedPlaneFriction", 
            "AdvancedInclinedPlaneFriction"
        ],
        # Hybrid systems - may contain multiple entities, need smart selection
        "hybrid": [
            "IntermediateHybrid", "AdvancedHybrid"
        ],
        "special": [
            "DifficultPulley"  # Though it's pulley but contains mass plane entity
        ]
    },
    
    # Scene types that use scene center
    "focus_on_center": {
        # Pulley systems - need global view of entire system
        "pulley": [
            "BasicPulley", "IntermediatePulley"
        ],
        # Orbital and celestial systems - need global view
        "orbital": [
            "DifficultOrbitalMotion"
        ],
        # Rotation systems - need global view
        "rotation": [
            "Rotation", "RigidBodyRotation"
        ],
        # Special systems
        "special": [
            "DifficultRocket", "DifficultProjectile", 
            "DifficultElectroMagnetic", "Test"
        ]
    }
}

# Adaptive camera parameter configuration: parameter groups for different scene types
ADAPTIVE_CAMERA_PARAMS = {
    "focus_body": {
        # Parameters for focusing on specific body
        "safety_margin": 0.8,      # Closer distance for clear body visibility
        "min_factor": 0.2,         # Allow very close distance
        "max_factor": 1.5,         # Limit maximum distance
        "fov": 45.0,
        "fixed_distance": 5.0      # Fixed focus distance
    },
    "scene_center_close": {
        # Scene center with close distance (e.g., collision scenes)
        "safety_margin": 0.6,
        "min_factor": 0.3,
        "max_factor": 2.0,
        "fov": 50.0
    },
    "scene_center_medium": {
        # Scene center with medium distance (e.g., pulley scenes)
        "safety_margin": 0.8,
        "min_factor": 0.4,
        "max_factor": 2.5,
        "fov": 45.0
    },
    "scene_center_far": {
        # Scene center with far distance (e.g., orbital scenes)
        "safety_margin": 1.0,
        "min_factor": 0.5,
        "max_factor": 3.0,
        "fov": 40.0
    }
}

# Scene type to parameter group mapping
SCENE_TO_PARAMS_MAP = {
    # Pulley scenes
    **{scene: "scene_center_medium" for scene in SCENE_FOCUS_CONFIG["focus_on_center"]["pulley"]},
    # Orbital scenes  
    **{scene: "scene_center_far" for scene in SCENE_FOCUS_CONFIG["focus_on_center"]["orbital"]},
    # Rotation scenes
    **{scene: "scene_center_medium" for scene in SCENE_FOCUS_CONFIG["focus_on_center"]["rotation"]},
    # Special scenes
    **{scene: "scene_center_medium" for scene in SCENE_FOCUS_CONFIG["focus_on_center"]["special"]},
    # Focus on mass scenes use close parameters when falling back to scene center
    **{scene: "scene_center_close" for scenes in SCENE_FOCUS_CONFIG["focus_on_mass"].values() for scene in scenes},
}

DEFAULT_CAMERA_CONFIGS = {
    "default": {
        "lookat": [0.0, 0.0, 0.0],
        "distance": 10,
        "azimuth": 0.0,
        "elevation": -30.0,
    },
    "AdvancedInclinedPlaneFriction": {
        "lookat": [0.0, 0.0, 1],
        "distance": 12,
        "azimuth": 90.0,
        "elevation": -45.0,
    },
    "IntermediateHybrid": {
        "lookat": [0.0, 0.0, 1],
        "distance": 15,
        "azimuth": 90.0,
        "elevation": -45.0,
    },
    "AdvancedHybrid": {
        "lookat": [0.0, 0.0, 1],
        "distance": 15,
        "azimuth": 90.0,
        "elevation": -45.0,
    },
    "IntermediateInclinedPlaneFriction": {
        "lookat": [0.0, 0.0, 1],
        "distance": 12,
        "azimuth": 90.0,
        "elevation": -45.0,
    },
    "BasicInclinedPlaneFriction": {
        "lookat": [3, 0, 1],
        "distance": 12,
        "azimuth": 90.0,
        "elevation": -45.0,
    },
    "Rotation": {
        "azimuth": 90.0,
        "elevation": -45.0,
        "lookat": [0, 0, 0],
        "distance": 5
    },
    "SpringBlockSystems": {
        "lookat": [-2, 0, -0.5],
        "distance": 5,
    },
    "BasicPulley": {
        "lookat": [0, 0, -0.5],
        "distance": 5
    },
    "IntermediatePulley": {
        "lookat": [0, 0, -0.5],
        "distance": 10
    },
    "OrbitalMotion": {
        "distance": 15,
    }
}

def custom_warning(*args, **kwargs):
    pass

class Recorder:
    def __init__(self, scene: Scene, 
                 cfg: DictConfig,
                 scene_folder: str, variation_index: int = 0, category: str = None):
        self.scene = scene
        self.cfg = cfg
        self.category = category

        # Overload config
        if category is not None:
            override_cfg = OmegaConf.load(f"config/recorder/{category}.yaml")
            self.cfg = OmegaConf.merge(self.cfg, override_cfg)

        self.scene_folder = scene_folder
        self.variation_index = variation_index
        self.model = self.to_MjModel()

        # initialize the counters for the bodies
        self.mass_counter = 0
        self.site_counter = 0
        self.plane_counter = 0
        self.tendon_counter = 0
        self.pulley_counter = 0
        self.prism_counter = 0
        
        self.data = mujoco.MjData(self.model)
        self.renaming_dict = {}
        
        self.params_set = scene.get_parameters()
        
        self.expected_steps = int(self.cfg.duration / self.cfg.dt)
        self.data_dict = defaultdict(lambda: defaultdict(list))
        global_params, entity_params, tendon_params = self.params_set
        
        # for param in entity_params:
        #     st()
        #     if "slope" in param.keys():
        #         self.data_dict[param["name"]]["slope_angle"] = param["slope"]
        
        if cfg.render:
            self.renderer = mujoco.Renderer(self.model, cfg.height, cfg.width)
            # if cfg.custom_camera:
            #     # Manually set the camera's pos
            #     self.cam = mujoco.MjvCamera()
            #     self.cam.lookat = cfg.lookat
            #     self.cam.distance = cfg.distance
            #     self.cam.azimuth = cfg.azimuth
            #     self.cam.elevation = cfg.elevation

            camera_cfg = DEFAULT_CAMERA_CONFIGS["default"].copy()
            if scene.tag in DEFAULT_CAMERA_CONFIGS:
                camera_cfg.update(DEFAULT_CAMERA_CONFIGS[scene.tag])

            if getattr(cfg, "custom_camera", False):
                for key in ("lookat", "distance", "azimuth", "elevation"):
                    if hasattr(cfg, key):
                        camera_cfg[key] = getattr(cfg, key)
            # camera_cfg = {
            #     "lookat": [7, 0, 0],
            #     "distance": 18,
            #     "azimuth": 90.0,
            #     "elevation": -30.0,
            # }
            camera_cfg = {
                "lookat": [0, 0, 0],
                "distance": 18,
                "azimuth": 90.0,
                "elevation": -30.0,
            }
            self._cam_cfg = camera_cfg

            self.cam = mujoco.MjvCamera()
            self.cam.lookat     = camera_cfg["lookat"]
            self.cam.distance   = camera_cfg["distance"]
            self.cam.azimuth    = camera_cfg["azimuth"]
            self.cam.elevation  = camera_cfg["elevation"]
            
            self.scene_center = self.scene.get_center()
        
        # Create a boolean mask for which tendons are actuated
        self.actuated_tendons = np.zeros(self.model.ntendon, dtype=bool)

        # Find all actuators that act on tendons
        tendon_actuators = self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON
        tendon_ids = self.model.actuator_trnid[tendon_actuators, 0]  # Extract the tendon indices for actuators

        # Mark these tendons as actuated
        self.tendon_actuators = tendon_actuators
        self.actuated_tendons[tendon_ids] = True
        self.actuated_tendon_ids = tendon_ids

        # # Init rockets
        # self.init_rocket()
            
    def to_MjModel(self) -> mujoco.MjModel:
        # Supress warning
        mujoco.mjcb_warning = custom_warning

        self.xml = self.scene.to_xml()
        self.spec = mujoco.MjSpec.from_string(self.xml)

        # Set options
        self.spec.option.timestep = self.cfg.dt
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_RK4
        
        self.model = self.spec.compile() # mujoco.MjModel.from_xml_string(self.xml)
        
        return self.model
    
    def prune_timesteps(self, data):
        if self.cfg.prune_first_contact:
            raise NotImplementedError("prune_first_contact not implemented")
            contact_idx = np.min(np.where(contact))
            sensor_data = {k: v[:contact_idx] for k, v in sensor_data.items()}
            times = times[:contact_idx]
        
        if self.cfg.prune_derivative:
            time = np.array(data['global']['time'])
            num_time_points = len(time)
            
            valid_mask = np.ones(num_time_points, dtype=bool)  # Mask to track valid derivative regions
            
            ## APPROACH 1
            # max_value = 0
            # init_values = []
            # for key, value_dict in data.items():
            #     if key == 'global':
            #         continue
                
            #     for subkey, series in value_dict.items():
            #         if subkey != "acceleration": continue
                    
            #         series = np.array(series)
            #         if series.ndim == 1:
            #             deriv = np.abs(np.diff(series) / np.diff(time))
            #         else:
            #             deriv = np.linalg.norm(np.diff(series[..., 3:], axis=0) / np.diff(time)[:, None], axis=1)

            #         init_values.append(deriv[0])
            #         max_value = max(max_value, max_value if np.max(deriv)>self.cfg.threshold_derivative else np.max(deriv))
            #         valid_mask &= deriv < self.cfg.threshold_derivative
            
            ## APPROACH 2
            for key, value_dict in data.items():
                if key == 'global':
                    continue

                for subkey, series in value_dict.items():
                    if subkey != "acceleration": continue
                    
                    series = np.array(series)
                    if series.ndim != 1:
                        series = np.linalg.norm(series[..., 3:], axis=1)
                    
                    mask = prune_spikes(series)

                    valid_mask &= mask

            # Find the first contiguous region where valid_mask is True
            start_idx, end_idx = None, None
            
            # Find start of first valid region
            for i in range(len(valid_mask)):
                if valid_mask[i]:
                    start_idx = i
                    break
                    
            # If we found a start, find the end
            if start_idx is not None:
                for i in range(start_idx, len(valid_mask)):
                    if not valid_mask[i]:
                        end_idx = i
                        break
                # Handle case where valid region extends to end
                if end_idx is None:
                    end_idx = len(valid_mask)-1
            else:
                # No valid region found
                start_idx = None
                end_idx = None
            
            return start_idx, end_idx
        
        if self.cfg.prune_tendon_length_change:
            tendon_keys = [k for k in data.keys() if 'length' in data[k].keys()]
            
            # Remove tendons that have constant force pulley in them
            tendon_keys = [k for k in tendon_keys if k[:12] != 'tendon_force']
            
            if len(tendon_keys) == 0: return 0, -1

            lengths = np.stack([data[k]['length'] for k in tendon_keys])
            # lengths = np.stack([data[f'tendon{t_idx}']['tendon_length'] for t_idx in range(self.tendon_counter)])
            lengths = lengths / lengths[:, 0, None] # relative lengths
            length_diff = np.diff(lengths)

            keep = length_diff < self.cfg.threshold_tendon_length_change
            
            keep = np.all(keep, axis = 0)

            remove = np.argwhere(1-keep).flatten()
            
            if len(remove) == 0:
                # Case 1: remove is empty
                return 0, keep.size

            if len(remove) == 1:
                # Case 2: remove has a single element
                if remove[0] > keep.size / 2:
                    return 0, remove[0]
                else:
                    return remove[0] + 1, keep.size

            if np.array_equal(remove, np.arange(keep.size)):
                # Case 3: remove is range(d), meaning no valid elements
                return None, None

            # General case: find largest gap in remove
            idx = np.argmax(np.diff(remove))
            return remove[idx] + 1, remove[idx + 1]

            for tendon_idx in range(self.tendon_counter):
                tendon_name = f"tendon_{tendon_idx}"
                length_diff = np.diff(data[tendon_name]["tendon_length"])
                remove_indices = np.where(length_diff > self.cfg.threshold_tendon_length_change)[0]
                try:
                    keep_index_min = remove_indices[np.argmax(remove_indices[1:] - remove_indices[:-1])]
                    keep_index_max = remove_indices[np.argmax(remove_indices[1:] - remove_indices[:-1]) + 1]
                except:
                    print(remove_indices.shape)
                    exit()
                
            return (keep_index_min, keep_index_max)
        else:
            assert False, "prune_timesteps no option selected"
              
    def get_info_from_wrap(self, wrap_id, wrap_type):
        if wrap_type == "mjWRAP_SITE":
            site_name = self.model.names[self.model.name_siteadr[wrap_id]:].split(b'\x00', 1)[0].decode("utf-8")
            body_id = self.model.site_bodyid[wrap_id]
            xpos = self.data.site_xpos[wrap_id]
            body_name = self.model.names[self.model.name_bodyadr[body_id]:].split(b'\x00', 1)[0].decode("utf-8")
            return site_name, body_name, xpos
        else:
            assert False, f"wrap_type {wrap_type} not implemented"
            return None, None, None
    
    def get_angle(self, position1, position2):
        vector = position2 - position1
        vector0 = np.array([1, 0, 0])   # x axis
        angle = np.arccos(np.dot(vector, vector0) / (np.linalg.norm(vector) * np.linalg.norm(vector0)))
        angle = np.degrees(angle)
        return angle

    def rename_body(self, body_name):
        if "spatial" in body_name:
            body_type = body_name
        else:
            body_type = [sub_str for sub_str in body_name.split("_") if not sub_str.isnumeric()][-1].lower()

        if body_name not in self.renaming_dict.keys():
            # come up with a new name for the body
            if "mass" in body_type:
                new_name = f"mass{self.mass_counter}"
                self.mass_counter += 1
            elif 'prism' in body_type:
                new_name = f"prism{self.mass_counter}"
                self.prism_counter += 1
            elif "plane" in body_type:
                new_name = f"plane{self.plane_counter}"
                self.plane_counter += 1
            elif "spatial" in body_type:
                new_name = f"tendon{self.tendon_counter}"
                self.tendon_counter += 1
            elif "pulley" in body_type:
                new_name = f"pulley{self.pulley_counter}"
                self.pulley_counter += 1
            else:
                # raise ValueError(f"body_name {body_name} not implemented")
                print(f'[recorder] Warning: body_name {body_name} not implemented')
                new_name = body_name
            self.renaming_dict[body_name] = new_name
            body_name = new_name
        else:
            body_name = self.renaming_dict[body_name]
        return body_name

    def update_metainfo(self, source=None, target=None):
        if target not in self.metainfo[source]:
            self.metainfo[source].append(target)

    def apply_external(self):
        constant_force_dict = self.scene.get_constant_force_dict()
        for body_name, force in constant_force_dict.items():
            body_index = self.model.body(body_name).id
            force = np.array(force)
            self.data.xfrc_applied[body_index] = force

    def set_initial_vel(self):
        
        # To fill the empty values in data (such as inertia, jacobian)
        mujoco.mj_forward(self.model, self.data)
        # mujoco.mj_step(self.model, self.data)
        
        # init_vel = parse_custom_data(get_text(self.model, 'init_vel'))
        init_vel = self.scene.get_init_velocity_dict()
        

        qvel = np.zeros_like(self.data.qvel)
        
        for body_name, velocity in init_vel.items():
            body_id = self.model.body(body_name).id
            nv = self.data.cdof.shape[0]
            Jl1, Jr1 = np.zeros((3, nv)), np.zeros((3, nv))

            mujoco.mj_jacBodyCom(self.model, self.data, Jl1, Jr1, body_id)
            Jcom1 = np.concatenate((Jl1, Jr1), axis = 0)
            
            vel = np.array(velocity)
            
            # Account for case when velocity is only 3D instead of 6D vector
            # vel = np.resize(vel, (6,))  # TODO(Aryan) changed to avoid error, check the correctness of this change
            vel.resize(6)
            
            qvel += Jcom1.T @ vel

        # mujoco.mj_resetData(self.model, self.data)
        self.data.qvel += qvel

    def control_velocity_actuators(self):
        for actuator in self.scene.actuators:
            actuator_name = actuator.name
            
            if actuator.type != "velocity": continue
            
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            
            self.data.ctrl[idx] = actuator.velocity
            
    def reset_external(self):
        self.data.xfrc_applied[self.external_bodies, :] = 0

    def calculate_gravity_forces(self):
        """
        Computes gravitational attraction for each pair of bodies listed by the scene's attraction forces,
        then applies them to self.data.xfrc_applied[].

        We expect each tuple to be: (bodyA_name, bodyB_name, "GRAVITY", G_value).

        The direction is from bodyA to bodyB for the force on A, and from B to A for the force on B.
        """
        attraction_forces = self.scene.get_attraction_forces()  # from Scene

        force_dict = {}

        # To load all data
        mujoco.mj_forward(self.model, self.data)

        for (bodyA_name, bodyB_name, force_type, G_val) in attraction_forces:
            if force_type != "GRAVITY":
                continue  # skip non-gravity forces if present

            # Find body indices
            bodyA_id = self.model.body(bodyA_name).id
            bodyB_id = self.model.body(bodyB_name).id

            # Masses
            # massA = self.model.body_mass[bodyA_id]
            # massB = self.model.body_mass[bodyB_id]

            massA = self.get_recursive_body_mass(self.spec.body(bodyA_name))
            massB = self.get_recursive_body_mass(self.spec.body(bodyB_name))

            if massA <= 0 or massB <= 0:
                continue

            # Positions
            posA = self.data.xpos[bodyA_id]  # COM position of body A
            posB = self.data.xpos[bodyB_id]  # COM position of body B

            # Distance vector bodyA->bodyB
            diff = posB - posA
            dist = np.linalg.norm(diff)
            if dist < 1e-9:
                continue  # avoid division by zero or extremely large forces

            # Unit direction vector from A->B
            direction = diff / dist

            # Magnitude of gravitational force
            force_magnitude = G_val * massA * massB / (dist ** 2)

            # Force on A is +direction * force_magnitude
            force_on_A = force_magnitude * direction
            # Force on B is -direction * force_magnitude
            force_on_B = -force_on_A

            # TODO: Add the bi-directional flag. For now, we assume it's always bi-directional.
            force_dict[bodyA_id] = force_dict.get(bodyA_id, 0) + force_on_A
            force_dict[bodyB_id] = force_dict.get(bodyB_id, 0) + force_on_B

        return force_dict
    
    def calculate_EM_forces(self):
        """
        Compute EM forces on each charged particles given in the scene based on the EM fields given.
        The charges do not apply any EM forces on each other (isolated / negligible)
        """

        force_dict = {}

        charged_particles = self.scene.get_charged_particles()

        for (body_name, charge) in charged_particles:
            # Find body indices
            body_id = self.model.body(body_name).id

            # Masses
            mass = self.get_recursive_body_mass(self.spec.body(body_name))

            if mass <= 0:
                continue

            position = self.data.xipos[body_id]  # COM position of the body
            velocity = self.data.cvel[body_id][3:]  # COM linear velocity of the body 

            Fields = self.scene.get_EM_fields(position)

            electrostatic_field = sum([x[0] for x in Fields])
            magnetic_field = sum([x[1] for x in Fields])
            if type(magnetic_field) in [int, float]:
                magnetic_field = np.zeros(3)

            F_electrostatic = charge * electrostatic_field
            F_magnetic = charge * np.cross(velocity, magnetic_field)

            force_dict[body_id] = force_dict.get(body_id, 0) + F_electrostatic + F_magnetic
        
        return force_dict

    def calculate_gravity_acceleration(self):
        """
        Computes gravitational attraction for each pair of bodies listed by the scene's attraction forces,
        then applies them to self.data.xfrc_applied[].

        We expect each tuple to be: (bodyA_name, bodyB_name, "GRAVITY", G_value).

        The direction is from bodyA to bodyB for the force on A, and from B to A for the force on B.
        """
        attraction_forces = self.scene.get_attraction_forces()  # from Scene

        force_dict = {}

        # To load all data
        mujoco.mj_forward(self.model, self.data)

        for (bodyA_name, bodyB_name, force_type, G_val) in attraction_forces:
            if force_type != "GRAVITY":
                continue  # skip non-gravity forces if present

            # Find body indices
            bodyA_id = self.model.body(bodyA_name).id
            bodyB_id = self.model.body(bodyB_name).id

            # Masses
            # massA = self.model.body_mass[bodyA_id]
            # massB = self.model.body_mass[bodyB_id]

            massA = self.get_recursive_body_mass(self.spec.body(bodyA_name))
            massB = self.get_recursive_body_mass(self.spec.body(bodyB_name))

            if massA <= 0 or massB <= 0:
                continue

            # Positions
            posA = self.data.xpos[bodyA_id]  # COM position of body A
            posB = self.data.xpos[bodyB_id]  # COM position of body B

            # Distance vector bodyA->bodyB
            diff = posB - posA
            dist = np.linalg.norm(diff)
            if dist < 1e-9:
                continue  # avoid division by zero or extremely large forces

            # Unit direction vector from A->B
            direction = diff / dist

            # Magnitude of gravitational force
            force_magnitude = G_val * massA * massB / (dist ** 2)

            # Force on A is +direction * force_magnitude
            force_on_A = force_magnitude * direction
            # Force on B is -direction * force_magnitude
            force_on_B = -force_on_A

            # TODO: Add the bi-directional flag. For now, we assume it's always bi-directional.
            force_dict[bodyA_id] = force_dict.get(bodyA_id, 0) + force_on_A
            force_dict[bodyB_id] = force_dict.get(bodyB_id, 0) + force_on_B

        return force_dict
    
    def has_joint_in_ancestry(self, body_idx):
        while body_idx != 0:  # 0 is the world body
            if self.model.body_jntnum[body_idx] > 0:
                return True
            body_idx = self.model.body_parentid[body_idx]
        return False

    def refresh_from_spec(self):
        self.model, self.data = self.spec.recompile(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        self.set_initial_vel()
        self.apply_external()
        self.control_velocity_actuators()
    
    def update_rocket(self):
        forces = {}
        rockets = self.scene.get_rockets()
        
        for rocket_name in rockets:
            v_exhaust = rockets[rocket_name]["v_exhaust"]
            dm_dt = rockets[rocket_name]["dm_dt"]
            min_mass = rockets[rocket_name]["min_mass"]

            body = self.spec.body(rocket_name)
            geom = body.bodies[-1].geoms[0]
            # dm_dt in scene params is signed (negative means mass loss).
            # Use physically meaningful mass depletion and clamp at min_mass.
            available_fuel_mass = max(geom.mass - min_mass, 0.0)
            mass_loss = min(abs(dm_dt) * self.cfg.dt, available_fuel_mass)
            geom.mass = geom.mass - mass_loss

            thrust = v_exhaust * mass_loss / max(self.cfg.dt, 1e-12)
            rocket_bid = self.model.body(rocket_name).id

            xmat = self.data.xmat[rocket_bid].reshape(3, 3)  # 3x3 rotation matrix
            forward = xmat[:, 2]  # +Z axis (launch direction in rocket local frame)

            forces[rocket_bid] = thrust * forward

        if len(rockets):
            self.refresh_from_spec()
        
        return forces

    def get_recursive_body_mass(self, body_spec):
        """Calculates the total effective mass of a body and its entire subtree."""
        total_mass = 0.0

        # 1. Add mass from explicit inertial if present
        if body_spec.explicitinertial:
            total_mass += body_spec.mass
        else:
            # 2. If no explicit inertial, add mass from direct geoms
            for geom_spec in body_spec.geoms:
                if geom_spec.mass is not None:
                    total_mass += geom_spec.mass
                # If mass is not specified on geom, it might default to 0 or calculate from density
                # MuJoCo's compilation handles this correctly, but for this manual spec traversal,
                # we'd need to know the rules. Assuming mass is explicitly given or is 0.

            # 3. Add mass from children (recursively)
            #    NOTE: If an explicit inertial is present on the *parent*, the children's inertias
            #    are ADDED to the parent's base. This function is calculating the *subtree's* mass.
            #    If we wanted just the *children's contribution* to a parent *without* an inertial,
            #    we would only sum the recursive call results here.
            for child_body_spec in body_spec.bodies:
                total_mass += self.get_recursive_body_mass(child_body_spec) # Recursively add child's total effective mass

        return total_mass

    def simulate(self) -> None:
        num_steps = 0
        stopped_due_to_instability = False
        self.frames = []
        all_data = defaultdict(lambda: defaultdict(list))
        self.metainfo = defaultdict(list)
        
        coefficient_friction = parse_custom_data(get_text(self.model, 'coefficient_friction'))
        coefficient_restitution = parse_custom_data(get_text(self.model, 'coefficient_restitution'))

        ## Remove later
        coefficient_restitution = [(t[0] + '.geom', t[1] + '.geom', *t[2:]) for t in coefficient_restitution]
        coefficient_friction = [(t[0] , t[1] , *t[2:]) for t in coefficient_friction]

        self.set_initial_vel()
        self.apply_external()

        self.control_velocity_actuators()

        cor_state = {}  # State dict for impulse-based COR correction
        while self.data.time < self.cfg.duration:
            
            ## (i) Zero out or reset existing external forces on the bodies that have mass
            # for body_idx in range(self.model.nbody):
            #     if self.model.body_mass[body_idx] != 0.0:
            #         self.data.xfrc_applied[body_idx] = 0.0

            # (ii) Compute and apply gravitational and EM forces
            gforce_dict = self.calculate_gravity_forces()  # We'll define this function below
            emforce_dict = self.calculate_EM_forces()  # We'll define this function below 

            # (iii) Then apply contact force, friction, step simulation, etc.
            contact_force_torque, applied_normal_frc, applied_friction_frc = calculate_contact_force(coefficient_friction, coefficient_restitution, self.model, self.data, damping = 1e-3)
            
            rocket_thrusts = self.update_rocket()

            self.data.qfrc_applied += contact_force_torque

            for body_idx, force in gforce_dict.items():
                self.data.xfrc_applied[body_idx, :3] += force

            for body_idx, force in emforce_dict.items():
                self.data.xfrc_applied[body_idx, :3] += force

            for rocket_bid, thrust in rocket_thrusts.items():
                self.data.xfrc_applied[rocket_bid, :3] += thrust

            mujoco.mj_step(self.model, self.data)
            mujoco.mj_rnePostConstraint(self.model, self.data)
            # for cacc to be meaningful, need to run mj_rnePostConstraint - https://github.com/google-deepmind/mujoco/issues/598#issuecomment-1605406993

            # Apply impulse-based COR correction AFTER mj_step
            apply_restitution_correction(coefficient_restitution, self.model, self.data, cor_state)

            self.data.qfrc_applied -= contact_force_torque

            for body_idx, force in gforce_dict.items():
                self.data.xfrc_applied[body_idx, :3] -= force

            for body_idx, force in emforce_dict.items():
                self.data.xfrc_applied[body_idx, :3] -= force

            for rocket_bid, thrust in rocket_thrusts.items():
                self.data.xfrc_applied[rocket_bid, :3] -= thrust

            num_steps += 1
            if num_steps > self.expected_steps + 50:
                stopped_due_to_instability = True
                print("Simulation stopped due to instability")
                break 
            all_data["global"]["time"].append(copy.deepcopy(self.data.time))
            all_data['global']['gravity'].append(self.model.opt.gravity[-1])

            # get sensor data
            for sensor_idx in range(self.model.nsensor):
                sensor_name = self.model.names[self.model.name_sensoradr[sensor_idx]:].split(b'\x00', 1)[0].decode("utf-8")
                sensor_i = self.data.sensor(sensor_idx)
                all_data["sensors"][sensor_name].append(copy.deepcopy(sensor_i.data))
            
            nv = self.data.cdof.shape[0]
            M_full = np.zeros((nv, nv))
            mujoco.mj_fullM(self.model, M_full, self.data.qM)  # Generalized inertia matrix
            
            # get body-specific data
            # if self.data.time > 1: st()
            for body_idx in range(self.model.nbody):
                # if self.model.body_jntnum[body_idx] == 0: continue
                
                if not self.has_joint_in_ancestry(body_idx): continue
                
                body_data = defaultdict(list)
                body_name = self.model.names[self.model.name_bodyadr[body_idx]:].split(b'\x00', 1)[0].decode("utf-8")

                Jl1, Jr1 = np.zeros((3, nv)), np.zeros((3, nv))
                mujoco.mj_jacBodyCom(self.model, self.data, Jl1, Jr1, body_idx)

                Jcom = np.concatenate((Jr1, Jl1), axis = 0)
                
                mass = self.model.body_mass[body_idx]
                # mass = self.get_recursive_body_mass(self.spec.body(body_name))

                if mass == 0:
                    continue
                
                # body_name = self.rename_body(body_name)
                g = copy.deepcopy(np.concatenate((np.zeros(3,), self.model.opt.gravity)))  # e.g. [0,0,0,0,0,-9.81]
                acceleration = copy.deepcopy(self.data.cacc[body_idx]) + g    # simulation acceleration does not include gravity
                gravcomp = copy.deepcopy(self.model.body_gravcomp[body_idx])
                position = copy.deepcopy(self.data.xpos[body_idx])
                com_position = copy.deepcopy(self.data.xipos[body_idx])
                com_velocity = copy.deepcopy(self.data.cvel[body_idx]) # ROT : LIN
                com_displacement = np.zeros(3) if len(all_data[body_name]["position"]) == 0 else (com_position - all_data[body_name]["position"][0])
                com_external_force = copy.deepcopy(self.data.cfrc_ext[body_idx])
                com_applied_force = copy.deepcopy(self.data.xfrc_applied[body_idx])
                com_inertia = Jcom @ M_full @ Jcom.T
                com_momentum = com_inertia @ com_velocity
                com_KE = 0.5 * com_momentum @ np.linalg.pinv(com_inertia) @ com_momentum
                com_KE_lin = 0.5 * mass * np.linalg.norm(com_velocity[3:]) ** 2
                com_KE_rot = com_KE - com_KE_lin
                # com_PE = mass * np.dot(com_displacement, self.model.opt.gravity)
                com_PE = 0
                em_PE = 0
                if len(all_data[body_name]["position"]):
                    r_cap = np.concatenate((np.zeros(3,), com_position))
                    r_cap_prev = np.concatenate((np.zeros(3,), all_data[body_name]["position"][0] + all_data[body_name]["com_offset"][0]))
                    g_net = copy.deepcopy(g)
                    g_net[3:] += gforce_dict.get(body_idx, 0) / mass
                    com_PE = -g_net @ (com_inertia @ r_cap - all_data[body_name]["inertia"][0] @ r_cap_prev)
                    em_PE = -np.dot(emforce_dict.get(body_idx, np.zeros(3)), com_velocity[3:] * self.cfg.dt) + all_data[body_name]["em_PE"][-1]
                    com_PE += em_PE

                net_F = com_inertia @ acceleration
                if len(all_data[body_name]["inertia"]): # F = m.dv/dt + v.dm/dt
                    net_F += ((com_inertia - all_data[body_name]["inertia"][-1])/self.cfg.dt) @ com_velocity

                all_data[body_name]["position"].append(position)
                all_data[body_name]["com_offset"].append(com_position - position)
                all_data[body_name]["velocity"].append(com_velocity)
                all_data[body_name]["velocity_linear"].append(com_velocity[3:])
                all_data[body_name]["velocity_angular"].append(com_velocity[:3])
                all_data[body_name]["com_external_force"].append(com_external_force)
                all_data[body_name]["com_applied_force"].append(com_applied_force)
                all_data[body_name]["mass"].append(mass)
                all_data[body_name]["acceleration"].append(acceleration)
                all_data[body_name]["acceleration_linear"].append(acceleration[3:])
                all_data[body_name]["acceleration_angular"].append(acceleration[:3])
                all_data[body_name]["net_force"].append(net_F)
                all_data[body_name]["net_force_linear"].append(net_F[3:])
                all_data[body_name]["net_torque"].append(net_F[3:])
                all_data[body_name]["gravcomp"].append(gravcomp)
                all_data[body_name]["displacement"].append(np.linalg.norm(com_displacement))   # displacement from initial position
                all_data[body_name]["momentum_linear"].append(com_momentum[3:])
                all_data[body_name]["momentum_angular"].append(com_momentum[:3])
                all_data[body_name]["kinetic_energy"].append(com_KE)
                all_data[body_name]["kinetic_energy_linear"].append(com_KE_lin)
                all_data[body_name]["kinetic_energy_angular"].append(com_KE_rot)
                all_data[body_name]["potential_energy"].append(com_PE)
                all_data[body_name]["inertia"].append(com_inertia)
                all_data[body_name]["inertia_z"].append(np.array([0, 0, 1, 0, 0, 0]) @ com_inertia @ np.array([0, 0, 1, 0, 0, 0]))
                all_data[body_name]["em_PE"].append(em_PE)

                if body_name in self.data_dict.keys():
                    for key, value in self.data_dict[body_name].items():
                        all_data[body_name][key].append(value)

            # get contact data
            active_pairs = []
            for con_idx in range(self.data.ncon):
                contact = self.data.contact[con_idx]
                geom_1, geom_2 = tuple(contact.geom)

                friction_frc = applied_friction_frc.get((min(geom_1, geom_2), max(geom_1, geom_2)), np.zeros(3,))
                normal_frc = applied_normal_frc.get((min(geom_1, geom_2), max(geom_1, geom_2)), 0)

                body_1, body_2 = self.model.geom_bodyid[geom_1], self.model.geom_bodyid[geom_2]
                
                # Reorder indices for fixed naming convention
                if body_1 > body_2: body_1, body_2 = body_2, body_1

                body_name_1 = self.model.names[self.model.name_bodyadr[body_1]:].split(b'\x00', 1)[0].decode("utf-8")
                
                # make sure we are not taking efc_addr = -1 case.
                contact_dir = copy.deepcopy(contact.frame[:3])
                contact_mag = copy.deepcopy(normal_frc) # self.data.efc_force[contact.efc_address]
                contact_force = contact_dir * contact_mag

                body_name_2 = self.model.names[self.model.name_bodyadr[body_2]:].split(b'\x00', 1)[0].decode("utf-8")
                
                # If contact for first time, fill 0s for all time before current timestep.
                if f"{body_name_1}_{body_name_2}" not in all_data['contact']:
                    all_data['contact'][f"{body_name_1}_{body_name_2}"] = [
                                        np.zeros(3) 
                                        for i in range(len(all_data['global']['time'])-1)
                                                                        ]
                if f"{body_name_1}_{body_name_2}" not in all_data['friction']:
                    all_data['friction'][f"{body_name_1}_{body_name_2}"] = [
                                        np.zeros(3) 
                                        for i in range(len(all_data['global']['time'])-1)
                                                                        ]

                all_data["contact"][f"{body_name_1}_{body_name_2}"].append(contact_force)
                all_data["friction"][f"{body_name_1}_{body_name_2}"].append(friction_frc)

                self.update_metainfo(source=body_name_1, target=body_name_2)
                self.update_metainfo(source=body_name_2, target=body_name_1)         

                active_pairs.append(f'{body_name_1}_{body_name_2}')

            # Add zero Normal force if not currently in contact
            for pair in all_data['contact']:
                if pair not in active_pairs:
                    all_data['contact'][pair].append(np.zeros(3))
            
            passive_tendon_forces = (
                self.model.tendon_stiffness * (self.data.ten_length - self.model.tendon_lengthspring[..., 0])
                + self.model.tendon_damping * self.data.ten_velocity
            )

            tendon_forces = np.where(
                self.actuated_tendons,
                np.bincount(self.actuated_tendon_ids, weights=-self.data.actuator_force[self.tendon_actuators], minlength=self.model.ntendon),
                passive_tendon_forces
            )

            # get tendon data
            for tendon_idx in range(self.model.ntendon):
                start_index = self.model.tendon_adr[tendon_idx]
                # Number of path elements for this tendon
                num_elements = self.model.tendon_num[tendon_idx]

                tendon_name = self.model.names[self.model.name_tendonadr[tendon_idx]:].split(b'\x00', 1)[0].decode("utf-8")
                # tendon_name = self.rename_body(tendon_name)
                
                efc_address = self.data.tendon_efcadr[tendon_idx]
                if efc_address != -1:
                    tendon_force = copy.deepcopy(self.data.efc_force[efc_address])
                else:
                    tendon_force = copy.deepcopy(tendon_forces[tendon_idx])

                for i in range(start_index, start_index + num_elements - 1):
                    wrap_id = self.model.wrap_objid[i]
                    wrap_id_next = self.model.wrap_objid[i+1]
                    wrap_type = mjWRAP_CONSTANTS[self.model.wrap_type[i]]
                    wrap_type_next = mjWRAP_CONSTANTS[self.model.wrap_type[i+1]]
                    wrap_name, body_name, position = self.get_info_from_wrap(wrap_id, wrap_type)
                    # body_name = self.rename_body(body_name)
                    wrap_name_next, body_name_next, position_next = self.get_info_from_wrap(wrap_id_next, wrap_type_next)
                    # body_name_next = self.rename_body(body_name_next)
                    angle = self.get_angle(position,  position_next)    
                    all_data[tendon_name][f"Angle_{body_name}_{body_name_next}"].append(angle)
                    if i == start_index:
                        self.update_metainfo(source=body_name, target=tendon_name)
                    if i+1 == start_index + num_elements - 1:
                        self.update_metainfo(source=body_name_next, target=tendon_name)
                tendon_data = self.data.tendon(tendon_idx)
                tendon_length = copy.deepcopy(tendon_data.length)
                tendon_velocity = copy.deepcopy(tendon_data.velocity)
                if len(tendon_length.shape) > 1 or len(tendon_velocity.shape) > 1:
                    print("debug when tendon_length or tendon_velocity is not 1D")
                tendon_length = tendon_length[0]
                tendon_velocity = tendon_velocity[0]
                all_data[tendon_name]["length"].append(tendon_length)
                all_data[tendon_name]["velocity"].append(tendon_velocity)
                all_data[tendon_name]["force"].append(tendon_force)
                all_data[tendon_name]["stiffness"].append(copy.deepcopy(self.model.tendon_stiffness[tendon_idx]))

            if self.cfg.render and len(self.frames) < self.data.time * self.cfg.fps:
                                # orbit camera function
                if self.cfg.orbit_camera:
                    (distance, azimuth, elevation, lookat) = self.cam_motion(self.cfg.get("body_id_to_track", None))
                    
                    # Check if there's specific focus body and distance
                    focus_body_id, focus_distance = self.get_focus_body_id()
                    has_specific_focus = focus_body_id is not None or self.cfg.get("body_id_to_track", None) is not None
                    
                    if has_specific_focus:
                        # Use fixed distance when there's specific focus
                        self.cam.distance = distance
                    else:
                        # Use adaptive distance when no specific focus (if enabled)
                        if getattr(self.cfg, 'adaptive_camera_distance', False):
                            adaptive_distance = self.calculate_adaptive_distance()
                            # optional: smooth transition, avoid sudden distance change
                            if hasattr(self, '_last_adaptive_distance'):
                                smoothing_factor = 0.95  # smoothing factor, the closer to 1, the smoother the change
                                adaptive_distance = (self._last_adaptive_distance * smoothing_factor + 
                                                   adaptive_distance * (1 - smoothing_factor))
                            self._last_adaptive_distance = adaptive_distance
                            self.cam.distance = adaptive_distance
                        else:
                            # If adaptive distance not enabled, use orbit camera distance
                            self.cam.distance = distance
                    
                    self.cam.azimuth = azimuth
                    self.cam.elevation = elevation
                    self.cam.lookat = lookat
                elif getattr(self.cfg, 'adaptive_camera_distance', False):
                    # If no orbit camera but adaptive distance enabled
                    adaptive_distance = self.calculate_adaptive_distance()
                    self.cam.distance = adaptive_distance
                    # optional: smooth transition, avoid sudden distance change
                    if hasattr(self, '_last_adaptive_distance'):
                        smoothing_factor = 0.95  # smoothing factor, the closer to 1, the smoother the change
                        self.cam.distance = (self._last_adaptive_distance * smoothing_factor + 
                                           adaptive_distance * (1 - smoothing_factor))
                    self._last_adaptive_distance = self.cam.distance
                
                # update rendering scene
                if self.cfg.custom_camera or self.cfg.orbit_camera or getattr(self.cfg, 'adaptive_camera_distance', False):
                    self.renderer.update_scene(self.data, camera=self.cam)    
                else:
                    self.renderer.update_scene(self.data)
                
                if not self.cfg.disable_trail:
                    trail_bodies = self.scene.get_trail_bodies()
                    rockets = self.scene.get_rockets()
                    
                    for i, (trail_body, max_len) in enumerate(trail_bodies):
                        idx = None
                        is_rocket = trail_body in rockets
                        radius = estimate_trail_radius_from_geom(self.model, trail_body + ".geom" if not is_rocket else ".collision_geom")
                        if is_rocket:
                            mass_diff = np.diff(all_data[trail_body + ".collision_geom"]["mass"])
                            mass_change_idx = np.nonzero(mass_diff)[0]
                            idx = mass_change_idx[-1] if len(mass_change_idx) > 0 else len(all_data[trail_body + ".collision_geom"]["mass"]) - 1
                        
                        # Adjust radius: make larger bodies have thinner trails, smaller bodies keep normal thickness
                        if is_rocket:
                            adjusted_radius = radius  # Rockets keep normal thickness
                        else:
                            # For other bodies, make large ones thinner
                            adjusted_radius = radius * 0.3  # Make trails much thinner for large bodies
                        
                        # Get the body color for stellar trail effect
                        body_color = get_body_color(self.model, trail_body)
                        
                        draw_trails(
                            self.renderer.scene, 
                            all_data[trail_body + ("" if not is_rocket else f".collision_geom")]["position"], 
                            all_data[trail_body + ("" if not is_rocket else ".collision_geom")]["velocity_linear"], 
                            all_data["global"]["time"], 
                            get_body_forward_vector(self.model, self.data, trail_body), 
                            idx,
                            max_len,
                            adjusted_radius,
                            body_color,
                            trail_body  # Pass body_name for consistent color selection
                        )

                        draw_regions(
                            self.model,
                            self.data,
                            self.renderer.scene,
                            self.scene.get_EM_configs(),
                        )
                
                # for rocket in self.scene.get_rockets():
                #     mass_diff = np.diff(all_data[rocket + ".collision_geom"]["mass"])
                #     mass_change_idx = np.nonzero(mass_diff)[0]
                #     idx = mass_change_idx[-1] if len(mass_change_idx) > 0 else len(all_data[rocket + ".collision_geom"]["mass"]) - 1
                #     draw_trails(
                #         self.renderer.scene, 
                #         all_data[rocket + ".collision_geom"]["position"], 
                #         all_data[rocket + ".collision_geom"]["velocity_linear"], 
                #         all_data["global"]["time"], 
                #         get_body_forward_vector(self.model, self.data, rocket), 
                #         idx
                #     )
                
                # Set far plane for camera
                for cam in self.renderer.scene.camera:
                    cam.frustum_far = max(100, cam.frustum_far)

                pixels = self.renderer.render()
                self.frames.append(pixels)       

        if self.cfg.prune_timesteps:
            keep_index_min, keep_index_max = self.prune_timesteps(all_data)
            
            if keep_index_min is None or keep_index_min == keep_index_max: # No valid range
                print("Simulation unstable. Cannot use Data recorded.")
                keep_index_min = keep_index_max = 0
                stopped_due_to_instability = True
            
            keep_range = (keep_index_min, keep_index_max)
        else:
            keep_range = (0, len(all_data["global"]["time"]) - 1)

        # save frames
        if self.cfg.render:
            os.makedirs(os.path.join(self.scene_folder), exist_ok=True)
            suf = f"_d{self._cam_cfg['distance']}_e{self._cam_cfg['elevation']}_a{self._cam_cfg['azimuth']}"
            imageio.mimsave(
                os.path.join(self.scene_folder, f"{self.scene.name}{suf}.mp4"),
                self.frames, fps=self.cfg.fps
            )
            filename = f"{self.scene.name}{suf}.mp4"
            print(f"[Recorder] Saved render video to: {os.path.join(self.scene_folder, filename)}")
        
        # # Plot all the data
        if self.cfg.plot_data:
            self.plot_data(all_data, self.variation_index, keep_range)
            print(f"[Recorder] Saved plots to: {os.path.join(self.scene_folder, f'plots_{self.scene.name}' + str(self.variation_index))}")
        
        if self.cfg.prune_timesteps:
            for key, value in all_data.items():
                all_data[key] = {sub_key: sub_value if len(sub_value) == 1 else sub_value[keep_index_min:keep_index_max] for sub_key, sub_value in value.items()}
        
        return all_data, self.metainfo, stopped_due_to_instability

    def plot_data(self, all_data, variation_index, keep_range=None):
        # Create plots directory if it doesn't exist
        plots_folder_name = f'plots_{self.scene.name}_{variation_index}'
        os.makedirs(os.path.join(self.scene_folder, plots_folder_name), exist_ok=True)
        # Plot data for each body/sensor
        for name, data_dict in all_data.items():
            if name == "global":
                continue
            # Create subplots for each data type
            num_plots = len(data_dict)
            # Calculate grid dimensions to make it roughly square
            n_cols = max(1, int(np.ceil(np.sqrt(num_plots))))
            n_rows = max(1, int(np.ceil(num_plots / max(n_cols, 1))))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), dpi=250, constrained_layout=True)
            # Flatten axes array to make indexing easier
            axes = axes.flatten() if num_plots > 1 else [axes]
                
            for ax_idx, (data_type, values) in enumerate(data_dict.items()):
                values = np.array(values)
                times = np.array(all_data["global"]["time"])
                ax = axes[ax_idx]
                
                # If data has multiple dimensions, plot each dimension
                if values.ndim > 2:
                    values = values.reshape(values.shape[0], -1)
                
                if len(values.shape) > 1 and values.shape[1] > 1:
                    if values.shape[1] == 6:
                        # Plot rotation XYZ and translation XYZ
                        for dim in range(3):
                            mean_value = np.mean(values[:, dim])
                            std_value = np.std(values[:, dim])
                            ax.plot(times, values[:, dim], label=f'Rotation {["X", "Y", "Z"][dim]} (mean: {mean_value:.1f}, std: {std_value:.1f})')
                            # Add annotations for rotation dimensions
                            self.add_annotations(ax, times, values, dim)
                                            
                        for dim in range(3,6):
                            mean_value = np.mean(values[:, dim])
                            std_value = np.std(values[:, dim])
                            ax.plot(times, values[:, dim],
                                    label=f'Translation {["X", "Y", "Z"][dim-3]} (mean: {mean_value:.1f}, std: {std_value:.1f})')
                            # Add annotations for translation dimensions
                            self.add_annotations(ax, times, values, dim)
                                            
                    elif values.shape[1] == 3:
                        # Plot translation XYZ
                        for dim in range(3):
                            mean_value = np.mean(values[:, dim])
                            std_value = np.std(values[:, dim])
                            minlen = min(len(times), len(values[:, dim]))

                            ax.plot(times[:minlen], values[:minlen, dim],
                                    label=f'Translation {["X", "Y", "Z"][dim]} (mean: {mean_value:.1f}, std: {std_value:.1f})')
                            # Add annotations for each dimension
                            self.add_annotations(ax, times[:minlen], values[:minlen], dim)
                    elif values.shape[1] == 36:
                        # Plot 6x6 matrix
                        for dim2 in range(6):
                            for dim1 in range(6):
                                dim = dim1 + dim2 * 6
                                mean_value = np.mean(values[:, dim])
                                std_value = np.std(values[:, dim])
                                ax.plot(times, values[:, dim], label=f'Row: {dim1+1} Column: {dim2+1} (mean: {mean_value:.1f}, std: {std_value:.1f})')
                                # Add annotations for each dimension
                                self.add_annotations(ax, times, values, dim)
                    else:
                        raise ValueError(f"Unsupported data shape: {values.shape}")
                    
                else:
                    ax.plot(times, values)
                    # Add annotations for 1D data
                    self.add_annotations(ax, times, values)
                
                # Add vertical line for keep range 
                if keep_range is not None:
                    keep_time_min, keep_time_max = keep_range
                    ax.axvspan(times[keep_time_min], times[keep_time_max], alpha=0.2, color='g', label='Keep Time Range')
                    ax.axvline(x=times[keep_time_min], color='g', linestyle='--', label='Keep Time Min')
                    ax.axvline(x=times[keep_time_max], color='g', linestyle='--', label='Keep Time Max')
                ax.legend()
                ax.set_title(f'{data_type}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.grid(True)

            # plt.tight_layout(pad=1.5)
            # Save combined plot
            print(f"[Recorder] saving plot for {name}")
            plt.savefig(os.path.join(self.scene_folder, plots_folder_name, f'{name}.png'))
            plt.close()
    
    def add_annotations(self, ax, times, values, dim=None):
            """Add percentage annotations to the plot at specific time points.
            
            Args:
                times: Array of time values
                values: Array of data values
                dim: Optional dimension index for multi-dimensional data
            """
            indices = [int(len(times) * p) for p in [0.25, 0.5, 0.75, 0.99]]
            for i, idx in enumerate(indices):
                percentage = [25, 50, 75, 100][i]
                value = values[idx] if dim is None else values[idx, dim]
                time = times[idx]
                ax.plot(time, value, 'o', color='black', markersize=6)
                ax.annotate(f"{percentage}% - {value:.5f}", (time, value), xytext=(5, 5),
                                textcoords='offset points', 
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))   

    def get_focus_body_id(self):
        """
        Determine camera focus body_id based on scene type and entity type
        Uses two-level priority:
        1. Entity-level priority: select among multiple entities in scene
        2. Body-level priority: select among bodies within chosen entity
        
        Returns:
        - body_id (int): focus body id, None if using scene center
        - focus_distance (float): suggested focus distance, None if using adaptive distance
        """
        scene_tag = self.scene.tag
        if scene_tag == "Visualization_1":
            return 3, 5
        elif scene_tag == "Visualization_2":
            return 1, 5
        elif scene_tag == "Visualization_3":
            return 6, 5
        
        # Check if smart focus is enabled
        if not getattr(self.cfg, 'enable_smart_focus', True):
            return None, None
        
        # Check if scene should focus on specific body
        should_focus_on_mass = False
        for category_scenes in SCENE_FOCUS_CONFIG["focus_on_mass"].values():
            if scene_tag in category_scenes:
                should_focus_on_mass = True
                break
        
        # If scene type should use scene center, return directly
        if not should_focus_on_mass:
            for category_scenes in SCENE_FOCUS_CONFIG["focus_on_center"].values():
                if scene_tag in category_scenes:
                    if getattr(self.cfg, 'debug_focus_camera', False):
                        print(f"[Focus Camera] Scene {scene_tag} configured for center focus")
                    return None, None
        
        # Collect all body names and ID mappings
        body_name_to_id = {}
        for body_idx in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_idx)
            if body_name:
                body_name_to_id[body_name] = body_idx
        
        # Step 1: Group all bodies by entity type
        entity_bodies = {}  # {entity_type: [(body_name, body_id), ...]}
        
        for body_name, body_id in body_name_to_id.items():
            if not body_name or "worldbody" in body_name.lower():
                continue
                
            # Parse entity type from body name
            entity_type = self._infer_entity_type_from_body_name(body_name)
            if entity_type:
                if entity_type not in entity_bodies:
                    entity_bodies[entity_type] = []
                entity_bodies[entity_type].append((body_name, body_id))
        
        # Step 2: Select highest priority entity type based on entity priority
        if not entity_bodies:
            if getattr(self.cfg, 'debug_focus_camera', False):
                print(f"[Focus Camera] No recognizable entities found in scene {scene_tag}")
            return None, None
        
        # Sort by entity priority
        sorted_entities = sorted(entity_bodies.items(), 
                               key=lambda x: ENTITY_PRIORITY_CONFIG.get(x[0], ENTITY_PRIORITY_CONFIG["default"]))
        selected_entity_type, entity_body_list = sorted_entities[0]
        
        if getattr(self.cfg, 'debug_focus_camera', False):
            print(f"[Focus Camera] Selected entity type: {selected_entity_type} from {list(entity_bodies.keys())}")
        
        # Step 3: Within selected entity, select specific body by body priority
        focus_candidates = []
        
        for body_name, body_id in entity_body_list:
            body_priority = self._get_body_priority_within_entity(body_name, selected_entity_type)
            if body_priority is not None:
                focus_candidates.append((body_priority, body_id, body_name))
        
        # If found candidates, select highest priority
        if focus_candidates:
            focus_candidates.sort(key=lambda x: x[0])  # Sort by priority
            selected_priority, selected_body_id, selected_body_name = focus_candidates[0]
            
            # Get focus distance with config override support
            focus_body_overrides = getattr(self.cfg, 'focus_body_params', {})
            focus_distance = (focus_body_overrides.get("fixed_distance") or 
                            ADAPTIVE_CAMERA_PARAMS["focus_body"]["fixed_distance"])
            
            if getattr(self.cfg, 'debug_focus_camera', False):
                print(f"[Focus Camera] Selected focus body: {selected_body_name} (ID: {selected_body_id}) "
                      f"from entity {selected_entity_type} with distance: {focus_distance}")
            
            return selected_body_id, focus_distance
        
        # If no suitable focus found, use scene center
        if getattr(self.cfg, 'debug_focus_camera', False):
            print(f"[Focus Camera] No suitable focus body found in entity {selected_entity_type}, using scene center")
        return None, None
    
    def _infer_entity_type_from_body_name(self, body_name: str) -> str:
        """
        从body名称推断entity类型
        body名称格式通常是：entity_name.body_part 或 entity_name_index.body_part
        """
        # 解析body名称，提取entity相关信息
        parts = body_name.split('.')
        if len(parts) < 2:
            return None
            
        entity_part = parts[0].lower()
        
        # 根据entity命名模式匹配entity类型
        if 'massprismplane' in entity_part:
            return "MassPrismPlaneEntity"
        elif 'stackedmassplane' in entity_part:
            return "StackedMassPlane"
        elif 'twosidemassplane' in entity_part:
            return "TwoSideMassPlane"
        elif 'directedmass' in entity_part:
            return "DirectedMass"
        elif 'complexcollisionplane' in entity_part:
            return "ComplexCollisionPlane"
        elif 'twodcollisionplane' in entity_part:
            return "TwoDCollisionPlane"
        elif 'springmassplane' in entity_part:
            return "SpringMassPlaneEntity"
        elif 'springblock' in entity_part:
            return "SpringBlockEntity"
        elif 'massboxplane' in entity_part:
            return "MassBoxPlaneEntity"
        elif 'massprism' in entity_part and 'pulley' in entity_part:
            return "MassPrismPulleyPlane"
        elif 'masswithmovablepulley' in entity_part:
            return "MassWithMovablePulley"
        elif 'masswithreversedmovablepulley' in entity_part:
            return "MassWithReversedMovablePulley"
        elif 'masswithfixedpulley' in entity_part:
            return "MassWithFixedPulley"
        elif 'constantforcefixedpulley' in entity_part:
            return "ConstantForceFixedPulley"
        elif 'fixedpulley' in entity_part:
            return "FixedPulleyEntity"
        
        return None
    
    def _get_body_priority_within_entity(self, body_name: str, entity_type: str) -> int:
        """
        Determine body priority within given entity type
        """
        parts = body_name.split('.')
        if len(parts) < 2:
            return None
            
        body_part = '.'.join(parts[1:]).lower()
        
        # Determine priority based on entity type and body type
        if entity_type in ["ComplexCollisionPlane", "TwoDCollisionPlane"]:
            # Body priority for collision entities
            if 'spring_mass' in body_part or 'spring-mass' in body_part:
                if 'mass-' in body_part:
                    return 1  # Individual mass in spring-mass system, highest priority
                else:
                    return 2  # Spring-mass container
            elif body_part.startswith('mass-') and 'fixed' not in body_part:
                return 3  # Independent mass body
            elif 'sphere' in body_part:
                return 4  # Sphere
            # Ignore fixed_mass and fixed_spring
        
        elif entity_type in ["MassPrismPlaneEntity", "StackedMassPlane", "TwoSideMassPlane", 
                           "DirectedMass", "MassBoxPlaneEntity", "MassPrismPulleyPlane"]:
            # Body priority for mass plane entities
            if 'mass' in body_part and 'fixed' not in body_part:
                if 'mass-' in body_part:
                    return 1  # Individual mass
                else:
                    return 2  # Mass container
        
        elif entity_type in ["SpringMassPlaneEntity", "SpringBlockEntity"]:
            # Body priority for spring entities
            if 'mass-' in body_part:
                return 1  # Mass in spring system
            elif 'mass' in body_part and 'fixed' not in body_part:
                return 2  # Other mass
        
        elif entity_type in ["MassWithMovablePulley", "MassWithReversedMovablePulley", 
                           "MassWithFixedPulley", "ConstantForceFixedPulley"]:
            # Body priority for pulley entities
            if 'mass' in body_part and 'fixed' not in body_part:
                return 1  # Mass part
        
        return None  # Don't care about this body type

    def cam_motion(self, body_id = None):
        """Return sigmoidally-mixed {orbit, box-track} trajectory."""
        # Auto-detect focus body
        focus_body_id, focus_distance = self.get_focus_body_id()
        
        # If body_id_to_track is specified in config, use it with priority
        if body_id is not None:
            focus_body_id = body_id
            focus_distance = 5.0  # Use fixed distance when manually specified
        
        return self.orbit_motion(self.data.time / self.cfg.duration, focus_body_id, focus_distance)

    def calculate_adaptive_distance(self):
        """
        Calculate adaptive camera distance based on scene object distribution
        Considers actual geometric sizes of objects, not just center positions
        Returns recommended camera distance value
        """
        base_distance = self._cam_cfg["distance"]
        
        # Get corresponding parameter group based on scene type
        scene_tag = self.scene.tag
        param_key = SCENE_TO_PARAMS_MAP.get(scene_tag, "scene_center_medium")
        params = ADAPTIVE_CAMERA_PARAMS.get(param_key, ADAPTIVE_CAMERA_PARAMS["scene_center_medium"])
        
        # Get config parameters, priority: specific param group config > global config override > default param group values
        param_override_key = f"{param_key}_params"
        param_overrides = getattr(self.cfg, param_override_key, {})
        
        safety_margin = (param_overrides.get("safety_margin") or 
                        getattr(self.cfg, 'adaptive_camera_safety_margin', None) or 
                        params["safety_margin"])
        min_distance_factor = (param_overrides.get("min_factor") or 
                              getattr(self.cfg, 'adaptive_camera_min_factor', None) or 
                              params["min_factor"])
        max_distance_factor = (param_overrides.get("max_factor") or 
                              getattr(self.cfg, 'adaptive_camera_max_factor', None) or 
                              params["max_factor"])
        fov_degrees = (param_overrides.get("fov") or 
                      getattr(self.cfg, 'adaptive_camera_fov', None) or 
                      params["fov"])
        
        # Collect all active objects' positions and sizes
        object_bounds = []  # Store bounding box information for each object
        
        for body_idx in range(self.model.nbody):
            if self.has_joint_in_ancestry(body_idx):
                body_pos = self.data.xpos[body_idx].copy()
                
                # Find all geom associated with this body
                body_geom_ids = []
                for geom_idx in range(self.model.ngeom):
                    if self.model.geom_bodyid[geom_idx] == body_idx:
                        body_geom_ids.append(geom_idx)
                
                # If there are associated geom, calculate their actual bounds
                if body_geom_ids:
                    # Calculate combined bounding box for all geom
                    body_min_bounds = body_pos.copy()
                    body_max_bounds = body_pos.copy()
                    
                    for geom_idx in body_geom_ids:
                        geom_pos = self.data.geom_xpos[geom_idx]  # Position of geom in world coordinates
                        geom_size = self.model.geom_size[geom_idx]  # Size of geom
                        geom_type = self.model.geom_type[geom_idx]  # Type of geom
                        
                        # Calculate actual bounds based on geom type
                        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                            # For boxes, size includes half width, half height, half depth
                            half_extents = geom_size[:3]
                            geom_min = geom_pos - half_extents
                            geom_max = geom_pos + half_extents
                        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                            # For spheres, size[0] is radius
                            radius = geom_size[0]
                            geom_min = geom_pos - radius
                            geom_max = geom_pos + radius
                        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                            # For cylinders, size[0] is radius, size[1] is half height
                            radius = geom_size[0]
                            half_height = geom_size[1]
                            geom_min = geom_pos - np.array([radius, radius, half_height])
                            geom_max = geom_pos + np.array([radius, radius, half_height])
                        elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                            # For planes, use default size defined in constants
                            from sim.constants import DEFAULT_PLANE_LENGTH, DEFAULT_PLANE_WIDTH, DEFAULT_PLANE_THICKNESS
                            plane_half_length = DEFAULT_PLANE_LENGTH / 2
                            plane_half_width = DEFAULT_PLANE_WIDTH / 2
                            plane_half_thickness = DEFAULT_PLANE_THICKNESS / 2
                            geom_min = geom_pos - np.array([plane_half_length, plane_half_width, plane_half_thickness])
                            geom_max = geom_pos + np.array([plane_half_length, plane_half_width, plane_half_thickness])
                        else:
                            # For other types, use average size as approximation
                            avg_size = np.mean(geom_size) if len(geom_size) > 0 else 0.1
                            geom_min = geom_pos - avg_size
                            geom_max = geom_pos + avg_size
                        
                        # Update total bounding box for the body
                        body_min_bounds = np.minimum(body_min_bounds, geom_min)
                        body_max_bounds = np.maximum(body_max_bounds, geom_max)
                    
                    object_bounds.append((body_min_bounds, body_max_bounds))
                else:
                    # If no associated geom, use center point (backward compatibility)
                    object_bounds.append((body_pos, body_pos))
        
        if len(object_bounds) < 1:
            return base_distance
        
        # Calculate 3D bounding box for the entire scene
        if len(object_bounds) == 1:
            scene_min_bounds, scene_max_bounds = object_bounds[0]
        else:
            all_min_bounds = np.array([bounds[0] for bounds in object_bounds])
            all_max_bounds = np.array([bounds[1] for bounds in object_bounds])
            scene_min_bounds = np.min(all_min_bounds, axis=0)
            scene_max_bounds = np.max(all_max_bounds, axis=0)
        
        scene_size = scene_max_bounds - scene_min_bounds
        
        # Calculate maximum extent of the scene
        max_extent = np.max(scene_size)
        
        # Calculate center of the scene
        scene_center = (scene_min_bounds + scene_max_bounds) / 2
        
        # Calculate maximum distance from scene center to any object boundary
        max_center_distance = 0
        for min_bounds, max_bounds in object_bounds:
            # Calculate distance from 8 corners of bounding box to scene center
            corners = np.array([
                [min_bounds[0], min_bounds[1], min_bounds[2]],
                [min_bounds[0], min_bounds[1], max_bounds[2]],
                [min_bounds[0], max_bounds[1], min_bounds[2]],
                [min_bounds[0], max_bounds[1], max_bounds[2]],
                [max_bounds[0], min_bounds[1], min_bounds[2]],
                [max_bounds[0], min_bounds[1], max_bounds[2]],
                [max_bounds[0], max_bounds[1], min_bounds[2]],
                [max_bounds[0], max_bounds[1], max_bounds[2]],
            ])
            corner_distances = np.linalg.norm(corners - scene_center, axis=1)
            max_center_distance = max(max_center_distance, np.max(corner_distances))
        
        # Consider scene size and object dispersion
        scene_radius = max(max_extent / 2, max_center_distance)
        
        # Calculate recommended distance: ensure entire scene is visible, with some margin
        # Use field of view angle to calculate appropriate distance
        fov_factor = 1.0 / np.tan(np.radians(fov_degrees / 2))
        recommended_distance = scene_radius * fov_factor * safety_margin
        
        # Limit range of distance change, avoid sudden changes
        min_distance = base_distance * min_distance_factor
        max_distance = base_distance * max_distance_factor
        
        adaptive_distance = np.clip(recommended_distance, min_distance, max_distance)
        
        # Optional: output debug information (only in debug mode)
        if getattr(self.cfg, 'debug_adaptive_camera', False):
            if int(self.data.time * 10) % 10 == 0:  # Output every 0.1 seconds
                print(f"[Adaptive Camera] t={self.data.time:.1f}s, "
                      f"scene={scene_tag}, param_set={param_key}, "
                      f"active_bodies={len(object_bounds)}, "
                      f"scene_size={scene_size}, "
                      f"scene_radius={scene_radius:.2f}, "
                      f"safety_margin={safety_margin}, "
                      f"recommended={recommended_distance:.2f}, "
                      f"final={adaptive_distance:.2f}")
        
        return adaptive_distance

    def orbit_motion(self, t: float, body_id: int, focus_distance: float = None):
        # if body_id is None: body_id = -1 
        
        # Decide which distance to use
        if focus_distance is not None:
            # Use fixed focus distance
            distance = focus_distance
        else:
            # Use original distance from config (will be overridden by adaptive distance if enabled)
            distance = self._cam_cfg["distance"]
        
        azimuth = self._cam_cfg["azimuth"] + 100 * unit_cos(t)
        elevation = self._cam_cfg["elevation"] 
        lookat = self.scene_center if body_id is None else self.data.xpos[body_id].copy()
        return distance, azimuth, elevation, lookat

@hydra.main(config_path="../config", config_name="config")
def run(cfg: DictConfig):
    cfg = cfg.recorder

    if hasattr(cfg, 'scene_path'):
        scene_path = cfg.scene_path
        scene_folder = f'runs/' + scene_path.split('/')[-1] + r'/'
        scene = parse_scene(scene_path)
    elif hasattr(cfg, 'xml_path'):
        raise NotImplementedError(f'Current version of recorder requires scene_parameters, which cannot be created without the simDSL file.')
    elif hasattr(cfg, 'simDSL_path'):
        scene = parse_scene(cfg.simDSL_path)
        scene_folder = os.path.dirname(cfg.simDSL_path)
    else:
        scene_path = 'sim/DSLs/textbook_questions/P79_Q22/question.yaml'
        scene_path = 'sim/DSLs/complex_collision_plane.yaml'
        
        scene_folder = os.path.dirname(scene_path)
        scene = parse_scene(scene_path)
    
    scene_type = scene.tag
    category = [k for k in SCENE_TYPE_TO_CATEGORY_MAP if scene_type in SCENE_TYPE_TO_CATEGORY_MAP[k]]
    if len(category) == 1:
        category = category[0]
    else: category = None

    recorder = Recorder(scene, cfg, scene_folder, category=category)
    print(f"[Recorder Init] Scene folder set to: {scene_folder}")
    data, metadata, instability = recorder.simulate()


def _process_scene(scene_path: str,
                   cfg: DictConfig,
                   combos: list | None = None):
    """
    Returns (scene_path, ok:bool, errmsg:str|None)
    """
    try:
        scene_folder = os.path.dirname(scene_path)
        scene        = parse_scene(scene_path)
        scene_type   = scene.tag
        category     = next((k for k in SCENE_TYPE_TO_CATEGORY_MAP
                             if scene_type in SCENE_TYPE_TO_CATEGORY_MAP[k]), None)

        # Batch mode
        if combos is None:
            recorder = Recorder(scene, cfg, scene_folder, category=category)
            recorder.simulate()
            logger.info("[Batch] Finished %s", scene.name)

        # Camera Sweep mode
        else:
            for dist, elev, azim in combos:
                run_cfg              = copy.deepcopy(cfg)
                run_cfg.custom_camera = True
                run_cfg.distance      = dist
                run_cfg.elevation     = elev
                run_cfg.azimuth       = azim

                Recorder(scene, run_cfg, scene_folder, category=category).simulate()
                logger.info("[Sweep] %s d%s e%s a%s finished",
                            scene.name, dist, elev, azim)
        return scene_path, True, None

    except Exception as e:
        err_msg = traceback.format_exc()
        with open("error.log", "a") as f:
            f.write(err_msg + "\n")
        logger.error("[Error] %s\n%s", scene_path, err_msg)
        return scene_path, False, err_msg

@hydra.main(config_path="../config", config_name="config")
def run_batch(cfg: DictConfig):
    cfg  = cfg.recorder
    ymls = glob.glob(os.path.join(cfg.input_dir, "**", "*.yaml"), recursive=True)
    ymls = [p for p in ymls if p.endswith("scene_output.yaml")]

    parallel    = getattr(cfg, "parallel", False)
    num_workers = getattr(cfg, "num_workers", multiprocessing.cpu_count())

    print(f"[Batch] Found {len(ymls)} YAML files. parallel={parallel}")

    if parallel:
        with ProcessPoolExecutor(max_workers=num_workers) as exe:
            futures = {exe.submit(_process_scene, p, cfg): p for p in ymls}
            for _ in tqdm(as_completed(futures), total=len(futures),
                          desc="Batch Render"):
                pass  # Just consume futures to drive progress bar
    else:
        for p in tqdm(ymls, desc="Batch Render"):
            _process_scene(p, cfg)


@hydra.main(config_path="../config", config_name="config")
def run_camera_sweep(cfg: DictConfig):
    base_cfg = cfg.recorder
    ymls     = glob.glob(os.path.join(base_cfg.input_dir,
                                      "**", "*.yaml"), recursive=True)
    ymls = [p for p in ymls if p.endswith("scene_output.yaml")]

    distances  = [5, 7, 10, 12, 15]
    elevations = [-30, -60, -90]
    azimuths   = [0, 45, 90]
    combos     = list(itertools.product(distances, elevations, azimuths))

    parallel    = getattr(base_cfg, "parallel", False)
    num_workers = getattr(base_cfg, "num_workers", multiprocessing.cpu_count())

    print(f"[Sweep] Total scenes: {len(ymls)} × {len(combos)} combos"
          f"  parallel={parallel}")

    if parallel:
        args = ((p, base_cfg, combos) for p in ymls)
        with ProcessPoolExecutor(max_workers=num_workers) as exe:
            futures = {exe.submit(_process_scene, *arg): arg[0] for arg in args}
            for _ in tqdm(as_completed(futures), total=len(futures),
                          desc="Camera Sweep"):
                pass
    else:
        for p in tqdm(ymls, desc="Camera Sweep"):
            _process_scene(p, base_cfg, combos=combos)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    if cfg.recorder.mode == "batch":
        run_batch(cfg)
    elif cfg.recorder.mode == "sweep":
        run_camera_sweep(cfg)
    elif cfg.recorder.mode == "normal":
        run(cfg)
    else: raise ValueError(f"Unknown mode: {cfg.recorder.mode}")


if __name__ == "__main__":
    main()
