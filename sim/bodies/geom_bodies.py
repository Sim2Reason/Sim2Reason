from .base_bodies import *
from sim.mesh_utils import *

class Sphere(Body):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        radius: float = DEFAULT_SPHERE_RADIUS,
        material: str = DEFAULT_MATERIAL,
        rgba: Tuple[float, float, float, float] = DEFAULT_RGBA,
        mass: float = 1.0,
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        init_velocity: Optional[Dict[str, List[Union[List, float]]]] = None,
        joint_option=("free", (0, 0, 0)),
        constant_force: Optional[Dict[str, List[Union[List, float]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(name, pos, quat, **kwargs)
        self.add_geom(
            Geom(
                name=f"{name}.geom",
                geom_type="sphere",
                pos=(0, 0, 0),
                size=(radius,),
                material=material,
                rgba=rgba,
                mass=mass,
            )
        )
        self.site = Site(f"{name}.site", (0, 0, 0), body_name=name)
        self.body_type = "sphere"
        self.add_site(self.site)
        if joint_option is not None: self.add_joint(Joint(joint_option[0], joint_option[1], f"{name}.joint"))
        self.center_site = Site(f"{name}.center_site", (0, 0, 0), body_name=name)
        self.add_site(self.center_site)

        # set initial velocity
        if (
            init_velocity is not None
            and init_velocity.get(InitVelocityType.SPHERE) is not None
        ):
            if isinstance(init_velocity[InitVelocityType.SPHERE][0], list):
                init_velocity[InitVelocityType.SPHERE] = init_velocity[
                    InitVelocityType.SPHERE
                ][0]  # multiple velocities choose the first one
            else:
                init_velocity[InitVelocityType.SPHERE] = init_velocity[
                    InitVelocityType.SPHERE
                ]
            self.init_velocity_dict[self.name] = init_velocity[InitVelocityType.SPHERE]

        # set external constant force
        if (
            constant_force is not None
            and constant_force.get(ConstantForceType.SPHERE) is not None
        ):
            if isinstance(constant_force[ConstantForceType.SPHERE][0], list):
                constant_force[ConstantForceType.SPHERE] = constant_force[
                    ConstantForceType.SPHERE
                ][0]
            else:
                constant_force[ConstantForceType.SPHERE] = constant_force[
                    ConstantForceType.SPHERE
                ]
            self.constant_force_dict[self.name] = constant_force[ConstantForceType.SPHERE]

    def get_connecting_tendon_sequences(
        self,
        direction: ConnectingDirection = ConnectingDirection.DEFAULT,
        connecting_option: Any = None,
    ) -> List[TendonSequence]:
        """
        Get the sites and geoms that tendons can connect to.
        Return the middle site of the sphere.
        """
        return [TendonSequence(
            elements=[self.site.create_spatial_site()],
            description="Tendon sequence returning the middle site of the sphere",
            name=f"{self.name}.connecting_tendon"
        )]

    def add_planar_joint(self, plane_slope: float = 0):
        # Convert plane_slope from degrees to radians
        theta = math.radians(plane_slope)
        # First axis lies in the plane, inclined at plane_slope along x
        axis1 = (0, math.cos(theta), math.sin(theta))
        # Second axis is perpendicular to axis1 in the plane
        axis2 = (1, 0, 0)  # Since rotation is only around x-axis, this remains constant

        self.joints = []  # Clear existing joints
        
        # Add two slide joints along the axes
        self.add_joint(Joint("slide", axis1, f"{self.name}.joint-1"))
        self.add_joint(Joint("slide", axis2, f"{self.name}.joint-2"))

        # Optionally, to prevent rotation, you can add a 'ball' joint with limited rotations
        # But since spheres are symmetric, rotations may not be necessary to constrain


class PolygonalPrism(Body):
    """
    A specialized Body class to represent a polygonal prism.
    """

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        sides: int = DEFAULT_POLYGON_SIDES,
        radius: float = DEFAULT_POLYGON_RADIUS,
        height: float = DEFAULT_CYLINDER_HEIGHT,
        material: str = DEFAULT_MATERIAL,
        rgba: Tuple[float, float, float, float] = DEFAULT_RGBA,
        mass: float = 1.0,
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        **kwargs,
    ) -> None:
        super().__init__(name, pos, quat, **kwargs)
        self.sides = sides
        self.radius = radius
        self.height = height
        self.body_type = "polygonal prism"
        self.add_geom(
            Geom(
                name=f"{name}.geom",
                geom_type="sdf",  # Use the mesh as an SDF, to avoid convex hull issues
                mesh="polygonal_cylinder",
                pos=(0, 0, 0),
                size=(self.height, self.radius, self.radius),  # originally it is 2 1 1
                quat=(0.707, 0, 0.707, 0),
                material=material,
                rgba=rgba,
                mass=mass,
                plugin="sdf",  # Add a plugin for the mesh to solve issues caused by convex hull
            )
        )
        self.center_site = Site(f"{name}.center_site", (0, 0, 0), body_name=name)
        self.top_site = Site(f"{name}.top_site", (height / 2, 0, 0), body_name=name)
        self.bottom_site = Site(
            f"{name}.bottom_site", (-height / 2, 0, 0), body_name=name
        )
        self.add_site(self.center_site)
        self.add_site(self.top_site)
        self.add_site(self.bottom_site)

    def to_xml(self) -> str:
        body_xml = f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}" quat="{' '.join(map(str, self.quat))}">"""
        for geom in self.geoms:
            body_xml += geom.to_xml() + "\n"
        for site in self.sites:
            body_xml += site.to_xml() + "\n"
        for joint in self.joints:
            body_xml += joint.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml


class Cylinder(Body):
    """
    A specialized Body class to represent a cylinder.
    """

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        radius: float = DEFAULT_CYLINDER_RADIUS,
        height: float = DEFAULT_CYLINDER_HEIGHT,
        material: str = DEFAULT_MATERIAL,
        rgba: Tuple[float, float, float, float] = DEFAULT_RGBA,
        mass: float = 1.0,
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        **kwargs,
    ) -> None:
        super().__init__(name, pos, quat, **kwargs)
        self.radius = radius
        self.height = height
        self.body_type = "cylinder"
        self.add_geom(
            Geom(
                name=f"{name}.geom",
                geom_type="cylinder",
                pos=(0, 0, 0),
                size=(radius, height / 2),
                material=material,
                rgba=rgba,
                mass=mass,
            )
        )
        self.center_site = Site(f"{name}.center_site", (0, 0, 0), body_name=name)
        self.top_site = Site(f"{name}.top_site", (0, 0, height / 2), body_name=name)
        self.bottom_site = Site(
            f"{name}.bottom_site", (0, 0, -height / 2), body_name=name
        )
        self.add_site(self.center_site)
        self.add_site(self.top_site)
        self.add_site(self.bottom_site)

    def set_horizontal(self) -> None:
        """
        Rotate the Cylinder around the x-axis by -90°, so that the original z-axis points to the positive y-axis.
        """
        # A -90° rotation corresponds to a half angle of -45°
        angle = -math.pi / 2  # -90° (in radians)
        half_angle = angle / 2
        # Calculate the rotation quaternion: (cos(half_angle), sin(half_angle), 0, 0)
        new_quat = (math.cos(half_angle), math.sin(half_angle), 0, 0)
        self.quat = new_quat
        # If needed, update the local coordinates of the site.
        # Here, it is assumed that the main axis of the cylinder after rotation becomes the y-axis.
        self.top_site.pos = (0, self.height / 2, 0)
        self.bottom_site.pos = (0, -self.height / 2, 0)

    def to_xml(self) -> str:
        body_xml = f"""<body name="{self.name}" pos="{' '.join(map(str, self.pos))}" quat="{' '.join(map(str, self.quat))}">"""
        for geom in self.geoms:
            body_xml += geom.to_xml() + "\n"
        for site in self.sites:
            body_xml += site.to_xml() + "\n"
        for joint in self.joints:
            body_xml += joint.to_xml() + "\n"
        body_xml += "</body>"
        return body_xml


class Disc(Body):
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        radius: float = DEFAULT_DISC_RADIUS,
        height: float = DEFAULT_DISC_HEIGHT,
        **kwargs
    ):
        super().__init__(name, pos)
        self.radius = radius
        self.height = height
        self.geom = Geom(
            name=f"{name}.geom",
            geom_type="cylinder",
            pos=(0, 0, 0),
            size=(radius, height),
            rgba=(0.5, 0.5, 0.5, 1),
            mass=1.0,
        )
        self.add_geom(self.geom)
        self.body_type = "disc"


class Bar(Body):  # Represents a bar with a start and end point
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        length: float = DEFAULT_BAR_LENGTH,  # full length of the bar
        width: float = DEFAULT_BAR_THICKNESS,  # width of the bar
        height: float = DEFAULT_BAR_THICKNESS,  # height of the bar
        quat: Optional[Tuple[float, float, float, float]] = None,
        end_pos: Optional[Tuple[float, float, float]] = None,
        mass: float = 1.0,
        **kwargs
    ):
        """
        If `end_pos` is provided, the bar's length and orientation are determined from pos->end_pos.
        If `end_pos` is not provided, it is assumed that `length` and `quat` are given (if quat is not given, identity is used).
        """

        if end_pos is not None:
            dx = end_pos[0] - pos[0]
            dy = end_pos[1] - pos[1]
            dz = end_pos[2] - pos[2]
            length = math.sqrt(dx * dx + dy * dy + dz * dz)

            # Normalize the direction vector
            if length > 0:
                dir_vec = (dx / length, dy / length, dz / length)
            else:
                dir_vec = (1, 0, 0)  # Default direction

            # We need a quat to align the bar's local x-axis (1,0,0) with the dir_vec direction
            # Algorithm: Calculate the quaternion that rotates (1,0,0) to dir_vec
            # If dir_vec and (1,0,0) are nearly parallel, we can use a simple formula; otherwise, we use the general method.
            x_axis = (1.0, 0.0, 0.0)
            dot = (
                x_axis[0] * dir_vec[0] + x_axis[1] * dir_vec[1] + x_axis[2] * dir_vec[2]
            )
            if abs(dot - 1.0) < 1e-12:
                # dir_vec and x_axis are in the same direction
                quat = (1.0, 0.0, 0.0, 0.0)
            elif abs(dot + 1.0) < 1e-12:
                # dir_vec and x_axis are in opposite directions
                # This means a 180-degree rotation around the z-axis or y-axis is needed, so we choose one orthogonal axis
                # For example, a 180-degree rotation around the z-axis: quat = (cos(π/2), 0, 0, sin(π/2)) = (0, 0, 0, 1)
                quat = (0.0, 0.0, 0.0, 1.0)
            else:
                # General case
                cross_x = 0.0
                cross_y = (
                    x_axis[2] * dir_vec[0] - x_axis[0] * dir_vec[2]
                )  # (1,0,0)x(dx,dy,dz)=(0,-dz,dy)
                cross_z = (
                    x_axis[0] * dir_vec[1] - x_axis[1] * dir_vec[0]
                )  # = (0, -dz, dy)
                # Actually from above: cross((1,0,0), (dx,dy,dz)) = (0, -dz, dy)
                angle = math.acos(dot)
                s = math.sin(angle / 2.0)
                qw = math.cos(angle / 2.0)
                # Normalize the rotation axis
                axis_len = math.sqrt(
                    cross_x * cross_x + cross_y * cross_y + cross_z * cross_z
                )
                if axis_len > 1e-12:
                    ux = cross_x / axis_len
                    uy = cross_y / axis_len
                    uz = cross_z / axis_len
                else:
                    # If the direction is close to the x-axis but not exactly the same
                    # Find an orthogonal axis
                    ux, uy, uz = 0.0, 0.0, 1.0
                qx = ux * s
                qy = uy * s
                qz = uz * s
                quat = (qw, qx, qy, qz)
        else:
            # If end_pos is not given, use the provided length and quat; if quat is not provided, use the identity quaternion
            if quat is None:
                quat = (1.0, 0.0, 0.0, 0.0)

        super().__init__(name, (0, 0, 0))

        self.length = length
        self.width = width
        self.height = height
        self.body_type = "bar"

        # Set the bar's pose
        self.set_pose(pos=pos, quat=quat)

        # Create the bar's geometry
        self.geom = Geom(
            name=f"{name}.geom",
            geom_type="box",
            pos=(length / 2, 0, 0),
            size=(length / 2, width / 2, height / 2),
            rgba=(0.5, 0.5, 0.5, 1),
            mass=mass,
        )
        self.add_geom(self.geom)

        # Create sites at both ends
        left_site = Site(f"{name}.left_site", (0, 0, 0), body_name=name)
        right_site = Site(f"{name}.right_site", (length, 0, 0), body_name=name)
        center_site = Site(f"{name}.center_site", (length / 2, 0, 0), body_name=name)
        self.add_site(left_site)
        self.add_site(right_site)
        self.add_site(center_site)


class Hemisphere(Body):
    """
    Body for generating a hemisphere mesh.
    """
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        radius: float = 1.0,
        thickness: float = -1.0,  # <0 means solid, >0 means with thickness
        material: str = DEFAULT_MATERIAL,
        rgba: Tuple[float, float, float, float] = DEFAULT_RGBA,
        mass: float = 1,
        plugin: str = "sdf",
        **kwargs,
    ):
        self.radius = radius
        self.thickness = thickness
        self.mass = mass
        super().__init__(name, pos, quat, **kwargs)
        self.body_type = "hemisphere"

        # 1. STL path
        stl_name = f"hemisphere_r{self.radius}_t{self.thickness}.stl"
        stl_path = os.path.join(REPO_PATH, GEOM_GENERATED_SOURCES_PATH, stl_name)

        # 2. Generate if not exist
        if not os.path.exists(stl_path):
            os.makedirs(os.path.dirname(stl_path), exist_ok=True)
            hemisphere_obj = get_hemisphere(radius=self.radius, thickness=self.thickness)
            export(hemisphere_obj, stl_path)

        # 3. Add Geom
        self.add_geom(
            Geom(
                name=f"{name}.geom",
                geom_type="mesh",
                mesh=stl_name[:-4],  # only contains name
                pos=(0, 0, 0),
                size=(1, 1, 1),
                quat=(1, 0, 0, 0),
                material=material,
                rgba=rgba,
                mass=self.mass,
                plugin=plugin,
            )
        )

        # 4. Site
        self.center_site = Site(f"{name}.center_site", (0, 0, 0), body_name=name)
        self.add_site(self.center_site)


class Bowl(Body):
    """
    Body for generating a bowl-shaped mesh.
    """
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        radius: float = 1.0,
        height: float = 0.0,     # Controls the height of the cutting plane
        thickness: float = -1.0, # <0 means solid, >0 means with thickness
        material: str = DEFAULT_MATERIAL,
        rgba: Tuple[float, float, float, float] = DEFAULT_RGBA,
        mass: float = 1,
        plugin: str = "sdf",
        **kwargs,
    ):
        self.height = height
        self.thickness = thickness
        self.mass = mass
        self.radius = radius
        super().__init__(name, pos, quat, **kwargs)
        self.body_type = "bowl"

        # 1. STL file name
        stl_name = f"bowl_r{self.radius}_h{self.height}_t{self.thickness}.stl"
        stl_path = os.path.join(REPO_PATH, GEOM_GENERATED_SOURCES_PATH, stl_name)

        # 2. Generate check
        if not os.path.exists(stl_path):
            os.makedirs(os.path.dirname(stl_path), exist_ok=True)
            bowl_obj = get_bowl(radius=self.radius, height=self.height, thickness=self.thickness)
            export(bowl_obj, stl_path)

        # 3. Geom
        self.add_geom(
            Geom(
                name=f"{name}.geom",
                geom_type="mesh",
                mesh=stl_name[:-4],  # only contains name
                pos=(0, 0, 0),
                size=(1, 1, 1),
                quat=(1, 0, 0, 0),
                material=material,
                rgba=rgba,
                mass=self.mass,
                plugin=plugin,
            )
        )

        # 4. Site
        self.center_site = Site(f"{name}.center_site", (0, 0, 0), body_name=name)
        self.add_site(self.center_site)


class SphereWithHole(Body):
    """
    Body for generating a sphere with a hole mesh.
    """
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        radius: float = 1.0,
        hole_radius: float = 0.5,
        hole_position: float = 0.0,
        thickness: float = -1.0,
        material: str = DEFAULT_MATERIAL,
        rgba: Tuple[float, float, float, float] = DEFAULT_RGBA,
        mass: float = 1,
        plugin: str = "sdf",
        **kwargs,
    ):
        self.radius = radius
        self.thickness = thickness
        self.mass = mass
        self.hole_radius = hole_radius
        self.hole_position = hole_position
        super().__init__(name, pos, quat, **kwargs)
        self.body_type = "sphere_with_hole"

        # 1. STL file name
        stl_name = f"sphereWithHole_r{self.radius}_holeR{self.hole_radius}_holePos{self.hole_position}_t{self.thickness}.stl"
        stl_path = os.path.join(REPO_PATH, GEOM_GENERATED_SOURCES_PATH, stl_name)

        # 2. Generate if not exist
        if not os.path.exists(stl_path):
            os.makedirs(os.path.dirname(stl_path), exist_ok=True)
            hole_sphere_obj = get_sphere_with_hole(
                radius=self.radius,
                hole_radius=self.hole_radius,
                hole_position=self.hole_position,
                thickness=self.thickness
            )
            print(f"stl_path: {stl_path}")
            export(hole_sphere_obj, stl_path)

        # 3. Add Geom
        self.add_geom(
            Geom(
                name=f"{name}.geom",
                geom_type="mesh",
                mesh=stl_name[:-4],  # only contains name
                pos=(0, 0, 0),
                size=(1, 1, 1),
                quat=(1, 0, 0, 0),
                material=material,
                rgba=rgba,
                mass=self.mass,
                plugin=plugin,
            )
        )

        # 4. Site
        self.center_site = Site(f"{name}.center_site", (0, 0, 0), body_name=name)
        self.add_site(self.center_site)

class RocketGeom(Body):
    """
    Body for generating a rocket mesh.
    """
    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float] = (0, 0, 0),
        quat: Tuple[float, float, float, float] = (1, 0, 0, 0),
        material: str = DEFAULT_MATERIAL,
        rgba: Tuple[float, float, float, float] = DEFAULT_RGBA,
        plugin: str = "sdf",
        **kwargs
    ):
        super().__init__(name, pos, quat, **kwargs)
        self.body_type = "rocket_geom"

        # 1. obj file names
        obj_names = [f"rocket.obj{i}.obj" for i in range(1, 14)]
        mat_names = [f[:-4] for f in obj_names]
        # stl_path = os.path.join(REPO_PATH, GEOM_FIXED_SOURCES_PATH, stl_name)

        # 2. Add Geoms
        for i, (obj_name, mat_name) in enumerate(zip(obj_names, mat_names)):
            self.add_geom(
                Geom(
                    name=f"{name}.geom{i}",
                    geom_type="mesh",
                    mesh=obj_name[:-4],  # only contains name
                    pos=(0, 0, 0),
                    size=(1, 1, 1),
                    quat=(1, 0, 0, 0),
                    material=mat_name,
                    rgba=None,
                    mass=0,
                    plugin=None,
                    contype="0",
                    conaffinity="0",
                )
            )

        self.add_child_body(Inertial())