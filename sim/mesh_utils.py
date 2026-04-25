try:
    import bpy
except ImportError:
    bpy = None

import ipdb, time, pathlib
import os
st = ipdb.set_trace

def get_sphere(radius: float = 1, thickness: float = -1):
    '''
    Create a sphere with the given radius and thickness.
    If thickness is negative, the sphere will be solid.
    '''
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, 0), segments=64, ring_count=32)
    
    if thickness > 0:

        bpy.ops.object.modifier_add(type='SOLIDIFY')
        bpy.context.object.modifiers["Solidify"].thickness = thickness
        bpy.ops.object.modifier_apply(modifier="Solidify")
    
    return bpy.context.object

# def get_bowl(radius: float = 1, height: float = 0, thickness: float = -1):
#     '''
#     Create a bowl (sphere bisected by a plane) with given radius and thickness.
#     The height parameter determines the position of the bisecting plane.
#     If thickness is negative, the bowl will be solid.
#     '''
#     sphere = get_sphere(radius)
#     bpy.context.view_layer.objects.active = sphere
#     sphere.select_set(True)
#     bpy.ops.object.mode_set(mode='EDIT')
#     bpy.ops.mesh.bisect(plane_co=(0, 0, height), plane_no=(0, 0, 1), threshold = 0, use_fill=thickness < 0, clear_inner=False, clear_outer=True)
    
#     bpy.ops.object.mode_set(mode='OBJECT')

#     if thickness > 0:

#         bpy.ops.object.modifier_add(type='SOLIDIFY')
#         bpy.context.object.modifiers["Solidify"].thickness = thickness
#         bpy.ops.object.modifier_apply(modifier="Solidify")
#     else:
#         bpy.ops.object.mode_set(mode='EDIT')
#         bpy.ops.object.mode_set(mode='EDIT')
#         bpy.ops.mesh.select_all(action='SELECT')

#         # Merge vertices by distance (use remove_doubles for older Blender versions)
#         bpy.ops.mesh.remove_doubles()

#         # Select and fix non-manifold geometry
#         bpy.ops.mesh.select_non_manifold()
#         bpy.ops.mesh.fill()

#         # Remove loose geometry
#         bpy.ops.mesh.delete_loose()

#         # Recalculate normals
#         bpy.ops.mesh.normals_make_consistent(inside=False)

#         bpy.ops.object.mode_set(mode='OBJECT')
    
#     return sphere

def get_bowl(radius: float = 1, height: float = 0, thickness: float = -1):
    '''
    Create a bowl (a sphere sliced flat at `height`) with given radius and thickness.
    If thickness < 0, it creates a solid (capped) bowl.
    '''
    sphere = get_sphere(radius)
    bpy.context.view_layer.objects.active = sphere
    sphere.select_set(True)

    # Bisect the sphere at given height
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.bisect(
        plane_co=(0, 0, height),
        plane_no=(0, 0, 1),
        threshold=0.0001,
        use_fill=False,
        clear_inner=False,
        clear_outer=True
    )
    bpy.ops.object.mode_set(mode='OBJECT')

    if thickness > 0:
        # Hollow shell using Solidify
        bpy.ops.object.modifier_add(type='SOLIDIFY')
        bpy.context.object.modifiers["Solidify"].thickness = thickness
        bpy.ops.object.modifier_apply(modifier="Solidify")
    else:
        # Solid bowl: cap and clean
        bpy.ops.object.mode_set(mode='EDIT')

        # Merge close verts (remove doubles)
        bpy.ops.mesh.remove_doubles(threshold=0.0001)

        # Recalculate normals
        bpy.ops.mesh.normals_make_consistent(inside=False)

        # Fill open edge
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_non_manifold()
        try:
            bpy.ops.mesh.fill()
        except:
            pass

        # Remove loose geometry
        bpy.ops.mesh.delete_loose()

        bpy.ops.object.mode_set(mode='OBJECT')

    return sphere

def get_hemisphere(radius: float = 1, thickness: float = -1):
    '''
    Create a hemisphere with given radius and thickness.
    If thickness is negative, the hemisphere will be solid.
    '''
    sphere = get_bowl(radius, thickness=thickness)
    return sphere

def get_sphere_with_hole(radius: float = 1, hole_radius: float = 0.5, hole_position: float = 0, thickness: float = -1):
    '''
    Create a sphere with a hole in it.
    The hole is a sphere with the given radius (hole_radius) and position (hole_position).
    If thickness is negative, the sphere will be solid.
    '''
    main_sphere = get_sphere(radius = radius, thickness = thickness)

    # Create the sphere that will act as the hole.
    # Its center is offset along the X-axis by hole_position.
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=hole_radius, 
        location=(hole_position, 0, 0), 
        segments=64, 
        ring_count=32
    )
    hole_sphere = bpy.context.object
    
    # Add a Boolean modifier to the main sphere to subtract the hole sphere.
    bool_mod = main_sphere.modifiers.new(name="HoleBoolean", type='BOOLEAN')
    bool_mod.object = hole_sphere
    bool_mod.operation = 'DIFFERENCE'
    
    # Make sure the main sphere is active and apply the Boolean modifier.
    bpy.context.view_layer.objects.active = main_sphere
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)
    
    # Delete the hole sphere as it is no longer needed.
    bpy.data.objects.remove(hole_sphere, do_unlink=True)

    main_sphere = bpy.context.object

    return main_sphere

def export(mesh, output_path: str):
    '''
    Export the given mesh to an STL file at the given output path.
    '''
    if isinstance(output_path, pathlib.Path): 
        output_path = str(output_path)
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    # bpy.ops.export_mesh.stl(filepath=output_path, check_existing=False, axis_forward='Y', axis_up='Z')
    bpy.ops.wm.stl_export(filepath=output_path, check_existing=False, forward_axis='Y', up_axis='Z')

def test():

    test_dir = pathlib.Path(__file__).parent / 'test_mesh_files'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    start = time.time()
    sphere = get_sphere()
    export(sphere, test_dir / "sphere_solid.stl")
    end = time.time()

    print(f'Time Elapsed (get_sphere): {round((end - start)*1000, 2)} ms')

    start = time.time()
    sphere = get_sphere(thickness=0.01)
    export(sphere, test_dir / "sphere_hollow.stl")
    end = time.time()

    print(f'Time Elapsed (get_sphere(thickness>0)): {round((end - start)*1000, 2)} ms')

    start = time.time()
    sphere = get_sphere_with_hole(hole_radius=1, hole_position=1)
    export(sphere, test_dir / "sphere_with_hole.stl")
    end = time.time()

    print(f'Time Elapsed (get_sphere_with_hole): {round((end - start)*1000, 2)} ms')

    start = time.time()
    sphere = get_sphere_with_hole(hole_radius=1, hole_position=1, thickness=0.01)
    export(sphere, test_dir / "sphere_hollow_with_hole.stl")
    end = time.time()

    print(f'Time Elapsed (get_sphere_with_hole(thickness>0)): {round((end - start)*1000, 2)} ms')

    start = time.time()
    sphere = get_bowl(height=0.5)
    export(sphere, test_dir / "bowl_solid.stl")
    end = time.time()

    print(f'Time Elapsed (get_bowl): {round((end - start)*1000, 2)} ms')

    start = time.time()
    sphere = get_bowl(height=0.5, thickness=0.01)
    export(sphere, test_dir / "bowl_hollow.stl")
    end = time.time()

    print(f'Time Elapsed (get_bowl(thickness>0)): {round((end - start)*1000, 2)} ms')
    
    start = time.time()
    sphere = get_hemisphere(thickness=-0.1)
    export(sphere, test_dir / "hemisphere_solid.stl")
    end = time.time()

    print(f'Time Elapsed (get_hemisphere): {round((end - start)*1000, 2)} ms')

    start = time.time()
    sphere = get_hemisphere(thickness=0.01)
    export(sphere, test_dir / "hemisphere_hollow.stl")
    end = time.time()

    print(f'Time Elapsed (get_hemisphere(thickness>0)): {round((end - start)*1000, 2)} ms')

if __name__ == "__main__":
    test()
