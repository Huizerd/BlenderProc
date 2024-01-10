import blenderproc as bproc  # should be at the top

import argparse
import math
import os

import imageio
import numpy as np
import yaml


def add_random_bezier_trajectory(num_control_points, rotational_disturbance, num_points):
    """
    From ChatGPT 4.
    """
    # generate random control points
    control_points = np.random.rand(num_control_points, 3) * 10  # scale as needed

    # apply rotational disturbances
    # for point in control_points:
    #     disturbance = np.random.randn(3) * rotational_disturbance
    #     rotation_matrix = create_rotation_matrix(disturbance)
    #     point[:] = np.dot(rotation_matrix, point)

    # generate bezier curve points from control points
    bezier_points = generate_bezier_points(control_points, num_points)

    # convert bezier points to transformation matrices
    prev_rotation = np.array([0, 0, 0])
    for point in bezier_points:
        # generate a smooth random rotation
        random_rotation = prev_rotation + np.random.randn(3) * rotational_disturbance  # small random change
        rotation_matrix = create_rotation_matrix(np.radians(random_rotation))
        prev_rotation = random_rotation

        # build transformation matrix
        transformation_mat = bproc.math.build_transformation_mat(point, rotation_matrix)
        # add to camera trajectory
        bproc.camera.add_camera_pose(transformation_mat)


def generate_bezier_points(control_points, num_points):
    """
    From ChatGPT 4.
    """
    # generate points on a bezier curve given control points
    curve_points = []
    n = len(control_points) - 1
    for t in np.linspace(0, 1, num_points):
        point = np.zeros(3)
        for i in range(len(control_points)):
            bernstein_poly = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            point += bernstein_poly * control_points[i]
        curve_points.append(point)
    return np.array(curve_points)


def add_regular_trajectory(unscaled_speed, distance_to_ground, rotational_speed, duration, time_step):
    """
    From ChatGPT 4.
    """
    position = np.array([0, 0, distance_to_ground], dtype=np.float64)  # initial position
    rotation = np.array([0, 0, 0], dtype=np.float64)  # initial rotation in euler angles
    unscaled_speed = np.array(unscaled_speed, dtype=np.float64)  # constant unscaled speed
    rotational_speed = np.array(rotational_speed, dtype=np.float64)  # constant rotational speed

    for _ in np.arange(0, duration, time_step):
        # compute scaled speed: multiply by distance to ground
        speed = unscaled_speed * (position[2] - 0.1)  # stop 10cm above plane

        # update position
        position += speed * time_step

        # update rotation
        rotation += rotational_speed * time_step
        # convert rotation to radians and then to rotation matrix
        rotation_matrix = create_rotation_matrix(np.radians(rotation))

        # build transformation matrix
        transformation_mat = bproc.math.build_transformation_mat(position, rotation_matrix)

        # add to camera trajectory
        bproc.camera.add_camera_pose(transformation_mat)


def create_rotation_matrix(angles):
    """
    From ChatGPT 4.
    """
    # create rotation matrix for given angles (in radians)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(angles[0]), -math.sin(angles[0])],
                   [0, math.sin(angles[0]),  math.cos(angles[0])]])

    Ry = np.array([[ math.cos(angles[1]), 0, math.sin(angles[1])],
                   [0, 1, 0],
                   [-math.sin(angles[1]), 0, math.cos(angles[1])]])

    Rz = np.array([[math.cos(angles[2]), -math.sin(angles[2]), 0],
                   [math.sin(angles[2]),  math.cos(angles[2]), 0],
                   [0, 0, 1]])

    return np.dot(Rz, np.dot(Ry, Rx))


def euler_to_quaternion(rotation):
    """
    From ChatGPT 4.
    """
    # assumes rotation is in radians
    cy = math.cos(rotation[2] * 0.5)
    sy = math.sin(rotation[2] * 0.5)
    cp = math.cos(rotation[1] * 0.5)
    sp = math.sin(rotation[1] * 0.5)
    cr = math.cos(rotation[0] * 0.5)
    sr = math.sin(rotation[0] * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("output_dir")
    parser.add_argument("--motion", choices=["regular", "random_bezier"], default="regular")
    parser.add_argument("--motion_file", default="motions/x-slow.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bproc.init()

    # load the objects into the scene
    # you can also load light/camera here but let's create those with blenderproc
    objs = bproc.loader.load_blend(f"resources/{args.name}/{args.name}.blend", obj_types=["mesh"])

    # enlarge plane to cover the whole scene
    plane = bproc.filter.one_by_attr(objs, "name", "Plane")
    scale = plane.get_scale()
    plane.set_scale([s * 10 for s in scale])
    plane.scale_uv_coordinates(10)

    # add sun
    light = bproc.types.Light()
    light.set_type("SUN")
    light.set_energy(5)

    # set camera resolution
    bproc.camera.set_resolution(180, 180)

    # seed
    np.random.seed(args.seed)

    # camera motion
    # regular trajectory
    if args.motion == "regular":
        with open(args.motion_file) as f:
            motion = yaml.load(f, Loader=yaml.SafeLoader)

        # generate camera trajectory
        add_regular_trajectory(
            unscaled_speed=motion["unscaled_speed"],
            distance_to_ground=motion["distance_to_ground"],
            rotational_speed=motion["rotational_speed"],
            duration=motion["duration"],
            time_step=motion["time_step"],
        )
    # random bezier trajectory
    elif args.motion == "random_bezier":
        with open(args.motion_file) as f:
            motion = yaml.load(f, Loader=yaml.SafeLoader)

        # generate camera trajectory
        add_random_bezier_trajectory(
            num_control_points=motion["num_control_points"],
            rotational_disturbance=motion["rotational_disturbance"],
            num_points=motion["num_points"],
        )

    # render the scene
    # TODO: we can set max amount of samples for quicker rendering
    # bproc.renderer.set_max_amount_of_samples(100)
    data = bproc.renderer.render()

    # output dir
    motion_name = os.path.splitext(os.path.basename(args.motion_file))[0]
    output_dir = f"{args.output_dir}/{args.name}/{motion_name}"
    os.makedirs(output_dir, exist_ok=True)

    # write frames as images
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    for i, color in enumerate(data["colors"]):
        image_path = os.path.join(f"{output_dir}/images", f"image_{i:03}.png")
        imageio.imwrite(image_path, color)

    # get forward flow for all frames
    # TODO: looks crappy/full of artifacts, so compute manually
    # data.update(bproc.renderer.render_optical_flow(get_backward_flow=False, get_forward_flow=True, blender_image_coordinate_style=False))

    # write data to hdf5 container and gif animation
    os.makedirs(f"{output_dir}/hdf5", exist_ok=True)
    bproc.writer.write_hdf5(f"{output_dir}/hdf5", data)
    bproc.writer.write_gif_animation(output_dir, data)  # creates gif dir inside
