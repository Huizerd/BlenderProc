import blenderproc as bproc  # should be at the top

import argparse
import os

import imageio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", default="lava")
    parser.add_argument("output_dir", default="output")
    args = parser.parse_args()

    bproc.init()

    # load the objects into the scene
    # you can also load light/camera here but let's create those with blenderproc
    objs = bproc.loader.load_blend(f"resources/{args.name}/{args.name}.blend", obj_types=["mesh"])

    # add sun
    light = bproc.types.Light()
    light.set_type("SUN")
    light.set_energy(5)

    # set camera resolution
    bproc.camera.set_resolution(180, 180)

    # set the camera to look down on the plane
    euler_down = [0, 0, 0]

    # move in straight line
    speed = 0.02
    height = 2
    frames = 100
    dist = speed * frames
    for i in range(frames):
        location_cam = [-dist / 2 + i * speed, 0, height]
        cam_pose = bproc.math.build_transformation_mat(location_cam, euler_down)
        bproc.camera.add_camera_pose(cam_pose)

    # render the scene
    # TODO: we can set max amount of samples for quicker rendering
    # bproc.renderer.set_max_amount_of_samples(100)
    data = bproc.renderer.render()

    # output dir
    output_dir = f"{args.output_dir}/{args.name}"
    os.makedirs(output_dir, exist_ok=True)

    # write frames as images
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    for i, color in enumerate(data["colors"]):
        image_path = os.path.join(f"{output_dir}/images", f"image_{i:03}.png")
        imageio.imwrite(image_path, color)
    
    # write data to hdf5 container and gif animation
    os.makedirs(f"{output_dir}/hdf5", exist_ok=True)
    bproc.writer.write_hdf5(f"{output_dir}/hdf5", data)
    bproc.writer.write_gif_animation(output_dir, data)  # creates gif dir inside
