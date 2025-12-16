import mujoco
import numpy as np
import imageio

MJCF_TEMPLATE = """
<mujoco model="falling_object">
    <option gravity="0 0 -9.81" timestep="{timestep}" integrator="RK4" impratio="{impratio}"/>

    <visual>
        <quality shadowsize="2048"/>
        <map znear="0.01"/>
    </visual>

    <asset>
        <!-- Floor texture -->
        <texture name="grid" type="2d"
                 builtin="checker"
                 width="512" height="512"
                 rgb1="0.2 0.2 0.2"
                 rgb2="0.85 0.85 0.85"/>
        <material name="floor_mat"
                  texture="grid"
                  texrepeat="12 12"
                  reflectance="0.15"/>

        <!-- Sphere material -->
        <material name="sphere_mat"
                  rgba="0.2 0.4 0.9 1"
                  specular="0.6"
                  shininess="0.8"/>
    </asset>

    <worldbody>
        <!-- Lights -->
        <light name="sun"
               pos="0 0 5"
               dir="0 0 -1"
               diffuse="1 1 1"
               specular="0.3 0.3 0.3"/>

        <light name="fill"
               pos="2 2 3"
               diffuse="0.3 0.3 0.3"/>

        <!-- Camera -->
        <camera name="angled_view"                
                pos="{camera_pos_x} {camera_pos_y} {camera_pos_z}"
                xyaxes="0.7 0.7 0  -0.3 0.3 0.9"/>

        <!-- Ground -->
        <geom name="ground"
              type="plane"
              size="5 5 0.1"
              material="floor_mat"/>

        <!-- Sphere -->
        <body name="sphere" pos="0 0 0.15">
            <joint type="free" damping="0.015"/>
            <geom type="sphere"
                  size="0.08"
                  mass="1"                
                  material="sphere_mat"
                  />
        </body>
    </worldbody>
</mujoco>
"""

def simulate_falling_object_with_video(
    sim_time=3.0,
    video_path="falling_object.mp4",
    fps=60,
    timestep=0.001,
    sub_iterations=10
):
    """
    Simulate a sphere falling onto a plane and save video.

    Parameters
    ----------
    sim_time : float
        Total simulation time in seconds.
    video_path : str
        Output video path.
    fps : int
        Frames per second in video.
    timestep : float
        Simulation timestep (seconds).
    sub_iterations : int
        Number of solver sub-iterations per timestep (contact solver accuracy).

    Returns
    -------
    final_position : np.ndarray
        Final position of the sphere (x, y, z).
    distance : float
        Euclidean distance from the origin.
    """

    # Set camera zoom
    zoom = 0.5
    camera_pos = np.array([2, -2, 1.5]) * zoom

    # Insert timestep & impratio into MJCF
    mjcf = MJCF_TEMPLATE.format(timestep=timestep, impratio=sub_iterations, camera_pos_x=camera_pos[0], camera_pos_y=camera_pos[1], camera_pos_z=camera_pos[2])

    # Load model and data
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)

    # Give the sphere an initial velocity for angled impact
    # qvel layout for a free joint: [vx, vy, vz, wx, wy, wz]
    data.qvel[:3] = np.array([0.1, 0.1, 0.0]) 

    # Renderer
    renderer = mujoco.Renderer(model, width=640, height=480)
    frames = []

    steps = int(sim_time / model.opt.timestep)
    frame_interval = int(1.0 / (fps * model.opt.timestep))

    for step in range(steps):
        mujoco.mj_step(model, data)

        if step % frame_interval == 0:
            renderer.update_scene(data, camera="angled_view")
            frame = renderer.render()
            frames.append(frame)

        # Optional early stop if sphere has settled
        vel = np.linalg.norm(data.qvel[:3])
        ang_vel = np.linalg.norm(data.qvel[3:])
        if vel < 1e-3 and ang_vel < 1e-3:
            break

    # Save video
    imageio.mimsave(video_path, frames, fps=fps)

    # Final position & distance
    body_id = model.body("sphere").id
    final_position = data.xpos[body_id].copy()
    distance = np.linalg.norm(final_position)

    return final_position, distance


if __name__ == "__main__":
    pos, dist = simulate_falling_object_with_video()
    print("Final position:", pos)
    print("Final distance from origin:", dist)
    print("Video saved as falling_object.mp4")
