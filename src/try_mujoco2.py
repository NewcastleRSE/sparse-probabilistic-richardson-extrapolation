import mujoco
import numpy as np
import imageio

MJCF_TEMPLATE = """
<mujoco model="colliding_spheres">
    <option gravity="0 0 -9.81" timestep="{timestep}" integrator="RK4" tolerance="{tolerance}"/>

    <visual>
        <quality shadowsize="2048"/>
        <map znear="0.01"/>
    </visual>

    <asset>
        <!-- Floor material -->
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.2 0.2 0.2" rgb2="0.85 0.85 0.85"/>
        <material name="floor_mat" texture="grid" texrepeat="12 12" reflectance="0.15"/>

        <!-- Sphere materials -->
        <material name="sphere_a_mat" rgba="0.9 0.2 0.2 1" specular="0.6" shininess="0.8"/>
        <material name="sphere_b_mat" rgba="0.2 0.4 0.9 1" specular="0.6" shininess="0.8"/>
    </asset>

    <worldbody>
        <!-- Camera -->
        <camera name="angled_view"
                pos="{camera_pos_x} {camera_pos_y} {camera_pos_z}"
                xyaxes="0.7 0.7 0  -0.3 0.3 0.9"/>

        <!-- Lights -->
        <light name="sun" pos="0 0 5" dir="0 0 -1" diffuse="1.5 1.5 1.5" specular="0.6 0.6 0.6"/>
        <light name="fill" pos="2 2 3" diffuse="0.6 0.6 0.6"/>

        <!-- Floor -->
        <geom name="floor" type="plane" size="5 5 0.1" material="floor_mat"/>

        <!-- Sphere A -->
        <body name="a" pos="0 0 0.5">
            <joint type="free" damping="0.01"/>
            <geom type="sphere" size="0.08" mass="1" material="sphere_a_mat" contype="1" conaffinity="1" solimp="0.95 0.95 0.001"/>
        </body>

        <!-- Sphere B -->
        <body name="b" pos="0.5 0.1 0.55">
            <joint type="free" damping="0.01"/>
            <geom type="sphere" size="0.08" mass="1" material="sphere_b_mat" contype="1" conaffinity="1" solimp="0.95 0.95 0.001"/>
        </body>
    </worldbody>

   
</mujoco>
"""

def simulate_colliding_spheres_video(
    sim_time=3.0,
    video_path="colliding_spheres.mp4",
    fps=60,
    timestep=0.00001, #0.01,
    tolerance=1e-3 #1e-4
):
    zoom = 1.0
    camera_pos = np.array([2, -2, 2.0]) * zoom

    mjcf = MJCF_TEMPLATE.format(
        timestep=timestep,
        tolerance=tolerance,
        camera_pos_x=camera_pos[0],
        camera_pos_y=camera_pos[1],
        camera_pos_z=camera_pos[2]
    )

    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)

    # Initial velocities to ensure motion and collisions
    data.qvel[:3] = [1.0, 0, 0.0]    # sphere A linear velocity
    data.qvel[3:6] = [0, 0, 0.5]     # sphere A angular velocity
    data.qvel[6:9] = [-1.0, 0, 0.0]  # sphere B linear velocity
    data.qvel[9:12] = [0, 0, -0.5]   # sphere B angular velocity

    renderer = mujoco.Renderer(model, width=640, height=480)
    frames = []

    frame_interval = int(1.0 / (fps * model.opt.timestep))

    step = 0
    stationary_steps = 0
    STEPS_REQUIRED = 3
    VEL_THRESH = 1e-3
    body_ids = [model.body("a").id, model.body("b").id]

    while True:
        mujoco.mj_step(model, data)

        if step % frame_interval == 0:
            renderer.update_scene(data, camera="angled_view")
            frames.append(renderer.render().copy())

        step += 1

        # Check if both spheres are nearly stationary
        stationary = True
        for bid in body_ids:
            cvel = data.cvel[bid]
            lin_vel_norm = np.linalg.norm(cvel[3:])
            ang_vel_norm = np.linalg.norm(cvel[:3])
            if lin_vel_norm > VEL_THRESH or ang_vel_norm > VEL_THRESH:
                stationary = False
                break

        if stationary:
            stationary_steps += 1
        else:
            stationary_steps = 0

        # Stop if settled or reached sim_time
        if stationary_steps >= STEPS_REQUIRED or step * model.opt.timestep >= sim_time:
            print("Stopping simulation")
            break

    renderer.close()
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Video saved as {video_path}")

    final_positions = [data.xpos[bid].copy() for bid in body_ids]
    distances = [np.linalg.norm(pos) for pos in final_positions]
    return final_positions, distances

if __name__ == "__main__":
    pos, dist = simulate_colliding_spheres_video()
    print("Final positions:", pos)
    print("Distances from origin:", dist)
