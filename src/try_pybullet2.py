import pybullet as p
import pybullet_data
import numpy as np
import time

def run_simulation(
        duration=5.0,
        dt=1/240,
        substeps=1,
        solver_iters=50,
        save_animation=False,
        video_filename="fall_spin.mp4",
    ):

    mode = p.GUI if save_animation else p.DIRECT
    physicsClient = p.connect(mode)

    # Zoomed-in camera settings
    p.resetDebugVisualizerCamera(
        cameraDistance=0.5,                # closer to the cube (default ~1.5)
        cameraYaw=45,                      # rotate horizontally
        cameraPitch=-50,                   # angle downward
        cameraTargetPosition=[0, 0, 0]   # look at where the cube will fall
    )

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Simulation parameters
    p.setPhysicsEngineParameter(
        fixedTimeStep=dt,
        numSubSteps=substeps,
        numSolverIterations=solver_iters
    )

    # Ground plane
    plane = p.loadURDF("plane.urdf")

    # Create a simple box (0.2m cube)
    box = p.loadURDF(
        #"cube_small.urdf",
        #"r2d2.urdf",
        #"husky/husky.urdf",
        #"duck_vhacd.urdf",
        #"table/table.urdf",
        #"franka_panda/panda.urdf",
        #"random_urdfs\997\997.urdf",
        #"bicycle/bike.urdf",   
        "objects/mug.urdf",     
        basePosition=[0, 0, 2.0],        # start 1 meter above ground
        baseOrientation=[0, 0, 0, 1],
        #globalScaling=2.0 
    )

    p.setGravity(0, 0, -9.81)

    # Give the box an initial spin
    p.resetBaseVelocity(
        box,
        angularVelocity=[3.0, -1.5, 5.0]  # spin around x, y, z
    )

    # Start recording
    if save_animation:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        #p.setRealTimeSimulation(1)
        log_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4,
            video_filename
        )

    times, dists_origin = [], []
    sim_time = 0.0

    while sim_time < duration:
        p.stepSimulation()

        pos, orn = p.getBasePositionAndOrientation(box)
        dist = np.sqrt(pos[0]**2 + pos[2]**2 + pos[1]**2)
        dists_origin.append(dist)
        times.append(sim_time)

        # make real-time video look normal
        if save_animation:
            time.sleep(dt)

        sim_time += dt

    if save_animation:
        p.stopStateLogging(log_id)
        print(f"Saved video to {video_filename}")

    p.disconnect()

    return np.array(times), np.array(dists_origin)



if __name__ == "__main__":
    t, h = run_simulation(
        duration=5.0,
        dt=0.001, #1/240,
        substeps=1, #2,
        solver_iters=100, #100,
        save_animation=True,           # record the animation
        video_filename="drop_spin.mp4"
    )

    print("Final distance from origin:", h[-1])

    if 1:
        import pybullet_data
        import os

        # Path to pybullet_data
        data_path = pybullet_data.getDataPath()

        # List all URDF files
        urdf_files = []
        for root, dirs, files in os.walk(data_path):
            for f in files:
                if f.endswith(".urdf"):
                    urdf_files.append(os.path.relpath(os.path.join(root, f), data_path))

        print("Available URDFs:")
        for f in sorted(urdf_files):
            if f[:6] != "random":
                print(f)
