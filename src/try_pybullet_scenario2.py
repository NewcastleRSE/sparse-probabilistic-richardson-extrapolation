import pybullet as p
import pybullet_data
import time
import numpy as np

# Connect to GUI (required for built-in video recorder)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Real-time simulation ON (critical!)
p.setRealTimeSimulation(1)

# Load world
p.setGravity(0, 0, -9.81)

plane = p.loadURDF("plane.urdf")

box = p.loadURDF(
            #"objects/mug.urdf", 
            #"teddy_large.urdf",    
            #"soccerball.urdf",
            "duck_vhacd.urdf",
        basePosition=[0, 0, 0.05],        # start 1 meter above ground
        baseOrientation=[0, 0, 0, 1],
        #globalScaling=2.0 
    )

p.setGravity(0, 0, -9.81)

# Give the box an initial spin
p.resetBaseVelocity(
    box,
    angularVelocity=[3.0, -1.5, 5.0]  # spin around x, y, z
)

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

 # Zoomed-in camera settings
p.resetDebugVisualizerCamera(
    cameraDistance=0.5,                # closer to the cube (default ~1.5)
    cameraYaw=45,                      # rotate horizontally
    cameraPitch=-50,                   # angle downward
    cameraTargetPosition=[0, 0, 0]   # look at where the cube will fall
)

# Start built-in MP4 recording
log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "pybullet_sc2.mp4")

# Run for 5 seconds in *wall-clock* time
total_time = 1.0
dt = 1.0/240

# Simulation parameters
p.setPhysicsEngineParameter(
    fixedTimeStep = dt,
    numSubSteps = 100,
    #numSolverIterations = 
)

# Initial time counter
sim_time = 0.0

# Run the simulation
while sim_time < total_time:
    # One step of simulation
    p.stepSimulation()
    sim_time += dt
    time.sleep(dt) 

#t0 = time.time()

#while time.time() - t0 < duration:
#    time.sleep(dt)  # allow GUI to render frames

# Stop recording
p.stopStateLogging(log_id)
print("✔ Video finished.")

 # Get final position and orientation of mug
pos, orn = p.getBasePositionAndOrientation(box)

# Get distance of mug from origin
dist = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)

# End simulation
p.disconnect()

print(f"\tCalculated final value: {dist}")
