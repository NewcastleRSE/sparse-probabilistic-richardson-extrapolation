import pybullet as p
import pybullet_data
import time

# Connect to GUI (required for built-in video recorder)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Real-time simulation ON (critical!)
p.setRealTimeSimulation(1)

# Load world
p.setGravity(0, 0, -9.81)
plane = p.loadURDF("plane.urdf")
box = p.loadURDF(
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

#.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

 # Zoomed-in camera settings
#p.resetDebugVisualizerCamera(
#    cameraDistance=0.5,                # closer to the cube (default ~1.5)
#    cameraYaw=45,                      # rotate horizontally
#    cameraPitch=-50,                   # angle downward
#    cameraTargetPosition=[0, 0, 0]   # look at where the cube will fall
#)

# Start built-in MP4 recording
log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "pybullet_rt.mp4")

# Run for 5 seconds in *wall-clock* time
duration = 5.0
t0 = time.time()

while time.time() - t0 < duration:
    time.sleep(0.01)  # allow GUI to render frames

# Stop recording
p.stopStateLogging(log_id)
print("✔ Video finished.")
