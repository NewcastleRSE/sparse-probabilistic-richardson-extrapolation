import pybullet as p
import pybullet_data
import numpy as np
import os
import subprocess

# -----------------------
# Simulation parameters
# -----------------------
duration = 5.0          # seconds
fps = 30                # output video FPS
dt = 1/240              # physics timestep
frames = int(duration * fps)

# -----------------------
# PyBullet setup
# -----------------------
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load plane and R2D2
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF("r2d2.urdf", [0,0,1.0])  # drop from 1m height

# -----------------------
# Camera setup
# -----------------------
width, height = 640, 480
cam_target = [0, 0, 0.5]
dist = 2.0
yaw = 45
pitch = -20

# Steps per rendered frame
substeps = int((1/fps)/dt)

# -----------------------
# Prepare frame folder
# -----------------------
folder = "frames"
os.makedirs(folder, exist_ok=True)

# -----------------------
# Helper: save PPM frame
# -----------------------
def save_ppm(filename, img):
    """Save NumPy image (H,W,3) as binary PPM (P6)"""
    height, width, _ = img.shape
    with open(filename, 'wb') as f:
        f.write(f"P6 {width} {height} 255\n".encode())
        f.write(img.tobytes())

# -----------------------
# Simulation + frame capture
# -----------------------
for i in range(frames):
    # Physics stepping
    for _ in range(substeps):
        p.stepSimulation()

    # Camera render
    _, _, px, _, _ = p.getCameraImage(
        width, height,
        viewMatrix=p.computeViewMatrixFromYawPitchRoll(cam_target, dist, yaw, pitch, 0, 2),
        projectionMatrix=p.computeProjectionMatrixFOV(60, width/height, 0.1, 100)
    )

    # Convert RGBA → RGB
    frame = np.reshape(px, (height, width, 4))[:, :, :3]
    save_ppm(os.path.join(folder, f"frame_{i:04d}.ppm"), frame)

print("✔ All frames saved")

# -----------------------
# Assemble video using FFmpeg
# -----------------------
output_file = "r2d2_video.mp4"
cmd = [
    "ffmpeg",
    "-y",
    "-framerate", str(fps),
    "-i", os.path.join(folder, "frame_%04d.ppm"),
    "-pix_fmt", "yuv420p",
    output_file
]

subprocess.run(cmd, check=True)
print(f"✔ Video created: {output_file}")
