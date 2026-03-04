import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mesa import Agent, Model

# ---------------------------
# Video recorder using triangles
# ---------------------------
class VideoRecorder3DTriangles:
    def __init__(self, model, filename="simulation3d_triangles.mp4", fps=30, scale=0.5):
        self.model = model
        self.filename = filename
        self.fps = fps
        self.scale = scale

        self.fig = plt.figure(figsize=(10,7), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(0, self.model.space.width)
        self.ax.set_ylim(0, self.model.space.height)
        self.ax.set_zlim(0, self.model.space.depth)
        self.ax.set_title("3D Flocking Simulation")
        self.triangles = []
        self.writer = FFMpegWriter(fps=fps)
        self.writer.setup(self.fig, self.filename)

    def create_triangle(self, pos, vel, scale):
        # Simple triangle in the plane perpendicular to velocity
        if np.linalg.norm(vel) < 1e-8:
            vel = np.array([1.0,0.0,0.0])
        direction = vel / np.linalg.norm(vel)
        # find two perpendicular vectors
        if np.allclose(direction, [0,0,1]):
            v1 = np.array([1,0,0])
        else:
            v1 = np.cross(direction, [0,0,1])
            v1 /= np.linalg.norm(v1)
        v2 = np.cross(direction, v1)
        v2 /= np.linalg.norm(v2)

        # Triangle points
        tip = pos + direction*scale
        base1 = pos - direction*0.5*scale + 0.3*scale*v1
        base2 = pos - direction*0.5*scale - 0.3*scale*v1

        return np.array([tip, base1, base2])

    def capture_frame(self):
        # Remove old triangles
        for tri in self.triangles:
            tri.remove()
        self.triangles = []

        # Draw triangles for each agent
        for agent in self.model.agent_list:
            tri_pts = self.create_triangle(agent.pos, agent.vel, self.scale)
            poly = Poly3DCollection([tri_pts], color=agent.color)
            self.ax.add_collection3d(poly)
            self.triangles.append(poly)

        self.ax.set_xlim(0, self.model.space.width)
        self.ax.set_ylim(0, self.model.space.height)
        self.ax.set_zlim(0, self.model.space.depth)

        self.fig.canvas.draw()
        self.writer.grab_frame()

    def close(self):
        self.writer.finish()
        plt.close(self.fig)

# ---------------------------
# Agent and model code stays mostly the same as before
# ---------------------------
class ContinuousAgent3D(Agent):
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        self.vel = np.zeros(3)
        self.last_sense_time = 0.0
        self.cached_neighbors = []
        self.color = np.random.rand(3,)

    def sense(self):
        self.cached_neighbors = self.model.get_neighbors_3d(self, self.model.interaction_radius + self.model.epsilon)
        self.last_sense_time = self.model.time

    def compute_force(self):
        force = np.zeros(3)
        for other in self.cached_neighbors:
            dvec = other.pos - self.pos
            for i, dim in enumerate([self.model.space.width, self.model.space.height, self.model.space.depth]):
                if abs(dvec[i]) > dim/2:
                    dvec[i] -= np.sign(dvec[i])*dim
            dist = np.linalg.norm(dvec)
            if dist < 1e-12:
                continue
            direction = dvec / dist
            if dist < self.model.repulsion_radius:
                force -= direction * (self.model.repulsion_radius - dist)
            elif dist < self.model.interaction_radius:
                force += direction * (dist - self.model.repulsion_radius)
        return force

    def step(self):
        if self.model.time - self.last_sense_time >= self.model.tau:
            self.sense()
        force = self.compute_force()
        self.vel += force * self.model.dt
        self.pos += self.vel * self.model.dt
        self.pos[0] %= self.model.space.width
        self.pos[1] %= self.model.space.height
        self.pos[2] %= self.model.space.depth

    def distance_from_origin(self):
        return np.linalg.norm(self.pos)

class FlockingModel3D(Model):
    def __init__(self, n_agents=20, width=10, height=10, depth=10,
                 dt=0.05, epsilon=0.05, tau=0.05,
                 interaction_radius=2.0, repulsion_radius=0.5,
                 record_video=True, video_filename="flocking3d_triangles.mp4", video_fps=20):
        super().__init__()
        self.dt = dt
        self.epsilon = epsilon
        self.tau = tau
        self.time = 0.0
        self.interaction_radius = interaction_radius
        self.repulsion_radius = repulsion_radius

        self.space = type("ContinuousSpace3D", (), {})()
        self.space.width = width
        self.space.height = height
        self.space.depth = depth

        self.agent_list = []
        for i in range(n_agents):
            agent = ContinuousAgent3D(i, self)
            agent.pos = np.random.rand(3) * np.array([width, height, depth])
            self.agent_list.append(agent)

        self.time_data = []
        self.distance_data = []

        self.record_video = record_video
        self.video = VideoRecorder3DTriangles(self, video_filename, video_fps) if record_video else None

    def get_neighbors_3d(self, agent, radius, include_center=False):
        neighbors = []
        for other in self.agent_list:
            if other is agent and not include_center:
                continue
            dvec = other.pos - agent.pos
            for i, dim in enumerate([self.space.width, self.space.height, self.space.depth]):
                if abs(dvec[i]) > dim/2:
                    dvec[i] -= np.sign(dvec[i])*dim
            dist = np.linalg.norm(dvec)
            if dist <= radius:
                neighbors.append(other)
        return neighbors

    def step(self):
        for agent in self.agent_list:
            agent.step()
        self.time += self.dt
        self.time_data.append(self.time)
        distances = {agent.unique_id: agent.distance_from_origin() for agent in self.agent_list}
        self.distance_data.append(distances)
        if self.record_video:
            self.video.capture_frame()

    def finalize(self):
        if self.record_video:
            self.video.close()

    def total_distance_from_origin(self):
        return sum(agent.distance_from_origin() for agent in self.agent_list)

# ---------------------------
# Run example
# ---------------------------
if __name__ == "__main__":
     # Set seed for reproducability
    np.random.seed(1)

    model = FlockingModel3D(
        n_agents=50,
        dt=0.02,
        tau=0.05,
        epsilon=0.05,
        record_video=True,
        video_filename="flocking3d_triangles.mp4",
        video_fps=20
    )

    for _ in range(500):
        model.step()
    model.finalize()

    print("Final total distance from origin:", model.total_distance_from_origin())
