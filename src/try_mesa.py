import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon
from mesa import Agent, Model
from mesa.space import ContinuousSpace


import numpy as np
from matplotlib.patches import Polygon

# ---------------------------
# Video recorder helper
# ---------------------------
class VideoRecorder:
    def __init__(self, model, filename="simulation.mp4", fps=30, wrap_visualization=False, final_frame_png="final_frame.png"):
        self.model = model
        self.filename = filename
        self.fps = fps
        self.wrap_visualization = wrap_visualization
        self.final_frame_png = final_frame_png

        self.fig, self.ax = plt.subplots()
        self.writer = FFMpegWriter(fps=fps)
        self.agent_artists = []

    def setup(self):
        self.ax.set_xlim(0, self.model.space.width)
        self.ax.set_ylim(0, self.model.space.height)
        self.ax.set_aspect("equal") 
        self.ax.set_title("Multi-Agent Simulation")

        # Create triangle for each agent
        for agent in self.model.agent_list:
            color = agent.color
            triangle = Polygon(self._triangle_coords(agent), color=color)
            self.ax.add_patch(triangle)
            self.agent_artists.append(triangle)

        self.writer.setup(self.fig, self.filename)

    def _triangle_coords(self, agent, size=0.3):
        """Return coordinates for a triangle pointing in agent.vel direction"""
        if np.linalg.norm(agent.vel) < 1e-8:
            direction = np.array([1.0, 0.0])
        else:
            direction = agent.vel / np.linalg.norm(agent.vel)
        perp = np.array([-direction[1], direction[0]])

        tip = agent.pos + direction * size
        base1 = agent.pos - direction * size * 0.5 + perp * size * 0.5
        base2 = agent.pos - direction * size * 0.5 - perp * size * 0.5
        return [tip, base1, base2]

    def capture_frame(self):
        # Clear all previous coordinates
        for patch, agent in zip(self.agent_artists, self.model.agent_list):
            patch.set_xy(self._triangle_coords(agent))

        if self.wrap_visualization:
            extra_patches = []
            W, H = self.model.space.width, self.model.space.height
            R = self.model.interaction_radius

            for agent in self.model.agent_list:
                pos = agent.pos
                shifts = []

                # Horizontal wrapping
                if pos[0] < R:
                    shifts.append(np.array([W, 0]))
                if pos[0] > W - R:
                    shifts.append(np.array([-W, 0]))
                # Vertical wrapping
                if pos[1] < R:
                    shifts.append(np.array([0, H]))
                if pos[1] > H - R:
                    shifts.append(np.array([0, -H]))
                # Diagonal combinations
                for dx, dy in [(s[0], s[1]) for s in shifts]:
                    coords = self._triangle_coords(agent)
                    coords_shifted = [c + np.array([dx, dy]) for c in coords]
                    triangle = Polygon(coords_shifted, color=agent.color)
                    self.ax.add_patch(triangle)
                    extra_patches.append(triangle)
            # Keep track so they can be removed in next frame
            self.extra_patches = extra_patches

        self.writer.grab_frame()
        # Remove extra patches to avoid accumulating
        if self.wrap_visualization:
            for p in self.extra_patches:
                p.remove()

    def close(self):
        self.writer.finish()

        if self.final_frame_png != "":    
            plt.savefig(self.final_frame_png, dpi=300)
       
        plt.close(self.fig)

# ---------------------------
# Agent class
# ---------------------------
class ContinuousAgent(Agent):
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        self.vel = np.zeros(2)
        self.last_sense_time = 0.0
        self.cached_neighbors = []
        self.color = np.random.rand(3,)

    def sense(self):
        self.cached_neighbors = self.model.space.get_neighbors(
            self.pos,
            self.model.interaction_radius + self.model.epsilon,
            include_center=False
        )
        self.last_sense_time = self.model.time

    def compute_force(self):
        force = np.zeros(2)

        for other in self.cached_neighbors:
            dvec = other.pos - self.pos

            # Toroidal distance
            for i, dim in enumerate([self.model.space.width, self.model.space.height]):
                if abs(dvec[i]) > dim / 2:
                    dvec[i] -= np.sign(dvec[i]) * dim

            dist = np.linalg.norm(dvec)
            if dist < 1e-12:
                continue

            direction = dvec / dist

            r_rep = self.model.repulsion_radius
            r_int = self.model.interaction_radius
            sigma = self.model.sigma
            delta = self.model.delta  # NEW smoothing parameter

            # --- Smooth cutoff functions ---
            def smooth_step(x):
                return 0.5 * (1.0 + np.tanh(x))

            # Repulsion smoothly turns off near r_rep
            repulsion_weight = smooth_step((r_rep - dist) / delta)

            # Attraction smoothly turns off near r_int
            attraction_weight = smooth_step((r_int - dist) / delta)

            # Softened repulsion force
            if dist < r_rep + delta:
                force -= (
                    repulsion_weight
                    * direction
                    * (r_rep - dist) / (dist + sigma)
                )

            # Smooth attraction force
            if dist > r_rep - delta and dist < r_int + delta:
                force += (
                    attraction_weight
                    * direction
                    * (dist - r_rep)
                )

        return force

    def step(self):
        if self.model.time - self.last_sense_time >= self.model.tau:
            self.sense()
        force = self.compute_force()
        self.vel += force * self.model.dt
        self.pos += self.vel * self.model.dt
        self.model.space.move_agent(self, self.pos)

    def distance_from_origin(self):
        return np.linalg.norm(self.pos)

# ---------------------------
# Flocking model
# ---------------------------
class FlockingModel(Model):
    def __init__(self, n_agents=30, width=10, height=10,
                 dt=0.05, tau=0.05, epsilon=0.05,
                 interaction_radius=2.0, repulsion_radius=0.5,
                 sigma=0.01, delta=0.01,
                record_video=False,
                video_filename="simulation.mp4",
                video_fps=30,
                wrap_visualization=False
                ):
        super().__init__()
        self.dt = dt
        self.tau = tau
        self.epsilon = epsilon
        self.time = 0.0
        self.interaction_radius = interaction_radius
        self.repulsion_radius = repulsion_radius
        self.sigma = sigma  # <--- new convergence parameter
        self.delta = delta
        self.space = ContinuousSpace(width, height, torus=True)

        self.agent_list = []
        for i in range(n_agents):
            agent = ContinuousAgent(i, self)
            self.agent_list.append(agent)
            random_pos = np.array([np.random.uniform(0, self.space.width),
                                   np.random.uniform(0, self.space.height)])
            self.space.place_agent(agent, random_pos)

        # Manual data collection
        self.time_data = []
        self.distance_data = []

        # Video
        self.record_video = record_video
        self.video = None
        if self.record_video:
            self.video = VideoRecorder(self, video_filename, video_fps, wrap_visualization)
            self.video.setup()

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
# Example usage
# ---------------------------
if __name__ == "__main__":

    # Set seed for reproducability
    np.random.seed(1)
    dt=1e-4
    sigma=0.04
    delta=0.05
    
    model = FlockingModel(
        n_agents=60,
        dt=dt, #0.02
        #epsilon=0.05, #0.05
        #tau=0.05, #0.05
        delta=delta,
        sigma=sigma,
        record_video=True,
        wrap_visualization=True,  # <- enable optional torus wrapping in video
        video_filename="flocking_triangles_wrapped.mp4",
        video_fps=30
    )

    total_time = 5
    n_steps = int(total_time/dt)

    for _ in range(n_steps):
        model.step()

    model.finalize()

    # Final total distance
    print("Final total distance from origin:", model.total_distance_from_origin())

    # Manual data collection DataFrame
    #import pandas as pd
    #df = pd.DataFrame(model.distance_data, index=model.time_data)
    #df.index.name = "time"
    #print(df.head())

    