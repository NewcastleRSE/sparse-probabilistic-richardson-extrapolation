##############################################################################
# Mulit-Agent Simulation Models using Mesa library
# https://mesa.readthedocs.io
# 
# It’s not an acronym. "Mesa" is just the English word mesa, meaning a flat-topped hill or plateau.
# A flat, stable platform to build agent-based models on.
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon, Circle
from mesa import Agent
from mesa import Model as MesaModel
from mesa.space import ContinuousSpace

# Application modules
from models.base_model import Model

# ---------------------------
# Video recorder helper
# ---------------------------
class VideoRecorder:
    def __init__(self, model, filename="simulation.mp4", fps=30, wrap_visualization=False, final_frame_png=""):
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
        self.ax.set_title("Multi-Agent Simulation")
        self.ax.set_aspect("equal") 

        # Create triangle for each agent
        for agent in self.model.agent_list:
            color = agent.color
            triangle = Polygon(self._triangle_coords(agent), color=color)
            self.ax.add_patch(triangle)
            self.agent_artists.append(triangle)
            
            # Add White Circle for agent 1
            if agent.unique_id == 1:                
                circle = Circle(agent.pos, radius=0.05, facecolor="white", edgecolor=None, zorder=triangle.get_zorder() + 1)
                self.ax.add_patch(circle)
                self.circle = circle
        
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

        # Update circle
        #self.circle.set_xy(self.model.agent_list[0].pos)

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
        # Save final state if req'd
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
        if unique_id == 0:
            self.color = [0, 0, 0]


    def sense(self):
        self.cached_neighbors = self.model.space.get_neighbors(
            self.pos,
            self.model.interaction_radius + self.model.neighbour_margin,
            include_center=False
        )
        self.last_sense_time = self.model.time

    # --- Smooth cutoff functions ---
    def smooth_step(self, x):
        return 0.5 * (1.0 + np.tanh(x))
        
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
            repulsion_softening = self.model.repulsion_softening
            
            transition_width = self.model.transition_width  # smoothing parameter

            # Softened repulsion force
            if dist < r_rep + transition_width:
                # Repulsion smoothly turns off near r_rep         
                repulsion_weight = self.smooth_step((r_rep - dist) / transition_width)

                force -= (
                      repulsion_weight
                    * direction
                    * (r_rep - dist) / (dist + repulsion_softening)
                )

            # Smooth attraction force
            if dist > r_rep - transition_width and dist < r_int + transition_width:
                # Attraction smoothly turns off near r_int
                attraction_weight = self.smooth_step((r_int - dist) / transition_width)

                force += (                  
                      attraction_weight
                    * direction
                    * (dist - r_rep)
                )

        return force

    def local_mean_velocity(self):
        """Return the mean velocity of cached neighbours."""
        if not self.cached_neighbors:
            return self.vel.copy()

        v_sum = np.zeros(2)
        for other in self.cached_neighbors:
            v_sum += other.vel

        return v_sum / len(self.cached_neighbors)

    def step(self):
        if self.model.time - self.last_sense_time >= self.model.sense_interval:
            self.sense()

        force = self.compute_force()

        # Alignment relaxation (chaos control)
        if self.model.alignment_strength > 0.0:
            v_mean = self.local_mean_velocity()
            force += self.model.alignment_strength * (v_mean - self.vel)

        self.vel += force * self.model.dt
        self.pos += self.vel * self.model.dt
        self.model.space.move_agent(self, self.pos)

    def distance_from_origin(self):
        return np.linalg.norm(self.pos)

# ---------------------------
# Flocking model
# ---------------------------
class FlockingModel(MesaModel):
    def __init__(self, n_agents=30, width=10, height=10,
                 dt=0.05, sense_interval=0.05, neighbour_margin=0.05,
                 alignment_strength=0,
                 interaction_radius=2.0, repulsion_radius=0.5,
                 repulsion_softening=0.01, transition_width=0.01,              
                record_video=False,
                video_filename="simulation.mp4",
                video_fps=30,
                wrap_visualization=False,
                final_frame_png=""
                ):
        super().__init__()
        self.dt = dt
        self.sense_interval = sense_interval
        self.neighbour_margin = neighbour_margin
        self.alignment_strength = alignment_strength
        self.time = 0.0
        self.interaction_radius = interaction_radius
        self.repulsion_radius = repulsion_radius
        self.repulsion_softening = repulsion_softening  
        self.transition_width = transition_width      
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
            self.video = VideoRecorder(self, video_filename, video_fps, wrap_visualization, final_frame_png)
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
    
    def distance_from_origin(self, agent_num):
        return self.agent_list[agent_num].distance_from_origin()

class MultiAgentModel(Model):
    """
    Class for multi-agent modelling using the Mesa python library.
    https://mesa.readthedocs.io

    Inherits Model class for use with SPRE analysis.
    """

    def __init__(self, params, parameter_filename, skip_true_value_calc : bool = False):
        """
        Sets up the multi-agent model class with model parameters.

        Parameters:  
            params : dict               Parameters for the model.
            parameter_filename : str    Filename and path of the file. 
            skip_true_value_calc : bool Skip evaulation of the true value (if not doing SPRE)     
        Returns:
            None         
        """
     
        # Default model parameters - can be overwritten in parameter file
        self.total_time = 5
        self.seed = 1
        self.n_agents = 60

        self.alignment_strength = 0
        self.neighbour_margin = 0.05
        self.sense_interval = 0.05

        self.transition_width = 0.05

        # Decide which of the parameters to use
        self.use_dt = True
        self.use_repulsion_softening = True
        self.use_transition_width = True

        # Default fixed values for parameters if not being used
        self.dt = 0.02
        self.repulsion_softening = 0.05
    
        self.final_model_plot_filename = ""
        self.final_mp4_filename = ""

        # General parameters for SPRE analysis
        ######################################

        # Use model f_z(x) = f(z+x)
        self.use_offset_model = False
      
        # Set model name    
        self.model_name = "Multi-Agent Flocking"

        # Set initial model description       
        self.description = "Multi-Agent Flocking Model"

        # Set model parameters from parameter file
        ##########################################

        # Call Parent’s constructor to set parameters - and overwrite any above here but not below
        super().__init__(params, parameter_filename)
      
        # Set default model parameters
        if self.final_mp4_filename is not None and self.final_mp4_filename != "":
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False
       
        # Do plot of final state
        if self.final_model_plot_filename is not None and self.final_model_plot_filename != "":
            self.do_final_model_plot = True

        # Set true value
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()
    
    def run_model_simulation(self, discrete_paras):
        """
        Simulates multi-agent model using the Mesa library:
        https://mujoco.readthedocs.io/

        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.           
        Returns:
            float   
        """
   
        # Set discretisation parameters
        i = 0
        if self.use_dt:
            dt = discrete_paras[i]
            i += 1
        else:
            dt = self.dt

        if self.use_repulsion_softening:
            repulsion_softening = discrete_paras[i]
            i += 1
        else:
            repulsion_softening = self.repulsion_softening

        if self.use_transition_width:
            transition_width = discrete_paras[i]
            i += 1
        else:
            transition_width = self.transition_width
      

        # repulsion_softening is a short-range regularisation length that prevents singular interaction forces at very small agent separations,
        # with the model converging to the point-particle limit as `repulsion_softening → 0`.

        # transition_width is a smoothing width that regularises the interaction cutoffs,
        # ensuring forces transition smoothly at the interaction radii and converge to the sharp cutoff model as `transition_width → 0`.

        # Output info on what is being simulated
        print(f"\tSimulating {self.description} with dt = {dt}, repulsion_softening = {repulsion_softening}, transition_width = {transition_width}, alignment_strength = {self.alignment_strength}")
 
        # Set seed for reproducability, agent added randomly
        np.random.seed(self.seed)

        model = FlockingModel(
            n_agents=self.n_agents,
            dt=dt, 
            neighbour_margin=self.neighbour_margin, 
            sense_interval=self.sense_interval, 
            alignment_strength=self.alignment_strength,
            repulsion_softening=repulsion_softening,
            transition_width=transition_width,
            record_video=self.save_animation,
            wrap_visualization = True,
            video_filename=self.final_mp4_filename,
            video_fps=30,
            final_frame_png=self.final_model_plot_filename
        )

        n_steps = int(np.floor(self.total_time/dt)) + 1

        # Do steps 1 to n_steps
        for step in range(1, n_steps + 1):
            model.step()
            # Record last two values to interpolate to estimate value at exactly total time
            if step == n_steps - 1:
                distance_1 = model.distance_from_origin(0)
            elif step == n_steps:
                distance_2 = model.distance_from_origin(0)

        model.finalize()
  
        # interpolate final result
        frac = (self.total_time - (dt * (n_steps - 1)))/dt
        distance = distance_1*(1 - frac) + distance_2*frac

        # Final total distance        
        print("Final distance from origin of agent 1:", distance)
        
        return distance

    def plot_final_model(self):
        """
        Plots the final simulated model which is in this case is a mp4 video if req'd.
        Handled in model simulation.

        Parameters:  
            None
        Returns:
            None                
        """
            
        if not self.save_animation and self.final_model_plot_filename != "":
            print("A video must be produced to take the last frame as a picture.")
        elif self.save_animation:            
            print(f"Video output to: {self.final_mp4_filename}")
            if self.final_model_plot_filename != "":
                print(f"Final frame output to: {self.final_model_plot_filename}")

    def get_cache_filename(self, discrete_paras : npt.NDArray):
        """
        Returns the model cache filename for the multi-agent model based on the model parameters.

        Parameters:  
            discrete_paras : npt.NDArray   Discretisation parameters
        Returns:
            str    
        """
      
        # Create filename with all settings and parameters used
        filename = f"ma_{self.total_time}_{self.seed}_{self.n_agents}_{self.neighbour_margin}_{self.sense_interval}_"
        filename += f"{self.use_dt}_{self.use_repulsion_softening}_{self.use_transition_width}_"
        filename += f"{self.dt}_{self.repulsion_softening}_{self.transition_width}_"
        
        if self.alignment_strength > 0:
            filename += f"{self.alignment_strength}_"

        if self.use_offset_model:
            filename += "_".join(str(i) for i in self.final_tols) + "_"

        filename += "_".join(str(i) for i in discrete_paras) + ".bin"

        # Shorten if too long
        if len(filename) > 99:
            filename = filename.replace("True", "T").replace("False", "F").replace("e", "").replace("ma_", "m")

        return filename
