##############################################################################
# Mulit-Agent Simulation Models using Mesa library
# https://mesa.readthedocs.io
# 
# Meaning of Mesa: It’s not an acronym, "Mesa" is just the English word mesa, 
# meaning a flat-topped hill or plateau. A flat, stable platform to build agent-based models on.
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D
from mesa import Agent
from mesa import Model as MesaModel
from mesa.space import ContinuousSpace

# Application modules
from models.base_model import Model

# To set LaTeX fonts later
import matplotlib as mpl

# Video recorder helper class
class VideoRecorder:
    """
    Handles recording and saving of a flocking simulation as a video,
    with optional periodic wrapping visualization and final frame export.
    """

    def __init__(
        self, model, filename: str = "simulation.mp4",
        fps: int = 30,
        wrap_visualization: bool = False,
        initial_frame_png: str = "",
        final_frame_png: str = "",
        trail_length: int = 8,
        trail_spacing: float = 0.2
    ) -> None:
        """
        Initialise the video recorder.

        Parameters:
            model                   Simulation model instance
            filename : str          Output video filename
            fps : int               Frames per second
            wrap_visualization : bool
                                    Whether to visualise periodic boundary wrapping
            initial_frame_png : str Optional filename for saving initial frame as PNG
            final_frame_png : str   Optional filename for saving final frame as PNG
            trail_length : int      Trail length of agents, set to 0 for no trails
            trail_spacing : float   Minimum physical distance between trail points
        """

        self.model = model
        self.filename = filename
        self.fps = fps
        self.wrap_visualization = wrap_visualization
        self.initial_frame_png = initial_frame_png
        self.final_frame_png = final_frame_png

        # Trails
        self.show_trails = (trail_length > 0)
        self.trail_length = trail_length

        # Per-agent position history for trails
        self.position_history = {agent.unique_id: [] for agent in model.agent_list}

        # Per-agent trail line artists
        self.trail_artists = {agent.unique_id: [] for agent in model.agent_list}

        self.trail_spacing = trail_spacing
        self.distance_since_last_sample = {
            agent.unique_id: 0.0 for agent in model.agent_list
        }

        self.fig, self.ax = plt.subplots()
        self.writer = FFMpegWriter(fps=fps)
        self.agent_artists = []


    def setup(self) -> None:
        """
        Initialise the matplotlib figure and create agent patches.

        Parameters:
            None
        Returns:
            None  
        """

        self.ax.set_xlim(0, self.model.space.width)
        self.ax.set_ylim(0, self.model.space.height)
        self.ax.set_title("Flock Simulation")
        self.ax.set_aspect("equal")
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
      
        # Create triangle patch for each agent
        for agent in self.model.agent_list:
            color = agent.color
            triangle = Polygon(self._triangle_coords(agent), color=color)
            self.ax.add_patch(triangle)
            self.agent_artists.append(triangle)

            # Add identifying white circle for agent with unique_id == 0
            # (drawn above triangle using z-order)
            if agent.unique_id == 0:
                circle = Circle(
                    agent.pos,
                    radius=0.05,
                    facecolor="white",
                    edgecolor=None,
                    zorder=triangle.get_zorder() + 1
                )
                self.ax.add_patch(circle)
                self.circle = circle

        # Initialise FFmpeg writer
        self.writer.setup(self.fig, self.filename)


    def _triangle_coords(self, agent, size: float = 0.3) -> list:
        """
        Compute the coordinates of a triangle oriented in the direction
        of the agent's velocity.

        Parameters:
            agent                 Agent instance
            size : float          Triangle size scale
        Returns:
            list                  List of 3 coordinate arrays
        """

        # Default direction if agent is almost stationary
        if np.linalg.norm(agent.vel) < 1e-8:
            direction = np.array([1.0, 0.0])
        else:
            direction = agent.vel / np.linalg.norm(agent.vel)

        # Perpendicular vector for triangle base
        perp = np.array([-direction[1], direction[0]])

        tip = agent.pos + direction * size
        base1 = agent.pos - direction * size * 0.5 + perp * size * 0.5
        base2 = agent.pos - direction * size * 0.5 - perp * size * 0.5

        return [tip, base1, base2]


    def capture_frame(self) -> None:
        """
        Update all agent patches and capture the current frame.

        Parameters:
            None
        Returns:
            None  
        """

        # Update main agent triangles
        for patch, agent in zip(self.agent_artists, self.model.agent_list):
            patch.set_xy(self._triangle_coords(agent))

        # Update white circle of first agent                           
        self.circle.center = self.model.agent_list[0].pos

        # Handle periodic wrapping visualisation
        if self.wrap_visualization:
                                                            
            extra_patches = []

            W, H = self.model.space.width, self.model.space.height
            R = self.model.interaction_radius

            for agent in self.model.agent_list:

                pos = agent.pos
                shifts = []

                # Detect proximity to horizontal boundaries
                if pos[0] < R:
                    shifts.append(np.array([W, 0]))
                if pos[0] > W - R:
                    shifts.append(np.array([-W, 0]))

                # Detect proximity to vertical boundaries
                if pos[1] < R:
                    shifts.append(np.array([0, H]))
                if pos[1] > H - R:
                    shifts.append(np.array([0, -H]))

                # Create wrapped copies for all detected shifts
                for dx, dy in [(s[0], s[1]) for s in shifts]:

                    coords = self._triangle_coords(agent)
                    coords_shifted = [c + np.array([dx, dy]) for c in coords]

                    triangle = Polygon(coords_shifted, color=agent.color)

                    self.ax.add_patch(triangle)
                    extra_patches.append(triangle)

            # Store extra patches for cleanup after frame capture
            self.extra_patches = extra_patches

        # Update agent trails
        if self.show_trails:
            for agent in self.model.agent_list:
                # Draw wrapped short dashed segments
                uid = agent.unique_id

                history = self.position_history[uid]

                # Update history
                if len(history) == 0:
                    history.append(agent.pos.copy())
                else:
                    last_pos = history[-1]

                    dx = agent.pos[0] - last_pos[0]
                    dy = agent.pos[1] - last_pos[1]

                    # Minimum image correction
                    W = self.model.space.width
                    H = self.model.space.height

                    if dx > W / 2:
                        dx -= W
                    elif dx < -W / 2:
                        dx += W

                    if dy > H / 2:
                        dy -= H
                    elif dy < -H / 2:
                        dy += H

                    distance = np.sqrt(dx**2 + dy**2)

                    if distance >= self.trail_spacing:
                        history.append(agent.pos.copy())

                        if len(history) > self.trail_length:
                            history.pop(0)

                # Remove old trail segments
                for line in self.trail_artists[uid]:
                    line.remove()
                self.trail_artists[uid].clear()

                W = self.model.space.width
                H = self.model.space.height

                for i in range(1, len(history)):

                    p0 = history[i - 1]
                    p1 = history[i]

                    dx = p1[0] - p0[0]
                    dy = p1[1] - p0[1]

                    # Minimum image convention (periodic wrapping)
                    if dx > W / 2:
                        dx -= W
                    elif dx < -W / 2:
                        dx += W

                    if dy > H / 2:
                        dy -= H
                    elif dy < -H / 2:
                        dy += H

                    # Draw wrapped segment
                    x_vals = [p0[0], p0[0] + dx]
                    y_vals = [p0[1], p0[1] + dy]

                    line = Line2D(
                        x_vals,
                        y_vals,
                        color=agent.color,
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.5,
                        zorder=0,
                    )

                    self.ax.add_line(line)
                    self.trail_artists[uid].append(line)
                                
        # Write frame to video
        self.writer.grab_frame()

        # Remove wrapped copies to avoid accumulation
        if self.wrap_visualization:
            for p in self.extra_patches:
                p.remove()


    def set_latex_fonts(self) -> None:
        """
        Set LaTeX fonts for use with plots.

        Parameters:
            None
        Returns:
            None  
        """

        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 14,
            "font.size": 14,
            "legend.fontsize": 12,
        })
        
    def save_initial_frame(self) -> None:
        """
        Save the initial frame.

        Parameters:
            None
        Returns:
            None  
        """

        # Set title
        self.ax.set_title(r"Flock Simulation, $t=0$")

        # Save final frame if requested
        if self.initial_frame_png != "":
            self.set_latex_fonts()
            plt.savefig(self.initial_frame_png, dpi=300)
            print(f"Initial frame output to {self.initial_frame_png}")

        # Set title back for video
        self.ax.set_title("Flock Simulation")
    
    def close(self, total_time : float) -> None:
        """
        Finalise video writing and optionally save the last frame.

        Parameters:
            total_time : float   Time to write in title of simulation plt
        Returns:
            None
        """

        # Finish writing video
        self.writer.finish()

        # Set title
        self.ax.set_title(fr"Flock Simulation, $t={total_time}$")

        # Save final frame if requested
        if self.final_frame_png != "":
            self.set_latex_fonts()
            plt.savefig(self.final_frame_png, dpi=300)

        # Close matplotlib figure
        plt.close(self.fig)

# Agent class
class ContinuousAgent(Agent):
    """
    Continuous-space agent implementing smooth interaction dynamics
    for flocking-style simulations.
    """

    def __init__(self, unique_id: int, model) -> None:
        """
        Initialise a continuous agent.

        Parameters:
            unique_id : int    Unique identifier for the agent
            model               Parent model instance
        Returns:
            None
        """

        self.unique_id = unique_id
        self.model = model

        # Position is initialised externally by the model
        self.pos = None

        # Agent velocity vector
        self.vel = np.zeros(2)

        # Cached neighbour list (not currently used for filtering)
        self.cached_neighbors = []

        # Random RGB colour for visualisation
        self.color = np.random.rand(3,)

        # Highlight first agent in black
        if unique_id == 0:
            self.color = [0, 0, 0]


    def soft_cutoff(self, dist: float) -> float:
        """
        Compute smooth interaction weight based on inter-agent distance.

        Parameters:
            dist : float    Distance to another agent
        Returns:
            float           Interaction weight in [0, 1]
        """

        R = self.model.interaction_radius
        w = self.model.cutoff_width

        # Special case: recover hard cutoff when no smoothing is applied
        if w == 0.0:
            return 1.0 if dist < R else 0.0

        # Smooth transition using rescaled hyperbolic tangent
        return 0.5 * (1.0 - np.tanh((dist - R) / w))


    def compute_force(self) -> np.ndarray:
        """
        Compute the total interaction force acting on this agent.

        Parameters:
            None
        Returns:
            np.ndarray      Resultant 2D force vector
        """

        force = np.zeros(2)

        # Loop over all agents (soft cutoff handles interaction strength)
        for other in self.model.agent_list:

            # Do not interact with self
            if other is self:
                continue

            dvec = other.pos - self.pos

            # Apply periodic boundary corrections (toroidal domain)
            for i, dim in enumerate(
                [self.model.space.width, self.model.space.height]
            ):
                if abs(dvec[i]) > dim / 2:
                    dvec[i] -= np.sign(dvec[i]) * dim

            dist = np.linalg.norm(dvec)

            # Unit vector pointing towards neighbour
            direction = dvec / dist

            # Distance-based interaction weighting
            w = self.soft_cutoff(dist)

            # Repulsive contribution (softened near zero distance)
            rep = -(self.model.repulsion_radius - dist) / (
                dist + self.model.repulsion_softening
            )
            rep = max(rep, 0.0)

            # Attractive contribution beyond repulsion radius
            att = (dist - self.model.repulsion_radius)
            att = max(att, 0.0)

            # Combined pairwise force vector
            f = (rep + att) * direction

            # Accumulate weighted force
            force += w * f

        return force


    def step(self) -> None:
        """
        Advance the agent state by one time step.

        Parameters:
            None
        Returns:
            None  
        """

        # Compute interaction force
        force = self.compute_force()

        # Explicit Euler update of velocity and position
        self.vel += force * self.model.dt
        self.pos += self.vel * self.model.dt

        # Update position in the model's spatial structure
        self.model.space.move_agent(self, self.pos)


    def distance_from_origin(self) -> float:
        """
        Compute Euclidean distance from the origin.

        Parameters:
            None
        Returns:
            float   Distance from (0, 0)
        """

        return np.linalg.norm(self.pos)


# Flocking model
class FlockingModel(MesaModel):
    """
    Continuous-space flocking model with smooth interaction forces
    and optional video recording.
    """

    def __init__(
        self,
        n_agents: int = 30,
        width: float = 10,
        height: float = 10,  
        dt: float = 0.02,    
        interaction_radius: float = 2.0,
        repulsion_radius: float = 0.5,
        repulsion_softening: float = 0.01,
        cutoff_width: float = 0.05,
        record_video: bool = False,
        video_filename: str = "simulation.mp4",
        video_fps: int = 30,
        wrap_visualization: bool = False,
        initial_frame_png: str = "",
        final_frame_png: str = "",
        trail_length : int = 0
    ) -> None:
        """
        Initialise the flocking model.

        Parameters:
            n_agents : int              Number of agents
            width : float              Width of simulation domain
            height : float             Height of simulation domain
            dt : float                 Time step size
            interaction_radius : float Maximum interaction distance
            repulsion_radius : float   Repulsion zone radius
            repulsion_softening : float
                                       Softening parameter for short-range repulsion
            cutoff_width : float       Smooth interaction cutoff width
            record_video : bool        Whether to record a simulation video
            video_filename : str       Output video filename
            video_fps : int            Video frames per second
            wrap_visualization : bool  Show periodic boundary wrapping
            initial_frame_png : str    Optional initial frame output filename
            final_frame_png : str      Optional final frame output filename
            trail_length : int         Number of trails to show after agent
        """

        super().__init__()

        # Time-stepping parameters. self.time is used by Mesa parent class, so use self.sim_time instead.
        self.dt = dt
        self.sim_time = 0.0

        # Interaction parameters
        self.interaction_radius = interaction_radius
        self.repulsion_radius = repulsion_radius
        self.repulsion_softening = repulsion_softening
        self.cutoff_width = cutoff_width

        # Continuous periodic spatial domain
        self.space = ContinuousSpace(width, height, torus=True)

        # Create and place agents
        self.agent_list = []

        for i in range(n_agents):

            agent = ContinuousAgent(i, self)
            self.agent_list.append(agent)

            # Random initial position in domain
            random_pos = np.array([
                np.random.uniform(0, self.space.width),
                np.random.uniform(0, self.space.height)
            ])

            self.space.place_agent(agent, random_pos)

        # Video recording setup
        self.record_video = record_video
        self.video = None

        if self.record_video:
            self.video = VideoRecorder(
                self,
                video_filename,
                video_fps,
                wrap_visualization,
                initial_frame_png,
                final_frame_png,
                trail_length
            )
            self.video.setup()


    def step(self) -> None:
        """
        Advance the model by one time step.

        Parameters:
            None
        Returns:
            None        
        """
      
        # Update all agents
        for agent in self.agent_list:
            agent.step()

        # Capture video frame if enabled
        if self.record_video:
            self.video.capture_frame()
            # Save initial frame if filename set to do so
            if self.sim_time == 0:
                self.video.save_initial_frame()

        # Advance time in case we need it       
        self.sim_time += self.dt

    def finalize(self, total_time : float) -> None:
        """
        Finalise model execution and close video writer if needed.

        Parameters:
            total_time : float    Time to write for title of simulation plot
        Returns:
            None
        """

        if self.record_video:
            self.video.close(total_time)

    def total_distance_from_origin(self) -> float:
        """
        Compute total distance of all agents from the origin.

        Returns:
            float   Sum of Euclidean distances
        """

        return sum(
            agent.distance_from_origin()
            for agent in self.agent_list
        )


    def distance_from_origin(self, agent_num: int) -> float:
        """
        Compute distance of a specified agent from the origin.

        Parameters:
            agent_num : int    Agent index

        Returns:
            float              Distance from (0, 0)
        """

        return self.agent_list[agent_num].distance_from_origin()


# Flock model class
class FlockModel(Model):
    """
    Multi-agent flocking model using the Mesa library.
    Compatible with SPRE/Richardson-style analysis.
    """

    def __init__(self, params: dict, parameter_filename: str, skip_true_value_calc: bool = False) -> None:
        """
        Sets up the multi-agent model with default and user-specified parameters.

        Parameters:  
            params : dict               Dictionary of model parameters
            parameter_filename : str    Path to parameter file
            skip_true_value_calc : bool Skip evaluation of the true value (if not doing SPRE)
        """

        # Default model parameters (can be overwritten in parameter file)
        self.total_time = 5
        self.seed = 1
        self.n_agents = 60
        self.cutoff_width = 0.05

        # Flags indicating which parameters are part of SPRE evaluation
        self.use_dt = True
        self.use_repulsion_softening = True
        self.use_cutoff_width = True

        # Default fixed values for parameters if not being used in SPRE
        self.dt = 0.02
        self.repulsion_softening = 0.05
        self.cutoff_width = 0.05

        # Filenames for plotting / video output
        self.initial_model_plot_filename = ""
        self.final_model_plot_filename = ""
        self.final_mp4_filename = ""
        self.trail_length = 0

        # SPRE general parameters
        self.use_offset_model = False  # Whether to simulate f(z+x) for SPRE, so to use model f_z(x) = f(z+x)
        self.model_name = "Flock"     # Model identifier
        self.description = "Flock Model"

        # Call parent constructor to load parameter file and overwrite defaults
        super().__init__(params, parameter_filename)

        # Decide whether to produce animation and final plot
        if self.final_mp4_filename:
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False

        if self.final_model_plot_filename:
            self.do_final_model_plot = True

        # Compute "true value" if needed for SPRE
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()


    def run_model_simulation(self, discrete_paras: npt.NDArray) -> float:
        """
        Simulate the multi-agent model for a given set of discretisation parameters.
        Uses the Mesa library:   https://mujoco.readthedocs.io/

        Parameters:  
            discrete_paras : NDArray   Discretisation parameters to use
        Returns:
            float   Final distance from origin of first agent
        """

        # Assign parameter values from SPRE input
        i = 0
        if self.use_dt:
            dt = discrete_paras[i]
            i += 1
        else:
            dt = self.dt

        # repulsion_softening is a short-range regularisation length that prevents singular interaction forces at very 
        # small agent separations, with the model converging to the point-particle limit as repulsion_softening tends to 0.
        if self.use_repulsion_softening:
            repulsion_softening = discrete_paras[i]
            i += 1
        else:
            repulsion_softening = self.repulsion_softening

        # cutoff_width controls how gradually the interaction forces switch on and off near their cutoff distances,
        # smoothing sharp transitions to improve numerical convergence.
        if self.use_cutoff_width:
            cutoff_width = discrete_paras[i]
            i += 1
        else:
            cutoff_width = self.cutoff_width

        # Informative output for user
        print(f"\tSimulating {self.description} with dt = {dt}, "
              f"repulsion_softening = {repulsion_softening}, cutoff_width = {cutoff_width}")

        # Set random seed for reproducibility
        np.random.seed(self.seed)

        # Create FlockingModel instance
        model = FlockingModel(
            n_agents=self.n_agents,
            dt=dt,
            repulsion_softening=repulsion_softening,
            cutoff_width=cutoff_width,
            record_video=self.save_animation,
            wrap_visualization=True,
            video_filename=self.final_mp4_filename,
            video_fps=30,
            initial_frame_png=self.initial_model_plot_filename,
            final_frame_png=self.final_model_plot_filename,
            trail_length=self.trail_length
        )

        n_steps = int(np.floor(self.total_time / dt)) + 1

        # Run simulation
        for step in range(1, n_steps + 1):
            
            model.step()

            # Record last two steps to interpolate distance at exact final time
            if step == n_steps - 1:
                distance_1 = model.distance_from_origin(0)
            elif step == n_steps:
                distance_2 = model.distance_from_origin(0)

        # Finalize model (close video if any)
        model.finalize(self.total_time)

        # Linear interpolation to total_time
        frac = (self.total_time - (dt * (n_steps - 1))) / dt
        distance = distance_1 * (1 - frac) + distance_2 * frac

        # Report result
        print("Final distance from origin of first agent:", distance)

        return distance


    def plot_final_model(self) -> None:
        """
        Report output filenames for final frame and video.

        Note that actual plotting is handled by FlockingModel video output.

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


    def get_cache_filename(self, discrete_paras: npt.NDArray[np.float64]) -> str:
        """
        Generate a unique cache filename based on model parameters and discretisation values.

        Parameters:  
            discrete_paras : NDArray[np.float64]   Discretisation parameters
        Returns:
            str    Filename string
        """

        # Start with model-specific identifiers
        filename = f"f_{self.total_time}_{self.seed}_{self.n_agents}_"
        filename += f"{self.use_dt}_{self.use_repulsion_softening}_{self.use_cutoff_width}_"

        if self.use_dt:
            filename += f"{self.dt}_"
        if self.use_repulsion_softening:
            filename += f"{self.repulsion_softening}_"
        if self.cutoff_width:
            filename += f"{self.cutoff_width}_"

        # Include offset model tolerances if used
        if self.use_offset_model:
            filename += "_".join(str(i) for i in self.final_tols) + "_"

        # Add discretisation parameters
        filename += "_".join(str(i) for i in discrete_paras) + ".bin"

        # Shorten name to avoid possible problems with too long names
        filename = filename.replace("True", "T").replace("False", "F").replace("e", "")

        return filename
