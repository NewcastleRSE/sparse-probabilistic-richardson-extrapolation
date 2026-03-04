##############################################################################
# Simulation Models using MuJoCo, Physics Modelling
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import numpy as np
import numpy.typing as npt
from pathlib import Path
import mujoco
import imageio

# Application modules
from models.base_model import Model

class MujocoModel(Model):
    """
    Class for Physics model using the MuJoCo (Multi-Joint dynamics with Contact) python library.
    https://mujoco.readthedocs.io/
    """

    def __init__(self, params, parameter_filename, skip_true_value_calc : bool = False):
        """
        Sets up the physics model class with model parameters.

        Parameters:  
            params : dict               Parameters for the model.
            parameter_filename : str    Filename and path of the file. 
            skip_true_value_calc : bool Skip evaulation of the true value (if not doing SPRE)     
        Returns:
            None         
        """
     
        self.total_time = 600.0
        # To decided when have objects stopped moving
        self.velocity_thresh = 1e-15
        self.steps_required_to_stop = 30

        # Use model f_z(x) = f(z+x)
        self.use_offset_model = False
        self.use_fixed_time = False
      
        # Set default camera parameters
        self.camera_position = np.array([2, -2, 1.5])
        # Smaller is closer to the object
        self.camera_distance_scale = 0.5                
        self.fps = 60

        # Default values for parameters if not set
        self.dt = 0.01
        self.solver_reference = 1e-6
        self.solver_impedance = 1e-6 
        
        # Whether to use these discrete parameters
        self.use_dt = True
        self.use_solver_reference = True
        self.use_solver_impedance = True
     
        # Set initial model description       
        self.description = "MuJoCo Physics Model"

        # Call Parent’s constructor to set parameters - and overwrite any above here but not below
        super().__init__(params, parameter_filename)

        # Set initial model name    
        self.model_name = "Mujoco"

        # Set default model parameters
        if self.final_mp4_filename is not None and self.final_mp4_filename != "":
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False
       
        # Set true value
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()
     
    def setup_model_world(self, dt : float, solver_reference : float, solver_impedance : float):
        """
        Sets up world in MuJoCo to simulate model.

        Parameters:  
            dt : float                 Timestep          
            solver_reference : float   Reference dynamics for correcting constraint errors
            solver_impedance : float   Effective stiffness and softness of the constraint
        Returns:
            None  
        """

        # Set camera zoom
        camera_pos = self.camera_position * self.camera_distance_scale

        # Read MJCF from file, XML file with the model setup
        # Set the directory of the file
        model_file = str(Path(self.cache_dir).parent / self.model_file)
      
        with open(model_file, "r") as f:            
            mjcf_str = f.read()

        # Insert timestep, impratio and camera position into MJCF
        mjcf = mjcf_str.format(timestep=dt, solver_reference=solver_reference, solver_impedance=solver_impedance, camera_pos_x=camera_pos[0], camera_pos_y=camera_pos[1], camera_pos_z=camera_pos[2])

        # Load model and data
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)

        # Set initial velocities for objects
        for body_id in range(self.model.nbody):

            # How many joints does this body have?
            njnt = self.model.body_jntnum[body_id]
            if njnt == 0:
                continue  # static body (e.g. floor)

            # First joint index for this body
            jntid = self.model.body_jntadr[body_id]

            # DOF start index in qvel
            dofadr = self.model.jnt_dofadr[jntid]

            # Read velocities from XML user field
            user_vals = self.model.body_user[body_id]

            if user_vals is not None and len(user_vals) >= 6:
                self.data.qvel[dofadr:dofadr+6] = np.array(user_vals[:6])

    def run_model_simulation(self, discrete_paras : npt.NDArray) -> float:
        """
        Uses MuJoCo (Multi-Joint dynamics with Contact) Python library to simulate scenario as given in XML setup file.
        https://mujoco.readthedocs.io/

        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.           
        Returns:
            float   
        """
   
        if self.use_fixed_time:
            return self.run_model_simulation_fixed_time(discrete_paras)
        else:
            return self.run_model_simulation_resting(discrete_paras)

    def get_discrete_parameter_value(self, discrete_paras : npt.NDArray, i : int, default_value : float, use_this_parameter : bool) -> tuple:
        """
        Set discrete parameter.

        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.  
            i : int                          Index of discrete parameter to use
            default_value : float            Default value if not using parameter value
            use_this_parameter : bool        Whether to use this discrete parameter         
        Returns:
            float   
        """
         
        if use_this_parameter:
            val = discrete_paras[i]
            i += 1
        else:
            val = default_value

        return val, i

    def run_model_simulation_fixed_time(self, discrete_paras):
        """
        Uses MuJoCo (Multi-Joint dynamics with Contact) Python library to simulate scenario as given in XML setup file.
        https://mujoco.readthedocs.io/

        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.           
        Returns:
            float   
        """
   
        # Set discretisation parameters
        # Assign parameter values from SPRE input
        i = 0
        dt, i = self.get_discrete_parameter_value(discrete_paras, i, self.dt, self.use_dt)
        solver_reference, i = self.get_discrete_parameter_value(discrete_paras, i, self.solver_reference, self.use_solver_reference)
        solver_impedance, i = self.get_discrete_parameter_value(discrete_paras, i, self.solver_impedance, self.use_solver_impedance)
     
       
        # Output info on what is being simulated
        print(f"\tSimulating {self.description} with dt = {dt}, solver reference = {solver_reference} and solver impedance = {solver_impedance}")
 
        # Setup model world
        self.setup_model_world(dt, solver_reference, solver_impedance)

        # Total time is used as an upper limit all objects should come to rest well before this
        steps = int(self.total_time / self.model.opt.timestep)

        # Set up if creating a video
        if self.save_animation:
            # Renderer
            renderer = mujoco.Renderer(self.model, width=640, height=480)
            self.frames = []
            frame_interval = int(1.0 / (self.fps * self.model.opt.timestep))
            if frame_interval == 0:
                frame_interval = 1
                print("Warning: frame interval too small, set a smaller time step!")

        # Get all body IDs, only include bodies with joints (movable bodies)
        body_ids = [i for i in range(self.model.nbody) if self.model.body_jntadr[i] != -1]
      
        n_steps = int(np.floor(self.total_time / dt)) + 1

        # Run simulation
        for step in range(1, n_steps + 1):
            mujoco.mj_step(self.model, self.data)

            # Save frames for video if creating one
            if self.save_animation and step % frame_interval == 0:
                renderer.update_scene(self.data, camera="angled_view")
                frame = renderer.render()
                self.frames.append(frame)

            # Record last two steps to interpolate distance at exact final time
            if step == n_steps - 1:
                distance_1 = self.distance_from_origin(body_ids[0])
            elif step == n_steps:
                distance_2 = self.distance_from_origin(body_ids[0])

        # Linear interpolation to total_time
        frac = (self.total_time - (dt * (n_steps - 1))) / dt
        distance = distance_1 * (1 - frac) + distance_2 * frac
        
        print(f"\tCalculated final value: {distance}")
        return distance
    

    def distance_from_origin(self, body_id: int) -> float:
        """
        Compute distance of a specified body from the origin.

        Parameters:
            body_id : int      Body index

        Returns:
            float              Distance from (0, 0)
        """
       
        pos = self.data.xpos[body_id]  # world position of body
        distance = np.linalg.norm(pos)
       
        return distance
    
    def total_distance_from_origin(self, body_ids: list[int]):
        """
        Compute total distance of specified bodys from the origin.

        Parameters:
            body_id : int      Body index

        Returns:
            float              Total distances from (0, 0)
        """

        total_distance = 0.0
        for body_id in body_ids:         
            total_distance += self.distance_from_origin(body_id)

        return total_distance

    def run_model_simulation_resting(self, discrete_paras):
        """
        Uses MuJoCo (Multi-Joint dynamics with Contact) Python library to simulate scenario as given in XML setup file.
        https://mujoco.readthedocs.io/

        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.           
        Returns:
            float   
        """
   
        # Set discretisation parameters
        dt = discrete_paras[0]
        solver_reference = discrete_paras[1]
        solver_impedance = discrete_paras[2]

        # Output info on what is being simulated
        print(f"\tSimulating {self.description} with dt = {dt}, solver reference = {solver_reference} and solver impedance = {solver_impedance}")
 
        # Setup model world
        self.setup_model_world(dt, solver_reference, solver_impedance)

        # Total time is used as an upper limit all objects should come to rest well before this
        steps = int(self.total_time / self.model.opt.timestep)

        # Set up if creating a video
        if self.save_animation:
            # Renderer
            renderer = mujoco.Renderer(self.model, width=640, height=480)
            self.frames = []
            frame_interval = int(1.0 / (self.fps * self.model.opt.timestep))
            if frame_interval == 0:
                frame_interval = 1
                print("Warning: frame interval too small, set a smaller time step!")

        # Get all body IDs, only include bodies with joints (movable bodies)
        body_ids = [i for i in range(self.model.nbody) if self.model.body_jntadr[i] != -1]
        stop_steps = 1

        for step in range(steps):
            mujoco.mj_step(self.model, self.data)

            # Save frames for video if creating one
            if self.save_animation and step % frame_interval == 0:
                renderer.update_scene(self.data, camera="angled_view")
                frame = renderer.render()
                self.frames.append(frame)

            # Check if everything has stopped
            stop_sim = True
            for body_id in body_ids:
                stop_sim = stop_sim and all(np.abs(self.data.cvel[body_id]) < self.velocity_thresh)

            # Stop if stationary for a number of steps
            if stop_sim:
                stop_steps += 1
                if stop_steps >= self.steps_required_to_stop:
                    break

        print("\tStop time: ", step*self.model.opt.timestep)
        for body_id in body_ids:
                print(np.abs(self.data.cvel[body_id]))

        # Final position & distance
        total_distance = 0.0
        for body_id in body_ids:          
            pos = self.data.xpos[body_id]  # world position of body
            distance = np.linalg.norm(pos)
            total_distance += distance

        print(f"\tCalculated final value: {total_distance}")
        return total_distance

    def plot_final_model(self):
        """
        Plots the final simulated model which is in this case is a mp4 video if req'd.

        Parameters:  
            None
        Returns:
            None                
        """
        
        if self.save_animation:            
            imageio.mimsave(self.final_mp4_filename, self.frames, fps=self.fps)

    def get_cache_filename(self, discrete_paras : npt.NDArray):
        """
        Returns the model cache filename for the Physics Mug model based on the model parameters.

        Parameters:  
            discrete_paras : npt.NDArray   Discretisation parameters
        Returns:
            str    
        """
      
        # Create filename with all settings and parameters used
        filename = f"mp_{self.total_time}_{self.model_file[:-4]}_"
        
        if not self.use_fixed_time:
            filename += f"{self.velocity_thresh}_"

        if self.use_offset_model:
            filename += "_".join(str(i) for i in self.final_tols) + "_"

        filename += "_".join(str(i) for i in discrete_paras) + ".bin"

        return filename
    

