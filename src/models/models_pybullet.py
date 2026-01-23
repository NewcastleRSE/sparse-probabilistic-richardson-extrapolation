##############################################################################
# Simulation Models using pybullet, Physics Modelling
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import numpy as np
import numpy.typing as npt
import pybullet
import pybullet_data
import time

# Application modules
from models.base_model import Model

class PhysicsMugModel(Model):
    """
    Class for Physics model of a mug falling on a surface.
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
     
        self.total_time = 5.0

        # To decided when have objects stopped moving
        self.velocity_thresh = 1e-1
        self.steps_required_to_stop = 30

        # Use model f_z(x) = f(z+x)
        self.use_offset_model = False

        # Set default camera parameters
        self.camera_distance = 0.5                # closer to the object (default ~1.5)
        self.camera_yaw = 45                      # rotate horizontally
        self.camera_pitch = -50                   # angle downward
        self.camera_target_position = [0, 0, 0]
        # Scalar factor to delay mp4 at each step
        self.mp4_delay = 1

        # Mug settings
        self.object = "objects/mug.urdf"
        self.mug_linear_velocity = [0, 0, 0]
        self.mug_angular_velocity = [3.0, -1.5, 5.0] 
        self.base_position = [0, 0, 2.0]        # start above ground
        self.base_orientation = [0, 0, 0, 1]
        self.global_scaling = 1.0 
        
        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename)

        # Set initial model name
        self.model_name = "PhysicsMug"
        self.description = "Physics Mug Model"

        # Set default model parameters
        if self.final_mp4_filename is not None and self.final_mp4_filename != "":
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False
       
        # Set true value
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()
       
    def setup_model_world(self, dt : float, substeps : int, solver_iters : int, mp4_mode : bool = False) -> int:
        """
        Sets up world in pybullet to simulate model

        Parameters:  
            dt : float            Time step taken at each iteration in simulation
            substeps : int        Subdivide the physics simulation step further by 'numSubSteps'.
                                  This will trade performance over accuracy.
            solver_iters : int    The maximum number of constraint solver iterations. If the
                                  solverResidualThreshold (default, 1e-7) is reached, the solver may terminate before the numSolverIterations.  
            mp4_mode : bool       Use mode for creating mp4     
        Returns:
            int   
        """

        mode = pybullet.GUI if mp4_mode else pybullet.DIRECT

        # Set up physics simulator
        physicsClient = pybullet.connect(mode)

        # Add path for objects
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Simulation parameters
        pybullet.setPhysicsEngineParameter(
            fixedTimeStep = dt,
            numSubSteps = substeps,
            numSolverIterations = solver_iters
        )

        # Set the ground plane
        plane = pybullet.loadURDF("plane.urdf")

        # Create a simple mug object or some other object
        mug_id = pybullet.loadURDF( 
            self.object,     
            basePosition = self.base_position,        # start above ground
            baseOrientation = self.base_orientation,
            globalScaling = self.global_scaling
        )

        # Set gravity in world
        pybullet.setGravity(0, 0, -9.81)

        # Add initial spin to mug
        pybullet.resetBaseVelocity(
            mug_id,
            angularVelocity = self.mug_angular_velocity,  # spin around x, y, z
            linearVelocity = self.mug_linear_velocity
        )
       
        return mug_id

    def is_body_at_rest(self, body_id : object, do_it) -> bool:
        """
        Sets up world in pybullet to simulate model

        Parameters:  
            body_id : int              The object (mug) ID to check if stationary
            threshold : float          Threshold to check if it is stationary
        Returns:
            object   
        """

        # Get linear and angular velocities
        lin_vel, ang_vel = pybullet.getBaseVelocity(body_id)
        if do_it:
            print(lin_vel, ang_vel)
        return all(np.abs(lin_vel) < self.velocity_thresh) and all(np.abs(ang_vel) < self.velocity_thresh)

    def run_model_simulation(self, discrete_paras):
        """
        Uses pybullet package to simulate a falling spinning mug onto a surface.
        https://pybullet.org/wordpress/

        Parameters:  
            discrete_paras : npt.NDArray     Discretisation parameters used to simulate model.           
        Returns:
            float   
        """
   
        # Set discretisation parameters
        dt = discrete_paras[0]
        substeps = int(np.round(1.0/discrete_paras[1]))
        solver_iters = int(np.round(1.0/discrete_paras[2]))

        mug = self.setup_model_world(dt, substeps, solver_iters)
      
        # Output info on what is being simulated
        print(f"\tSimulating {self.description} with dt = {dt}, {substeps} substeps and {solver_iters} solver iterations")
 
        # Initial time counter and stationary counter
        sim_time = 0.0
        stationary_count = 0

        # Run the simulation
        while sim_time < self.total_time:
            # One step of simulation
            pybullet.stepSimulation()
            sim_time += dt   
            #if sim_time > 550:         
            #    print(sim_time) 
            # Now stop if the mug (or object) is stationary
            if self.is_body_at_rest(mug, False):
                stationary_count += 1
                if stationary_count >= self.steps_required_to_stop:                    
                    break
            else:
                stationary_count = 0

        print("\tEnd time:", sim_time)

        # Get final position and orientation of mug
        pos, orn = pybullet.getBasePositionAndOrientation(mug)

        # Get distance of mug from origin
        dist = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
       
        # End simulation
        pybullet.disconnect()

        # Do video if req'd
        if self.save_animation:
            self.record_mp4()

        print(f"\tCalculated final value: {dist}")
        return dist

    def plot_final_model(self):
        """
        Plots the final simulated model which is in this case is a mp4 video if req'd.

        Parameters:  
            None
        Returns:
            None                
        """
   
        if self.save_animation:
            self.record_mp4()

    def record_mp4(self):
        """
        Create mp4 video of the model of a falling spinning mug onto a surface.
       
        Parameters:  
            None         
        Returns:
            None
        """

        # Set discretisation parameters
        dt = self.final_tols[0]
        substeps = int(np.round(1.0/self.final_tols[1]))
        solver_iters = int(np.round(1.0/self.final_tols[2]))

        # Output info on what is being simulated
        print(f"\tSimulating {self.description} for mp4 with dt = {dt}, {substeps} substeps and {solver_iters} solver iterations")

        self.setup_model_world(dt, substeps, solver_iters, True)

        # Zoomed-in camera settings
        pybullet.resetDebugVisualizerCamera(
            cameraDistance = self.camera_distance,                # closer to the cube (default ~1.5)
            cameraYaw = self.camera_yaw,                      # rotate horizontally
            cameraPitch = self.camera_pitch,                   # angle downward
            cameraTargetPosition = self.camera_target_position   # look at where the cube will fall
        )
 
        # Hide GUI
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        # Start video recording
        log_id = pybullet.startStateLogging(
            pybullet.STATE_LOGGING_VIDEO_MP4,
            self.final_mp4_filename
        )
            
        # Initial time counter
        sim_time = 0.0

        while sim_time < self.total_time:
            # One step of simulation
            pybullet.stepSimulation()

            # Make real-time video look normal - but only if dt ~= 1/240 - needs updating otherwise  
            time.sleep(dt * self.mp4_delay) # fudge factor

            sim_time += dt
  
        # Stop filming mug
        pybullet.stopStateLogging(log_id)
        print(f"Saved video to {self.final_mp4_filename}")

        # End simulation
        pybullet.disconnect()

    
    def get_cache_filename(self, discrete_paras : npt.NDArray):
        """
        Returns the model cache filename for the Physics Mug model based on the model parameters.

        Parameters:  
            discrete_paras : npt.NDArray   Discretisation parameters
        Returns:
            str    
        """
      
        # Create filename with all settings and parameters used
        filename = f"pm_{self.total_time}_"
        if self.use_offset_model:
            filename += "_".join(str(i) for i in self.final_tols) + "_"

        filename += "_".join(str(i) for i in discrete_paras) + ".bin"

        return filename

class PhysicsDuckModel(PhysicsMugModel):
    """
    Class for Physics model of a rubber duck falling a tiny bit and coming to rest on a surface.
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
     
        # Model parameters that can be updated in scenario file
        self.total_time = 5.0
      
        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename, skip_true_value_calc = True)

        # Set default camera parameters
        self.camera_distance = 0.5                # closer to the object (default ~1.5)
        self.camera_yaw = 45                      # rotate horizontally
        self.camera_pitch = -50                   # angle downward
        self.camera_target_position = [0, 0, 0]

        # Mug settings
        #self.object = "duck_vhacd.urdf"
        self.mug_angular_velocity = [3.0, -1.5, 5.0] 
        self.base_position = [0, 0, 1.0]        # start above ground
        self.base_orientation = [0, 0, 0, 1]
        self.global_scaling = 1.0 
       
        # Set initial model name
        self.model_name = "PhysicsDuck"
        self.description = "Physics Rubber Duck Model"

         # Set default model parameters
        if self.final_mp4_filename is not None and self.final_mp4_filename != "":
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False
       
        # Set true value
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()
    
class PhysicsQuickModel(PhysicsMugModel):
    """
    Class for Physics model of some object in a quick scenerio.
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
     
       
        self.total_time = 5.0
      
        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename, skip_true_value_calc = True)

        # Parameters set unchangable for this model below
        # Set default camera parameters
        self.camera_distance = 0.2                # closer to the object (default ~1.5)
        self.camera_yaw = 45                      # rotate horizontally
        self.camera_pitch = -50                   # angle downward
        self.camera_target_position = [0, 0, 0]

        # Object settings
        #self.object = "domino/domino.urdf"
        self.object = "lego/lego.urdf"
        self.mug_angular_velocity = [0.0, 0.0, 0.0] 
        self.base_position = [0, 0, 1.0]        # start above ground
        self.base_orientation = [0.2, -0.1, 0.05, 1]
        self.global_scaling = 1.0 
        
        # Set initial model name
        self.model_name = "PhysicsQuick"
        self.description = "Physics Quick Model"

         # Set default model parameters
        if self.final_mp4_filename is not None and self.final_mp4_filename != "":
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False

        # Set true value
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()

class PhysicsSlickModel(PhysicsMugModel):
    """
    Class for Physics model of some object in a slick scenerio.
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
     
        self.total_time = 2.0
      
        # Call Parent’s constructor to set parameters
        super().__init__(params, parameter_filename, skip_true_value_calc = True)

        # Parameters set unchangable for this model below
        # Set default camera parameters
        self.camera_distance = 2.0                # closer to the object (default ~1.5)
        self.camera_yaw = 45                      # rotate horizontally
        self.camera_pitch = -50                   # angle downward
        self.camera_target_position = [0, 0, 0]

        # Object settings
        #self.object = "domino/domino.urdf"
        self.object = "soccerball.urdf"
        self.mug_linear_velocity = [-1.0, 0.0, 0]
        self.mug_angular_velocity = [-1.0, 0.0, 0.0] 
        self.base_position = [0, 0, 0.55]        # start above ground
        self.base_orientation = [0.2, -0.1, 0.05, 1]
        self.global_scaling = 1.0 
        
        # Set initial model name
        self.model_name = "PhysicsSlick"
        self.description = "Physics Slick Model"

         # Set default model parameters
        if self.final_mp4_filename is not None and self.final_mp4_filename != "":
            self.save_animation = True
            self.do_final_model_plot = True
        else:
            self.save_animation = False

        # Set true value
        if not self.save_animation and not skip_true_value_calc:
            self.set_true_value()   
