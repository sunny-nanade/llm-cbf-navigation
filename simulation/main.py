import pybullet as p
import pybullet_data
import time
import numpy as np

class Simulation:
    def __init__(self, headless=False):
        """
        Initializes the PyBullet simulation environment.
        """
        self.headless = headless
        if self.headless:
            self.client = p.connect(p.DIRECT)
        else:
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load ground plane and robot
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = self.load_robot()
        
        # Simulation parameters
        self.timestep = 1.0 / 240.0
        p.setTimeStep(self.timestep)

    def load_robot(self):
        """
        Loads the robot model into the simulation.
        Returns the ID of the robot.
        """
        start_pos = [0, 0, 0.5]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        # Using a simple sphere as a placeholder for the robot
        robot_urdf = self.create_simple_urdf("robot_sphere.urdf")
        robot_id = p.loadURDF(robot_urdf, start_pos, start_orientation)
        return robot_id

    def create_simple_urdf(self, file_name):
        """Creates a simple URDF file for a sphere."""
        with open(file_name, "w") as f:
            f.write("""
<robot name="simple_sphere">
  <link name="base_link">
    <visual>
      <geometry>
        <sphere radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.04"/>
    </inertial>
  </link>
</robot>
            """)
        return file_name

    def run(self, duration=10):
        """
        Runs the simulation for a given duration.
        """
        print("Starting simulation...")
        for _ in range(int(duration / self.timestep)):
            # --- Control logic will go here ---
            
            p.stepSimulation()
            if not self.headless:
                time.sleep(self.timestep)
        print("Simulation finished.")

    def get_robot_state(self):
        """
        Returns the current state (position, orientation, velocity) of the robot.
        """
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = p.getBaseVelocity(self.robot_id)
        return np.array(pos), np.array(p.getEulerFromQuaternion(ori)), np.array(vel), np.array(ang_vel)

    def close(self):
        """
        Disconnects from the PyBullet server.
        """
        p.disconnect()

if __name__ == "__main__":
    sim = Simulation()
    try:
        sim.run(duration=10)
    finally:
        sim.close()
