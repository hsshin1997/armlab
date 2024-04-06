"""!
Implements the RXArm class.

The RXArm class contains:

* last feedback from joints
* functions to command the joints
* functions to get feedback from joints
* functions to do FK and IK
* A function to read the RXArm config file

You will upgrade some functions and also implement others according to the comments given in the code.
"""
import numpy as np
from functools import partial

from numpy import block
from kinematics import FK_dh, FK_pox, get_pose_from_T, get_IK_joint_angles
import time
import csv
from builtins import super
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
from interbotix_robot_arm import InterbotixRobot
from interbotix_descriptions import interbotix_mr_descriptions as mrd
from config_parse import parse_pox_param_file as parse_pox
from config_parse import parse_dh_param_file as parse_dh
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import rospy
"""
TODO: Implement the missing functions and add anything you need to support them
"""
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def _ensure_initialized(func):
    """!
    @brief      Decorator to skip the function if the RXArm is not initialized.

    @param      func  The function to wrap

    @return     The wraped function
    """
    def func_out(self, *args, **kwargs):
        if self.initialized:
            return func(self, *args, **kwargs)
        else:
            print('WARNING: Trying to use the RXArm before initialized')

    return func_out


class RXArm(InterbotixRobot):
    """!
    @brief      This class describes a RXArm wrapper class for the rx200
    """
    def __init__(self, dh_config_file=None, pox_config_file=None):
        """!
        @brief      Constructs a new instance.

                    Starts the RXArm run thread but does not initialize the Joints. Call RXArm.initialize to initialize the
                    Joints.

        @param      dh_config_file  The configuration file that defines the DH parameters for the robot
        """
        super().__init__(robot_name="rx200", mrd=mrd)
        self.joint_names = self.resp.joint_names
        print(self.joint_names)
        self.num_joints = 5
        # Gripper
        self.gripper_state = True # true means closed
        # State
        self.initialized = False
        # Cmd
        self.position_cmd = None
        self.moving_time = 2.0
        self.accel_time = 0.5
        # Feedback
        self.position_fb = None
        self.velocity_fb = None
        self.effort_fb = None
        # DH Params
        self.dh_params = []
        self.pox_params = []
        self.dh_config_file = dh_config_file
        if (dh_config_file is not None):
            self.dh_params = RXArm.parse_dh_param_file(dh_config_file)
        elif (pox_config_file is not None):
            print("Loading POX config params...")
            self.pox_params = parse_pox(pox_config_file)
        #POX params
        self.pox_config_file = pox_config_file
        self.M_matrix = []
        self.S_list = []
        if self.pox_params:
            self.M_matrix, self.S_list = self.pox_params
            print("M_Matrix = " + str(self.M_matrix))
            print("S_List = " + str(self.S_list))
        # self.pox_config_file = pox_config_file
        self.grabbed_block = {}
        self.home_pose = np.array([0, 0, 0, 0, 0])
        self.last_grabbed_angle = None

    def initialize(self):
        """!
        @brief      Initializes the RXArm from given configuration file.

                    Initializes the Joints and serial port

        @return     True is succes False otherwise
        """
        self.initialized = False
        # Wait for other threads to finish with the RXArm instead of locking every single call
        rospy.sleep(0.25)
        """ Commanded Values """
        self.position = [0.0] * self.num_joints  # radians
        """ Feedback Values """
        self.position_fb = [0.0] * self.num_joints  # radians
        self.velocity_fb = [0.0] * self.num_joints  # 0 to 1 ???
        self.effort_fb = [0.0] * self.num_joints  # -1 to 1

        # Reset estop and initialized
        self.estop = False
        self.enable_torque()
        self.moving_time = 2.0
        self.accel_time = 0.5
        self.set_gripper_pressure(1.0)
        self.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=False)
        self.open_gripper()
        self.initialized = True
        return self.initialized

    def sleep(self):
        self.moving_time = 2.0
        self.accel_time = 1.0
        self.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=True)
        self.go_to_sleep_pose(moving_time=self.moving_time,
                              accel_time=self.accel_time,
                              blocking=False)
        self.initialized = False

    def set_positions(self, joint_positions):
        """!
         @brief      Sets the positions.

         @param      joint_angles  The joint angles
         """
        self.set_joint_positions(joint_positions,
                                 moving_time=self.moving_time,
                                 accel_time=self.accel_time,
                                 blocking=False)

    def set_moving_time(self, moving_time):
        self.moving_time = moving_time

    def set_accel_time(self, accel_time):
        self.accel_time = accel_time

    def disable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.torque_joints_off(self.joint_names)

    def enable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.torque_joints_on(self.joint_names)

    def get_positions(self):
        """!
        @brief      Gets the positions.

        @return     The positions.
        """
        return self.position_fb

    def get_velocities(self):
        """!
        @brief      Gets the velocities.

        @return     The velocities.
        """
        return self.velocity_fb

    def get_efforts(self):
        """!
        @brief      Gets the loads.

        @return     The loads.
        """
        return self.effort_fb
    def get_gripper_state(self):
        return self.gripper_state

#   @_ensure_initialized

    def get_ee_pose(self):
        """!
        @brief      TODO Get the EE pose.

        @return     The EE pose as [x, y, z, phi] or as needed.
        """
        joint_angle = -self.get_positions()
        # joint_angle[1] = -joint_angle[1]
        joint_angle[0] = -joint_angle[0]
        joint_angle[2] = -joint_angle[2]
        joint_angle[3] = -joint_angle[3]
        # np.array([0,0,0,0,0]
        g_st = FK_pox(joint_angle, self.M_matrix, self.S_list)
        ee_pose = get_pose_from_T(g_st)
        return ee_pose.tolist()

    @_ensure_initialized
    def get_wrist_pose(self):
        """!
        @brief      TODO Get the wrist pose.

        @return     The wrist pose as [x, y, z, phi] or as needed.
        """
        return [0, 0, 0, 0]

    def parse_pox_param_file(self):
        """!
        @brief      TODO Parse a PoX config file

        @return     0 if file was parsed, -1 otherwise 
        """
        print("Parsing POX config file...")
        pox_params = parse_pox(self.dh_config_file)
        print("POX config file parse exit.")
        return -1

    def parse_dh_param_file(self):
        print("Parsing DH config file...")
        dh_params = parse_dh(self.dh_config_file)
        print("DH config file parse exit.")
        return dh_params

    def get_dh_parameters(self):
        """!
        @brief      Gets the dh parameters.

        @return     The dh parameters.
        """
        return self.dh_params

    def grabBlock(self, block_info):
        x,y,z = block_info['location']
        theta = block_info['theta']
        is_large = block_info['is_large']
        pose = (x,y,z-20,theta,-np.pi/2)
        print("Grabbing " + "large" if is_large else "small" + " Block @ (" + str(pose[0]) + "," + str(pose[1]) + "," + str(pose[2]) + "," + str(pose[3]) + ")")

        joint_angles, psi = get_IK_joint_angles(pose)
        if joint_angles is not None:
            first_pose = joint_angles[0]
            first_pose[2] = -first_pose[2]
            first_pose[3] = -first_pose[3]
            # first_pose[4] = -first_pose[4]

            # raise arm up
            # self.rxarm.set_joint_positions((first_pose[0], 0, np.pi/2, 0, 0))

            print("moving to first pose")
            self.set_joint_positions(first_pose)
            final_pose = joint_angles[1]
            final_pose[2] = -final_pose[2]
            final_pose[3] = -final_pose[3]

            print("Moving to second pose")
            self.set_joint_positions(final_pose)
            self.close_gripper()
            print("moving to home pose")
            self.set_joint_positions(first_pose)
            # self.set_joint_positions(self.home_pose)
            self.grabbed_block = block_info
            self.last_grabbed_angle = psi
            print("Grabbing Complete!")
        else:
            print("No solution for desired pose!")

    def dropBlock(self, world_coord, gap=5, drop_angle=None, add_z=15):
        print("dropping off block at " + str(world_coord))
        is_large = self.grabbed_block['is_large']
        x,y,z = world_coord
        if drop_angle is None:
            if self.last_grabbed_angle is not None and np.abs(self.last_grabbed_angle - (-np.pi/4)) < 0.0001: # Should always be the case
                drop_angle = self.last_grabbed_angle
            else:
                add_z = 20 if is_large else 5 # Block is always grabbed at 20cm down from top
                drop_angle = -np.pi/2
        pose = (x, y, z+add_z+gap, 0, drop_angle)
        # joint_angles_all_solutions = IK_geometric(pose)
        joint_angles, _ = get_IK_joint_angles(pose)
        if joint_angles is not None:
            first_pose = joint_angles[0]
            first_pose[2] = -first_pose[2]
            first_pose[3] = -first_pose[3]
            # self.set_joint_positions(first_pose)
            self.set_joint_positions(first_pose)
            final_pose = joint_angles[1]
            final_pose[2] = -final_pose[2]
            final_pose[3] = -final_pose[3]
            self.set_joint_positions(final_pose)
            self.open_gripper()
            self.set_joint_positions(first_pose)
            # self.set_joint_positions(first_pose)
            # self.set_joint_positions((first_pose[0], 0, -np.pi/2, 0, 0))
            self.set_joint_positions(self.home_pose)
        else:
            print("No solution for desired pose!")



class RXArmThread(QThread):
    """!
    @brief      This class describes a RXArm thread.
    """
    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, rxarm, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      RXArm  The RXArm
        @param      parent  The parent
        @details    TODO: set any additional initial parameters (like PID gains) here
        """
        QThread.__init__(self, parent=parent)
        self.rxarm = rxarm
        joint1_pid = [640, 0, 3600] # default [640, 0, 3600]
        joint2_pid = [800, 0, 0] # default [ 800, 0, 0] # updated [4800, 0, 3600]
        joint3_pid = [800, 0, 0] # default [ 800, 0, 0] # updated [800, 200, 3600]
        joint4_pid = [800, 0, 0] # default [ 800, 0, 0]
        joint5_pid = [640, 0, 3600] # default [ 800, 0, 0]
        joint6_pid = [640, 0, 3600] # default [ 800, 0, 0]
        self.rxarm.set_joint_position_pid_params('waist', joint1_pid)
        # No I term in the shoulder motor
        self.rxarm.set_joint_position_pid_params('shoulder', joint2_pid)
        self.rxarm.set_joint_position_pid_params('elbow', joint3_pid)
        self.rxarm.set_joint_position_pid_params('wrist_angle', joint4_pid)
        self.rxarm.set_joint_position_pid_params('wrist_rotate', joint5_pid)
        self.rxarm.set_joint_position_pid_params('gripper', joint6_pid)
        rospy.Subscriber('/rx200/joint_states', JointState, self.callback)
        rospy.Subscriber('/rx200/gripper/command', Float64, self.gripper_state_callback)
        rospy.sleep(0.5)

    def callback(self, data):
        self.rxarm.position_fb = np.asarray(data.position)[0:5]
        self.rxarm.velocity_fb = np.asarray(data.velocity)[0:5]
        self.rxarm.effort_fb = np.asarray(data.effort)[0:5]
        self.updateJointReadout.emit(self.rxarm.position_fb.tolist())
        self.updateEndEffectorReadout.emit(self.rxarm.get_ee_pose())
        #for name in self.rxarm.joint_names:
        #    print("{0} gains: {1}".format(name, self.rxarm.get_motor_pid_params(name)))
        if (__name__ == '__main__'):
            print(self.rxarm.position_fb)
    
    def gripper_state_callback(self, angle):
        print("Grippper State Callback = " + str(angle.data))
        # if angle.data < 0.0:
        #     self.rxarm.gripper_state = True
        # elif angle.data > 0.0:
        #     self.rxarm.gripper_state = False
        if np.abs(angle.data) <= 5.0:
            self.rxarm.gripper_state = False # Gripper Open (0.0)
        else:
            self.rxarm.gripper_state = True # Gripper Closed (250.0 or -250.0)

    def run(self):
        """!
        @brief      Updates the RXArm Joints at a set rate if the RXArm is initialized.
        """
        while True:

            rospy.spin()


if __name__ == '__main__':
    rxarm = RXArm()
    print(rxarm.joint_names)
    armThread = RXArmThread(rxarm)
    armThread.start()
    try:
        joint_positions = [-1.0, 0.5, 0.5, 0, 1.57]
        rxarm.initialize()

        rxarm.go_to_home_pose()
        rxarm.set_gripper_pressure(0.5)
        rxarm.set_joint_positions(joint_positions,
                                  moving_time=2.0,
                                  accel_time=0.5,
                                  blocking=True)
        rxarm.close_gripper()
        rxarm.go_to_home_pose()
        rxarm.open_gripper()
        rxarm.sleep()

    except KeyboardInterrupt:
        print("Shutting down")
