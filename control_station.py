#!/usr/bin/python
"""!
Main GUI for Arm lab
"""
import os
# from re import X
# from shlex import join
script_path = os.path.dirname(os.path.realpath(__file__))

import argparse
import sys
import cv2
import numpy as np
import rospy
import time
from functools import partial

from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
from PyQt4.QtGui import (QPixmap, QImage, QApplication, QWidget, QLabel,
                         QMainWindow, QCursor, QFileDialog)

from ui import Ui_MainWindow
from rxarm import RXArm, RXArmThread
from camera import Camera, VideoThread
from state_machine import StateMachine, StateMachineThread
from kinematics import IK_geometric, get_IK_joint_angles
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class Gui(QMainWindow):
    """!
    Main GUI Class

    Contains the main function and interfaces between the GUI and functions.
    """
    def __init__(self, parent=None, dh_config_file=None, pox_config_file=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        """ Groups of ui commonents """
        self.joint_readouts = [
            self.ui.rdoutBaseJC,
            self.ui.rdoutShoulderJC,
            self.ui.rdoutElbowJC,
            self.ui.rdoutWristAJC,
            self.ui.rdoutWristRJC,
        ]
        self.joint_slider_rdouts = [
            self.ui.rdoutBase,
            self.ui.rdoutShoulder,
            self.ui.rdoutElbow,
            self.ui.rdoutWristA,
            self.ui.rdoutWristR,
        ]
        self.joint_sliders = [
            self.ui.sldrBase,
            self.ui.sldrShoulder,
            self.ui.sldrElbow,
            self.ui.sldrWristA,
            self.ui.sldrWristR,
        ]
        """Objects Using Other Classes"""
        self.camera = Camera()
        print("Creating rx arm...")
        if (dh_config_file is not None):
            self.rxarm = RXArm(dh_config_file=dh_config_file)
        elif (pox_config_file is not None):
            self.rxarm = RXArm(pox_config_file=pox_config_file)
        else:
            self.rxarm = RXArm()
        print("Done creating rx arm instance.")
        self.sm = StateMachine(self.rxarm, self.camera)
        """
        Attach Functions to Buttons & Sliders
        TODO: NAME AND CONNECT BUTTONS AS NEEDED
        """
        # Video
        self.ui.videoDisplay.setMouseTracking(True)
        self.ui.videoDisplay.mouseMoveEvent = self.trackMouse
        self.ui.videoDisplay.mousePressEvent = self.calibrateMousePress
        self.ui.videoDisplay.mousePressEvent = self.grabBlock

        # Buttons
        # Handy lambda function falsethat can be used with Partial to only set the new state if the rxarm is initialized
        #nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state if self.rxarm.initialized else None)
        nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state)
        self.ui.btn_estop.clicked.connect(self.estop)
        self.ui.btn_init_arm.clicked.connect(self.initRxarm)
        self.ui.btn_torq_off.clicked.connect(
            lambda: self.rxarm.disable_torque())
        self.ui.btn_torq_on.clicked.connect(lambda: self.rxarm.enable_torque())
        self.ui.btn_sleep_arm.clicked.connect(lambda: self.rxarm.sleep())


        #User Buttons
        self.ui.btnUser1.setText("Calibrate")
        self.ui.btnUser1.clicked.connect(partial(nxt_if_arm_init, 'calibrate'))
        self.ui.btnUser2.setText('Open Gripper')
        # self.ui.btnUser2.clicked.connect(lambda: self.rxarm.open_gripper())
        self.ui.btnUser2.clicked.connect(self.setRecordingWaypointGripperStateOpen)
        self.ui.btnUser3.setText('Close Gripper')
        # self.ui.btnUser3.clicked.connect(lambda: self.rxarm.close_gripper())
        self.ui.btnUser3.clicked.connect(self.setRecordingWaypointGripperStateClosed)
        self.ui.btnUser4.setText('Unstack')
        self.ui.btnUser4.clicked.connect(partial(nxt_if_arm_init, 'unstack_stacks'))
        self.ui.btnUser5.setText('Record Waypoint')
        self.ui.btnUser5.clicked.connect(partial(nxt_if_arm_init, 'record_waypoint'))
        self.ui.btnUser6.setText('Save Waypoint')
        self.ui.btnUser6.clicked.connect(partial(nxt_if_arm_init, 'save_waypoint'))
        self.ui.btnUser7.setText('Detect')
        self.ui.btnUser7.clicked.connect(partial(nxt_if_arm_init, 'detect'))
        self.ui.btnUser8.setText('Pick and Sort')
        self.ui.btnUser8.clicked.connect(partial(nxt_if_arm_init, 'pick_and_sort'))
        self.ui.btnUser9.setText('Pick n Stack')
        self.ui.btnUser9.clicked.connect(partial(nxt_if_arm_init, 'pick_n_stack'))
        self.ui.btnUser10.setText('Line em Up')
        self.ui.btnUser10.clicked.connect(partial(nxt_if_arm_init, 'line_em_up'))
        self.ui.btnUser11.setText('Stack em High')
        self.ui.btnUser11.clicked.connect(partial(nxt_if_arm_init, 'stack_em_high'))
        self.ui.btnUser12.setText('To the Sky')
        self.ui.btnUser12.clicked.connect(partial(nxt_if_arm_init, 'to_the_sky'))

        # Sliders
        for sldr in self.joint_sliders:
            sldr.valueChanged.connect(self.sliderChange)
        self.ui.sldrMoveTime.valueChanged.connect(self.sliderChange)
        self.ui.sldrAccelTime.valueChanged.connect(self.sliderChange)
        # Direct Control
        self.ui.chk_directcontrol.stateChanged.connect(self.directControlChk)
        # Status
        self.ui.rdoutStatus.setText("Waiting for input")
        """initalize manual control off"""
        self.ui.SliderFrame.setEnabled(False)
        """Setup Threads"""

        # State machine
        self.StateMachineThread = StateMachineThread(self.sm)
        self.StateMachineThread.updateStatusMessage.connect(
            self.updateStatusMessage)
        self.StateMachineThread.start()
        self.VideoThread = VideoThread(self.camera)
        self.VideoThread.updateFrame.connect(self.setImage)
        self.VideoThread.start()
        self.ArmThread = RXArmThread(self.rxarm)
        self.ArmThread.updateJointReadout.connect(self.updateJointReadout)
        self.ArmThread.updateEndEffectorReadout.connect(
            self.updateEndEffectorReadout)
        self.ArmThread.start()

        # Additional variables added by us
        self.click_num = 0

    """ Slots attach callback functions to signals emitted from threads"""

    @pyqtSlot(str)
    def updateStatusMessage(self, msg):
        self.ui.rdoutStatus.setText(msg)

    @pyqtSlot(list)
    def updateJointReadout(self, joints):
        for rdout, joint in zip(self.joint_readouts, joints):
            rdout.setText(str('%+.2f' % (joint * R2D)))

    ### TODO: output the rest of the orientation according to the convention chosen
    @pyqtSlot(list)
    def updateEndEffectorReadout(self, pos):
        self.ui.rdoutX.setText(str("%+.2f mm" % (1000 * pos[0])))
        self.ui.rdoutY.setText(str("%+.2f mm" % (1000 * pos[1])))
        self.ui.rdoutZ.setText(str("%+.2f mm" % (1000 * pos[2])))
        self.ui.rdoutPhi.setText(str("%+.2f rad" % (pos[3])))
        #self.ui.rdoutTheta.setText(str("%+.2f" % (pos[4])))
        #self.ui.rdoutPsi.setText(str("%+.2f" % (pos[5])))

    @pyqtSlot(QImage, QImage, QImage)
    def setImage(self, rgb_image, depth_image, tag_image):
        """!
        @brief      Display the images from the camera.

        @param      rgb_image    The rgb image
        @param      depth_image  The depth image
        """
        if (self.ui.radioVideo.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(rgb_image))
        if (self.ui.radioDepth.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(depth_image))
        if (self.ui.radioUsr1.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(tag_image))
        if (self.ui.radioUsr2.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(self.camera.convertQtBlockFrame()))            

    """ Other callback functions attached to GUI elements"""

    def estop(self):
        self.rxarm.disable_torque()
        self.sm.set_next_state('estop')

    # def stop_recording(self):
    #     self.sm.set_next_state('initialize_rxarm')

    def sliderChange(self):
        """!
        @brief Slider changed

        Function to change the slider labels when sliders are moved and to command the arm to the given position
        """
        for rdout, sldr in zip(self.joint_slider_rdouts, self.joint_sliders):
            rdout.setText(str(sldr.value()))

        self.ui.rdoutMoveTime.setText(
            str(self.ui.sldrMoveTime.value() / 10.0) + "s")
        self.ui.rdoutAccelTime.setText(
            str(self.ui.sldrAccelTime.value() / 20.0) + "s")
        self.rxarm.set_moving_time(self.ui.sldrMoveTime.value() / 10.0)
        self.rxarm.set_accel_time(self.ui.sldrAccelTime.value() / 20.0)

        # Do nothing if the rxarm is not initialized
        if self.rxarm.initialized:
            joint_positions = np.array(
                [sldr.value() * D2R for sldr in self.joint_sliders])
            # Only send the joints that the rxarm has
            self.rxarm.set_positions(joint_positions[0:self.rxarm.num_joints])

    def directControlChk(self, state):
        """!
        @brief      Changes to direct control mode

                    Will only work if the rxarm is initialized.

        @param      state  State of the checkbox
        """
        if state == Qt.Checked and self.rxarm.initialized:
            # Go to manual and enable sliders
            self.sm.set_next_state("manual")
            self.ui.SliderFrame.setEnabled(True)
        else:
            # Lock sliders and go to idle
            self.sm.set_next_state("idle")
            self.ui.SliderFrame.setEnabled(False)
            self.ui.chk_directcontrol.setChecked(False)

    def trackMouse(self, mouse_event):
        """!
        @brief      Show the mouse position in GUI

                    TODO: after implementing workspace calibration display the world coordinates the mouse points to in the RGB
                    video image.

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        # print(self.rxarm.get_positions())

        pt = mouse_event.pos()
        if self.camera.DepthFrameRaw.any() != 0:
            z = self.camera.DepthFrameRaw[pt.y()][pt.x()]
            x = pt.x()
            y = pt.y()
            # self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
            #                                  (x, y, z))
            pixel = self.camera.VideoFrame[y, x]
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            # h, s, v = self.camera.DepthFrameHSV[y, x, :]
            # print(self.camera.DepthFrameRGB[y, x])

            self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
                                             (x, y, z))
            frame_c = z * np.dot(np.linalg.inv(self.camera.intrinsic_matrix), np.array([x, y, 1]).reshape((3, 1)))
            frame_c = np.vstack((frame_c, 1))
            frame_w = np.dot(np.linalg.inv(self.camera.extrinsic_matrix), frame_c)
            #print(frame_c)
            self.ui.rdoutMouseWorld.setText("(%.0f,%.0f,%.0f)" %
                                             (frame_w[0, 0], frame_w[1, 0], frame_w[2, 0]))

    def calibrateMousePress(self, mouse_event):
        """!
        @brief Record mouse click positions for calibration

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        """ Get mouse posiiton """
        pt = mouse_event.pos()
        self.camera.last_click[0] = pt.x()
        self.camera.last_click[1] = pt.y()
        self.camera.new_click = True
        # print(self.camera.last_click)

    def grabBlock(self, mouse_event):
        self.click_num += 1
        # self.rxarm.open_gripper()
        # self.rxarm.close_gripper()
        print("In Grab Block Event!")
        pt = mouse_event.pos()
        x = pt.x()
        y = pt.y()
        x,y,z1 = self.camera.uvToWorld((x,y))

        prev_pose = self.rxarm.get_positions()
        home_pose = np.array([0, 0, 0, 0, 0])

        print("gripper state = " + str(self.rxarm.gripper_state))
        # gripper state = true means closed
        # if not self.rxarm.gripper_state:
        if self.click_num % 2 == 1:
            min_dist = -1
            min_dist_block = None
            print("Found detections: " + str(self.camera.block_detections))

            for block in self.camera.block_detections.tolist():
                print("Checking block at " + str(block['location']))
                blockx, blocky, blockz = block['location']
                blockphi = block['theta']
                dx = blockx - x
                dy = blocky - y
                dist = dx ** 2 + dy ** 2
                if dist < min_dist or min_dist == -1:
                    print("Setting block as min distance block")
                    min_dist = dist
                    min_dist_block = block
            if min_dist_block:
                print("Found min distance block at " + str(min_dist_block['location']))
                x,y,z = min_dist_block['location']
                phi = min_dist_block['theta']
                pose = (x, y, z + 30, phi, -np.pi/2)
                # joint_angles_all_solutions = IK_geometric(pose)
                joint_angles, _ = get_IK_joint_angles(pose)
                if joint_angles is not None:
                #    try:
                #       self.rxarm.open_gripper()
                #    except:
                #       pass

                    first_pose = joint_angles[0]
                    first_pose[2] = -first_pose[2]
                    first_pose[3] = -first_pose[3]
                    # first_pose[4] = -first_pose[4]

                    # raise arm up
                    # self.rxarm.set_joint_positions((first_pose[0], 0, np.pi/2, 0, 0))

                    self.rxarm.set_joint_positions(first_pose)
                    final_pose = joint_angles[1]
                    final_pose[2] = -final_pose[2]
                    final_pose[3] = -final_pose[3]
                    # final_pose[4] = -final_pose[4]
                    # final_pose[2] = -final_pose[2]
                    # print("final pose: " + str(final_pose))
                    self.rxarm.set_joint_positions(final_pose)
                    # joint_angles = joint_angles_all_solutions[1, :] # Elbow up, counter clockwise wrist turn
                    # joint_angles[2] = -joint_angles[2]
                    # joint_angles[3] = -joint_angles[3]
                    # self.rxarm.set_joint_positions(joint_angles)
                    self.rxarm.close_gripper()
                    # self.rxarm.set_joint_positions(first_pose)
                    # self.rxarm.set_joint_positions((first_pose[0], 0, -np.pi/2, 0, 0))
                    self.rxarm.set_joint_positions(home_pose)
                else:
                    print("No solution for desired pose!")
        
        elif self.click_num % 2 == 0:
            pose = (x, y, 30, 0, -np.pi/2)
            # joint_angles_all_solutions = IK_geometric(pose)
            joint_angles, _ = get_IK_joint_angles(pose)
            if joint_angles is not None:
                first_pose = joint_angles[0]
                first_pose[2] = -first_pose[2]
                first_pose[3] = -first_pose[3]
                self.rxarm.set_joint_positions(first_pose)
                final_pose = joint_angles[1]
                final_pose[2] = -final_pose[2]
                final_pose[3] = -final_pose[3]
                self.rxarm.set_joint_positions(final_pose)
                self.rxarm.open_gripper()
                self.rxarm.set_joint_positions(first_pose)
                # self.rxarm.set_joint_positions((first_pose[0], 0, -np.pi/2, 0, 0))
                self.rxarm.set_joint_positions(home_pose)
            else:
                print("No solution for desired pose!")

    def initRxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.ui.SliderFrame.setEnabled(False)
        self.ui.chk_directcontrol.setChecked(False)
        self.rxarm.enable_torque()
        self.sm.set_next_state('initialize_rxarm')

    def setRecordingWaypointGripperStateClosed(self):
        self.sm.rw_gripper_closed = True

    def setRecordingWaypointGripperStateOpen(self):
        self.sm.rw_gripper_closed = False


### TODO: Add ability to parse POX config file as well
def main(args=None):
    """!
    @brief      Starts the GUI
    """
    app = QApplication(sys.argv)
    app_window = Gui(dh_config_file=args['dhconfig'], pox_config_file=args['poxconfig'])
    app_window.show()
    sys.exit(app.exec_())


# Run main if this file is being run directly
### TODO: Add ability to parse  POX config file as well
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c",
                    "--dhconfig",
                    required=False,
                    help="path to DH parameters csv file")
    ap.add_argument("-p",
                    "--poxconfig",
                    required=False,
                    help="path to POX parameters csv file")  
    main(args=vars(ap.parse_args()))
