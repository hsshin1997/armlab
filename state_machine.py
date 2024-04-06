"""!
The state machine that implements the logic.
"""
from cgitb import small
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
from camera import Camera
import rospy
import cv2
import csv
from tf.transformations import quaternion_matrix, rotation_matrix

# ==============================================================================
# ==============================================================================
# Added from Dr. Gaskell's gitlab 

def quaternion_rotation_matrix(Q):
    # Extract the values from Q
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    # rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    # R = np.array([[0, -1, 0],[-1, 0, 0], [0, 0, -1]])
    # rot_matrix = np.dot(R, rot_matrix)
    # rot_matrix = np.dot(rot_matrix, R.T)
    rot_matrix = np.dot(np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]), np.array([[0, 1, 0],[-1, 0, 0], [0, 0, 1]]))
    # print(rot_matrix)

    return rot_matrix

def recover_homogenous_affine_transformation(p, p_prime):
    '''points_transformed_1 = points_transformed_1 = np.dot(
    A1, np.transpose(np.column_stack((points_camera, (1, 1, 1, 1)))))np.dot(
    A1, np.transpose(np.column_stack((points_camera, (1, 1, 1, 1)))))
    Find the unique homogeneous affine transformation that
    maps a set of 3 points to another set of 3 points in 3D
    space:

        p_prime == np.dot(p, R) + t

    where `R` is an unknown rotation matrix, `t` is an unknown
    translation vector, and `p` and `p_prime` are the original
    and transformed set of points stored as row vectors:

        p       = np.array((p1,       p2,       p3))
        p_prime = np.array((p1_prime, p2_prime, p3_prime))

    The result of this function is an augmented 4-by-4
    matrix `A` that represents this affine transformation:

        np.column_stack((p_prime, (1, 1, 1))) == \
            np.dot(np.column_stack((p, (1, 1, 1))), A)

    Source: https://math.stackexchange.com/a/222170 (robjohn)
    '''

    # construct intermediate matrix
    Q = p[1:] - p[0]
    Q_prime = p_prime[1:] - p_prime[0]

    # calculate rotation matrix
    R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
               np.row_stack((Q_prime, np.cross(*Q_prime))))

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)

    # calculate affine transformation matrix
    return np.transpose(np.column_stack((np.row_stack((R, t)), (0, 0, 0, 1))))

def recover_homogeneous_transform_svd(m, d):
    ''' 
    finds the rigid body transform that maps m to d: 
    d == np.dot(m,R) + T
    http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
    '''
    # calculate the centroid for each set of points
    d_bar = np.sum(d, axis=0) / np.shape(d)[0]
    m_bar = np.sum(m, axis=0) / np.shape(m)[0]

    # we are using row vectors, so tanspose the first one
    # H should be 3x3, if it is not, we've done this wrong
    H = np.dot(np.transpose(d - d_bar), m - m_bar)
    [U, S, V] = np.linalg.svd(H)

    R = np.matmul(V, np.transpose(U))
    # if det(R) is -1, we've made a reflection, not a rotation
    # fix it by negating the 3rd column of V
    if np.linalg.det(R) < 0:
        V = [1, 1, -1] * V
        R = np.matmul(V, np.transpose(U))
    T = d_bar - np.dot(m_bar, R)
    return np.transpose(np.column_stack((np.row_stack((R, T)), (0, 0, 0, 1))))
# ==============================================================================
# ==============================================================================



class Waypoint(object):
    def __init__(self, joint_pos, gripper_state):
        self.joint_pos = joint_pos
        self.gripper_state = gripper_state
    def get_joint_pos(self):
        return self.joint_pos
    def get_gripper_state(self):
        return self.gripper_state


class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [0.006135923322290182,0.0644271969795227,-0.05062136799097061,-0.010737866163253784,-0.0076699042692780495,False],
            [0.030679617077112198,0.03681553900241852,-0.7286409139633179,0.5414952039718628,-0.006135923322290182,False],
            [0.4832039475440979,0.0475534051656723,-1.2348545789718628,0.7240389585494995,-0.006135923322290182,False],
            [0.4832039475440979,0.5599030256271362,-1.3683109283447266,1.2992817163467407,0.003067961661145091,False],
            [0.47860202193260193,0.5445631742477417,-1.3683109283447266,1.2992817163467407,0.003067961661145091,True],
            [0.47400006651878357,-0.11658254265785217,-1.1351457834243774,0.7102331519126892,0.004601942375302315,True],
            [-1.2394565343856812,-0.10431069880723953,-1.1750292778015137,0.7117670774459839,0.006135923322290182,True],
            [-1.2517284154891968,0.7010292410850525,-1.2624661922454834,1.3867186307907104,0.010737866163253784,True],
            [-1.2486604452133179,0.7040972113609314,-1.2624661922454834,1.3943885564804077,0.010737866163253784,False],
            [-1.193437099456787,-0.013805827125906944,-1.211844801902771,0.8145438432693481,0.010737866163253784,False],
            [-0.3328738510608673,0.1089126393198967,-1.2547962665557861,0.977145791053772,0.00920388475060463,False],
            [-0.34514567255973816,0.6856894493103027,-1.352971076965332,1.4496119022369385,0.010737866163253784,False],
            [-0.3466796576976776,0.6872234344482422,-1.3591070175170898,1.477223515510559,0.010737866163253784,True],
            [-0.31446605920791626,-0.11658254265785217,-1.2532622814178467,0.849825382232666,0.00920388475060463,True],
            [0.44025251269340515,-0.08897088468074799,-1.2317866086959839,0.849825382232666,0.00920388475060463,True],
            [0.4832039475440979,0.6795535087585449,-1.366776943206787,1.4680196046829224,0.00920388475060463,True],
            [0.4939418137073517,0.6918253302574158,-1.3652429580688477,1.4879614114761353,0.00920388475060463,False],
            [0.4693981409072876,-0.11658254265785217,-1.2517284154891968,0.8620972037315369,0.00920388475060463,False],
            [-1.2333205938339233,0.013805827125906944,-1.1612235307693481,0.8666991591453552,0.00920388475060463,False],
            [-1.2471264600753784,0.644271969795227,-1.2624661922454834,1.2808740139007568,0.010737866163253784,False],
            [-1.24252450466156,0.644271969795227,-1.2701361179351807,1.2870099544525146,0.010737866163253784,True],
            [-1.2087769508361816,-0.023009711876511574,-1.179631233215332,0.8436894416809082,0.00920388475060463,True],
            [-0.34514567255973816,0.05368933081626892,-1.2164467573165894,0.9587380290031433,0.00920388475060463,True],
            [-0.3374757766723633,0.6273981332778931,-1.3744468688964844,1.4281361103057861,0.00920388475060463,True],
            [-0.3374757766723633,0.6258642077445984,-1.3744468688964844,1.431204080581665,0.00920388475060463,False],
            [-0.3344078063964844,-0.23930101096630096,-1.3115535974502563,0.7976700067520142,0.00920388475060463,False],
            [0.08590292930603027,0.05675728991627693,-0.3942330777645111,0.2761165499687195,0.00920388475060463,False]

        ]
        
        # [
        #     [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
        #     [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
        #     [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
        #     [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
        #     [0.0,             0.0,      0.0,         0.0,     0.0],
        #     [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
        #     [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
        #     [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
        #     [np.pi/2,         0.5,     0.3,      0.0,     0.0],
        #     [0.0,             0.0,     0.0,      0.0,     0.0]]
        self.recorded_waypoints = []
        self.rw_gripper_closed = False

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "record_waypoint":
            self.record_waypoint()

        if self.next_state == "save_waypoint":
            self.save_waypoint()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == 'pick_and_sort':
            self.pick_and_sort()

        if self.next_state == 'pick_n_stack':
            self.pick_n_stack()

        if self.next_state == 'line_em_up':
            self.line_em_up()

        if self.next_state == 'stack_em_high':
            self.stack_em_high()

        if self.next_state == 'to_the_sky':
            self.to_the_sky()

        if self.next_state == 'unstack_stacks':
            self.unstack_stacks()


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"
        # print(self.camera.tag_detections_recent)

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.current_state = "execute"
        for wp in self.waypoints:
            print("in wp for loop")
            if self.next_state != "estop" or self.current_state == "estop":
                self.rxarm.set_positions(wp[0:4])
                if wp[-1]:
                    self.rxarm.close_gripper()
                else:
                    self.rxarm.open_gripper()
                rospy.sleep(3)
                self.next_state = "execute"
            else:
                self.next_state = "estop"
        self.next_state = "idle"

    def record_waypoint(self):
        self.status_message = "State: Record Waypoints"
        # self.recorded_waypoints.append(Waypoint(self.rxarm.get_positions().tolist(), self.rxarm.get_gripper_state()))
        self.recorded_waypoints.append(Waypoint(self.rxarm.get_positions().tolist(), self.rw_gripper_closed))
        self.next_state = "idle"
        # if self.current_state != "record_waypoint":
        #     print("Disabling torque")
        #     self.current_state = "record_waypoint"
        #     self.rxarm.disable_torque()
        #     print("Disabled torque")

    def save_waypoint(self):
        self.status_message = "State: Stopped Recording Waypoints and Exporting to a File"
        self.current_state = "save_waypoint"
        f = open('waypoints.csv', 'w')
        writer = csv.writer(f)
        for waypoint in self.recorded_waypoints:
            joint_positions = waypoint.get_joint_pos()
            gripper_state = waypoint.get_gripper_state()
            row = [joint_positions[0], joint_positions[1], joint_positions[2], joint_positions[3], joint_positions[4], gripper_state] 
            writer.writerow(row)
        f.close()
        print("Finished exporting to CSV file")
        self.next_state = "idle"


    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"
        print(self.rxarm.get_positions())
        print("End effector pose = " + str(self.rxarm.get_ee_pose()))

        """TODO Perform camera calibration routine here"""
        # K = np.array([[943.7872828, 0.0, 680.7729052],
        #               [0.0, 940.9715492, 378.9719702],
        #               [0.0, 0.0, 1.0]])

        K  = np.array([[746.3671875, 0.0, 568.2421875], [ 0.0, 748.6484375, 397.35546875], [0.0, 0.0, 1.0]])
        # D = np.array([0.107892, -0.1508564, 0.0056768, 0.0029476, 0.0])
        D = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        K_inv = np.linalg.inv(K)

        # apriltag_bundle_pos = np.array([0.059475965632, 0.189883619815, 0.959018052586])
        # apriltag_bundle_q = np.array([-0.703728498574, 0.71009270338, 0.0101352315685, -0.0207805192877])
        apriltag_bundle_pos, apriltag_bundle_q = self.camera.get_april_matrix()
        print(apriltag_bundle_pos)
        print(apriltag_bundle_q)

        # points_uvd = np.array([[484, 348, 976], [1015, 201, 956], [964, 588, 979],
        #                        [390, 588, 991], [531, 349, 975], [817, 443, 974],
        #                        [294, 444, 985], [626, 156, 962], [770, 300, 967],
        #                        [1063, 250, 957], [915, 588, 980], [485, 586, 990]])
        # print(points_uvd.shape)
        # points_uvd = np.zeros(4,3)
        # tag_ids = []
        points_uvd = (np.dot(K, (self.camera.tag_detections_recent).T) * 1000).T / 970
        # print("Camera points: ")
        # print(points_uvd)
        # for detection in data.detections:
        #     if detection.id[0] not in tag_ids:
        #         points_uvd[detection.id[0]-1, :] = detection.pose.pose.pose.position
        #         tag_ids += [detection.id[0]]
        #     if len(tag_ids) == 4:
        #         break

        points_uv = np.delete(points_uvd, -1, axis=1)
        depth_camera = np.transpose(np.delete(points_uvd, (0, 1), axis=1))
        # points_world = np.array([[-200, 175, 0], [350, 325, 0], [300, -75, 0],
        #                          [-300, -75, 0], [-150, 175, 0], [150, 75, 0],
        #                          [-400, 75, 0], [-50, 375, 0], [100, 225, 0],
        #                          [400, 275, 0], [-250, -75, 0], [-200, -75, 0]])
        points_world = np.array([[250, 275, 0], [-250, -25, 0], [-250, 275, 0], [250, -25, 0]])
        points_ones = np.ones(depth_camera.size)
        points_camera = np.transpose(depth_camera * np.dot(K_inv, np.transpose(np.column_stack((points_uv, points_ones)))))


        # # Method 1: use solvePnP
        # [_, R_exp_1, t_1] = cv2.solvePnP(points_world.astype(np.float32), points_uv.astype(np.float32), K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        # R_1, _ = cv2.Rodrigues(R_exp_1)
        # extrinsic_matrix_1 = np.row_stack((np.column_stack((R_1, t_1)), (0, 0, 0, 1)))
        # print("Extrinsix matrix using solvePnP: ")
        # print(extrinsic_matrix_1)
        # print("World point calculated using extrinsix matrix from solvePnP: ")
        # points_transformed_pnp_1 = np.dot(np.linalg.inv(extrinsic_matrix_1), np.transpose(np.column_stack((points_camera, points_ones))))
        # print(points_transformed_pnp_1)

        # # Method 2: use solvePnPRansac
        # [_, R_exp_2, t_2, _] = cv2.solvePnPRansac(points_world.astype(np.float32), points_uv.astype(np.float32), K, D)
        # #t = t + np.array([0, 25, 0])
        # R_2, _ = cv2.Rodrigues(R_exp_2)
        # extrinsic_matrix_2 = np.row_stack((np.column_stack((R_2, t_2)), (0, 0, 0, 1)))
        # print("Extrinsix matrix using solvePnPRansac: ")
        # print(extrinsic_matrix_2)
        # print("World point calculated using extrinsix matrix from solvePnPRansac: ")
        # points_transformed_pnp_2 = np.dot(np.linalg.inv(extrinsic_matrix_2), np.transpose(np.column_stack((points_camera, points_ones))))
        # print(points_transformed_pnp_2)

        # # Method 3: naive extrinsic matrix
        # print("Extrinsix matrix using naive method: \n")
        # print(self.camera.extrinsic_matrix)
        # points_transformed_ideal = np.dot(np.linalg.inv(self.camera.extrinsic_matrix), np.transpose(np.column_stack((points_camera, points_ones))))
        # print("World point calculated using extrinsix matrix from naive: \n")
        # print(points_transformed_ideal)

        # Method 4: 3Daffine 
        # _, T_affine_cv, _ = cv2.estimateAffine3D(points_camera.astype(np.float32), points_world.astype(np.float32), confidence=0.99)
        # A_affine_cv = np.row_stack((T_affine_cv, (0.0, 0.0, 0.0, 1.0)))
        # points_transformed_affine_cv = np.dot(np.linalg.inv(A_affine_cv), np.transpose(np.column_stack((points_camera, points_ones))))
        # print("Extrinsix matrix using 3Daffine: \n")
        # print(A_affine_cv)
        # print("World point calculated using extrinsix matrix from 3Daffine: \n")
        # print(points_transformed_affine_cv)

        # # Method 5: Affine
        # A_affine = recover_homogenous_affine_transformation(points_world[0:8:3], points_camera[0:8:3])
        # points_transformed_affine = np.dot(np.linalg.inv(A_affine), np.transpose(np.column_stack((points_camera, points_ones))))
        # print("Extrinsix matrix using affine: \n")
        # print(A_affine)
        # print("World point calculated using extrinsix matrix from affine: \n")
        # print(points_transformed_affine)
        # self.camera.extrinsic_matrix = A_affine

        # # Method 6: SVD
        # A_svd = recover_homogeneous_transform_svd(points_world, points_camera)
        # points_transformed_svd = np.dot(np.linalg.inv(A_svd), np.transpose(np.column_stack((points_camera, points_ones))))
        # print("Extrinsix matrix using SVD: \n")
        # print(A_svd)
        # print("World point calculated using extrinsix matrix from SVD: \n")
        # print(points_transformed_svd)

        # Method 7: Apriltag bundle
        R_at = quaternion_rotation_matrix(apriltag_bundle_q)
        T_at = 1000.0 * apriltag_bundle_pos
        A_at = np.transpose(np.column_stack((np.row_stack((R_at, T_at)), (0, 0, 0, 1))))
        points_transformed_at = np.dot(np.linalg.inv(A_at), np.transpose(np.column_stack((points_camera, points_ones))))
        print(A_at)
        print(points_transformed_at)

        self.status_message = "Calibration - Completed Calibration"
        # self.camera.extrinsic_matrix = np.linalg.inv(A_at)
        # R_at = quaternion_matrix(np.array([-0.704, 0.710, 0.010, 0.021]))
        # T_at = np.array([0.175, 0.016, 0.963]).reshape(3) * 1000.0
        # A_at = np.transpose(np.column_stack((np.row_stack((R_at, T_at)), (0, 0, 0, 1))))
        self.camera.extrinsic_matrix = A_at
        self.camera.intrinsic_matrix = K
        # self.camera.setExtrinsicMatrix()
        print(self.status_message)
        print(self.camera.extrinsic_matrix)

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        print("In Detect State")

        # detected_block = self.camera.detectBlocksInDepthImage()
        # print("Detected Block? " + str(detected_block))
        # if detected_block:
        #     print("Changing to Idle State")
        #     self.next_state = "idle"

        print("Calibration Points = " + str(self.camera.getBoardDepthCalibrationPoints()))
        self.camera.detectBlocksInDepthImage()
        if self.camera.block_detections.size > 0:
            print("found block")
            for block_info in self.camera.block_detections:
                print(block_info['color'])
                print(block_info['location'])
                print(block_info['theta'])
                print(block_info['is_large'])
            self.next_state = "idle"
            

        # rospy.sleep(1)
        # self.next_state = "idle"

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        # try:
        #     self.rxarm.enable_torque()
        # except Exception as e:
        #     print(e)
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.rxarm.close_gripper()
        self.rxarm.open_gripper()
        self.next_state = "idle"

    

    def pick_and_sort(self): # comp 1
        '''Small Blocks to left of arm, Large blocks to right of arm
        Initial blocks in front of arm in positive half plane (right of arm)
        Level 3 is 9 blocks, random sizes, random colors, possibly stacked
        180s'''

        self.current_state = 'pick_and_sort'

        print("In Pick and Sort State")
        if self.next_state != "estop" or self.current_state == "estop":
            print("Getting large and Small Blocks")
            large_blocks, small_blocks = self.camera.largeAndSmall()

            print("Sorting large blocks")
            # Sort Large Blocks in order of distance from home pose
            large_blocks_distance_sorted = self.camera.sortByDistance(large_blocks)

            print("Sorting Small Blocks")
            # Sort Small Blocks the same way
            small_blocks_distance_sorted = self.camera.sortByDistance(small_blocks)

            print("Placing Large Blocks")
            # Place large blocks right of the arm in the negative? half plane
            large_block_placements = [(450, -25, 0), (375, -125, 0), (375, -25, 0), (300, -125, 0), (300, -25, 0), (225, -125, 0), (225, -25, 0), (150, -125, 0), (150, -25, 0)]
            if large_blocks.size > 0:
                for i, large_block in enumerate(large_blocks_distance_sorted):
                    drop_location = large_block_placements[i]
                    self.rxarm.grabBlock(large_block)
                    self.rxarm.dropBlock(drop_location)

            print("Placing Small Blocks")
            # Place small blocks left of the arm in the negative half plane
            small_block_placements = [(-450, -25, 0), (-375, -125, 0), (-375, -25, 0), (-300, -125, 0), (-300, -25, 0), (-225, -125, 0), (-225, -25, 0), (-150, -125, 0), (-150, -25, 0)]
            if small_blocks.size > 0:
                for i, small_block in enumerate(small_blocks_distance_sorted):
                    drop_location = small_block_placements[i]
                    self.rxarm.grabBlock(small_block)
                    self.rxarm.dropBlock(drop_location)
            
            print("Exiting State. Returning to Idle")
            self.next_state = 'idle'

        else:
            print("ESTOPPING!!")
            self.next_state = 'estop'


    def pick_n_stack(self): # comp 2
        '''Blocks placed in front of arm in positive half plane
        Stack all blocks 3 tall to the left or right of the arm in the negative half plane
        Level 3 is 9 blocks, random sizes, random colors, possible stacked 2 high
        360s'''
        self.current_state = 'pick_n_stack'

        print("In Pick n' Stack State")
        if self.next_state != "estop" or self.current_state == "estop":
            print("Getting large and Small Blocks")
            large_blocks, small_blocks = self.camera.largeAndSmall()

            print("Sorting large blocks")
            # Sort Large Blocks in order of distance from home pose
            large_blocks_distance_sorted = self.camera.sortByDistance(large_blocks)

            print("Sorting Small Blocks")
            # Sort Small Blocks the same way
            small_blocks_distance_sorted = self.camera.sortByDistance(small_blocks)

            block_list = np.array(large_blocks_distance_sorted.tolist() + small_blocks_distance_sorted.tolist())

            stack_placements = [(350, -25), (250, -75), (150, -125)]
            stack_last_z = [0, 0, 0]

            print("Stacking Blocks")
            for i, block in enumerate(block_list):
                # Get current stack
                stack_idx = i % 3
                # Get Z and update
                add_z = 40 if block['is_large'] else 26
                z = stack_last_z[stack_idx]
                stack_last_z[stack_idx] = z + add_z
                x,y = stack_placements[stack_idx]
                # Pick up and drop off
                drop_location = (x,y,z)
                self.rxarm.grabBlock(block)
                self.rxarm.dropBlock(drop_location, gap=12)

            print("Exiting State. Returning to Idle")
            self.next_state = 'idle'

        else:
            print("ESTOPPING!!")
            self.next_state = 'estop'




    def line_em_up(self): # comp 3
        '''Line up large blocks in rainbow order. Line up small blocks in separate line in rainbow order
        Some stacked (no more than 4 high). Block centers must be within 3 cm of a straight line. Each line length
        must be < 30 cm. 
        600s'''

        self.current_state = 'line_em_up'

        print("In Line em Up State")
        if self.next_state != "estop" or self.current_state == "estop":

            # need to detect if any blocks in the locations we are already going to place them!

            y = -25
            x0 = 425
            large_spacing = 62 # technically 62.5
            small_spacing = 48 # technically 48.5

            large_rainbow, small_rainbow = self.camera.rainbowOrder()

            print("Lining up large blocks")
            for i,block in enumerate(reversed(large_rainbow)):
                self.rxarm.grabBlock(block)
                x = x0 - large_spacing * i
                drop_location = (x,y,0)
                self.rxarm.dropBlock(drop_location)

            print("Lining up small blocks")
            for i,block in enumerate(reversed(small_rainbow)):
                self.rxarm.grabBlock(block)
                x = -x0 + small_spacing * i
                drop_location = (x,y,0)
                self.rxarm.dropBlock(drop_location)
            
            print("Exiting State. Returning to Idle")
            self.next_state = 'idle'

        else:
            print("ESTOPPING!!")
            self.next_state = 'estop'

    def stack_em_high(self): # comp 4
        '''Stack up the large blocks in one stack in rainbow color order
        Stack up the small blocks in another stack in rainbow color order
        Blocks placed in specific config (no more than 4 high stacks)
        Big and small blocks level 2.
        600s
        '''
        self.current_state = 'stack_em_high'

        print("In Stack em High State")
        if self.next_state != "estop" or self.current_state == "estop":

            # need to detect if any blocks in the locations we are already going to place them!

            large_rainbow, small_rainbow = self.camera.rainbowOrder()

            large_stack_placement, small_stack_placement = (400,-25), (-400, -25)

            for rainbow, stack_placement in zip([large_rainbow, small_rainbow], [large_stack_placement, small_stack_placement]):

                print(rainbow)
                print(stack_placement)
                stack_last_z = 0
                print("Stacking Blocks")
                for i, block in enumerate(rainbow):
                    print(block)
                    # Get Z and update
                    add_z = 40 if block['is_large'] else 26
                    z = stack_last_z
                    stack_last_z = z + add_z
                    x,y = stack_placement
                    # Pick up and drop off
                    drop_location = (x,y,z)
                    self.rxarm.grabBlock(block)
                    print("Last Grabbed Angle = " + str(self.rxarm.last_grabbed_angle))
                    if self.rxarm.last_grabbed_angle is not None: print("LHS = " + str(self.rxarm.last_grabbed_angle - (-np.pi/4)))
                    if self.rxarm.last_grabbed_angle is not None and np.abs(self.rxarm.last_grabbed_angle - (-np.pi/4)) < 0.0001:
                        print("Setting Drop Angle to -45 degrees!")
                        drop_angle = -np.pi/4
                    else:
                        drop_angle = 0
                    self.rxarm.dropBlock(drop_location, gap=2, drop_angle=drop_angle)
            
            print("Exiting State. Returning to Idle")
            self.next_state = 'idle'

        else:
            print("ESTOPPING!!")
            self.next_state = 'estop'

    def to_the_sky(self): # comp 5
        '''Stack up the large blocks in any color order. Stack as high as possible.
        Randomly placed blocks, some stacked no more than 3 high
        '''
        
        self.current_state = 'to_the_sky'

        if self.next_state != "estop" or self.current_state == "estop":

            large_blocks, _ = self.camera.largeAndSmall()
            stack_placement = (-275, 0)
            stack_last_z = 0
            
            print("In To the Sky State")
            for i, block in enumerate(large_blocks):
                z = stack_last_z
                stack_last_z = z + 40
                x,y = stack_placement
                drop_location = (x,y,z)
                self.rxarm.grabBlock(block)
                print("Last Grabbed Angle = " + str(self.rxarm.last_grabbed_angle))
                if i < 3:
                    print("Setting drop angle to -90 degrees")
                    drop_angle = -np.pi/2
                else:
                    print("Setting drop angle top 0 degrees")
                    drop_angle = 0
                self.rxarm.dropBlock(drop_location, gap=2, drop_angle=drop_angle)
            
            print("Exiting State. Returning to Idle")
            self.next_state = 'idle'

        else:
            print("ESTOPPING!!")
            self.next_state = 'estop'

    
    def unstack_stacks(self):

        self.current_state = 'unstack_stacks'

        if self.next_state != "estop" or self.current_state == "estop":

            stacked = True
            while stacked:
                print("In Stacked Loop")
                n = 0
                for block in self.camera.block_detections:
                    x,y,z = block['location']
                    if z > 55:
                        print("Found a stack!")
                        n += 1
                        newx, newy = self.getRandomOpenArea()
                        drop_location = (newx, newy, 0)
                        self.rxarm.grabBlock(block)
                        self.rxarm.dropBlock(drop_location)
                print("Found a total of " + str(n) + " stacks")
                if n == 0:
                    print("No stacks detected...")
                    stacked = False

                # Redetect stacks
                pose = (0,0, np.pi/2, 0, 0) # Lift arm to detect stacks
                self.rxarm.set_joint_positions(pose)
                self.camera.detectBlocksInDepthImage()

            print("Exiting State. Returning to Idle")
            self.next_state = 'idle'

        else:
            print("ESTOPPING!!")
            self.next_state = 'estop'
            
                
    def getRandomOpenArea(self, min_dist=20, bounds=(-250, -25, 250, 275)):
        xmin, ymin, xmax, ymax = bounds
        xmin_dist, ymin_dist = 9999, 9999
        xexclude = (-100, 100)
        yexclude = (-175, 175)
        moved_block_locations = []

        while True:
            print("Finding random open area")
            x = np.random.randint(xmin, xmax)
            y = np.random.randint(ymin, ymax)
            if x >= xexclude[0] and x <= xexclude[1]:
                if y >= yexclude[0] and y <= yexclude[1]:
                    print("Excluding (" + str(x) + "," + str(y) + ")")
                    continue # SKIP THIS. Dont hit the robot....
            print("Testing (" + str(x) + "," + str(y) + ")")
            # Check if area is has a block near this region
            print(self.camera.block_detections)
            for block in self.camera.block_detections:
                xblock, yblock, _ = block['location']
                add_width = 20 if block['is_large'] else 13
                xdist = np.abs(x - xblock) - add_width
                ydist = np.abs(y - yblock) - add_width
                print("xdist = " + str(xdist))
                print("ydist = " + str(ydist))
                if xdist < xmin_dist and ymin_dist < ydist:
                    xmin_dist = xdist
                    ymin_dist = ydist
            if xmin_dist > min_dist and ymin_dist > min_dist:
                print("Below min dist! Returning open area")
                return (x,y)
            # Reset min distance
            xmin_dist = 9999
            ymin_dist = 9999





    def unstack_blocks(self):
        '''Unstack all stacks. Should be run initially for all tasks with potential stacks. Eases task'''
        pass

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)