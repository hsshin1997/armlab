"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError
import copy
# from state_machine import quaternion_rotation_matrix

from tf.transformations import quaternion_matrix, rotation_matrix


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








class Camera():
    """!
    @brief      This class describes a camera.
    """
    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.BlockFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.array([[943.7872828, 0.0, 680.7729052],
                                          [0.0, 940.9715492, 378.9719702],
                                          [0.0, 0.0, 1.0]])
        # self.extrinsic_matrix = np.array([[1.0, 0.0, 0.0, 9],
        #                                   [0.0, -1.0, 0.0, 144],
        #                                   [0.0, 0.0, -1.0, 973],
        #                                   [0.0, 0.0, 0.0, 1.0]])
        self.extrinsic_matrix = np.array([[1.0, 0.0, 0.0, 9],
                                    [0.0, -1.0, 0.0, 144],
                                    [0.0, 0.0, -1.0, 973],
                                    [0.0, 0.0, 0.0, 1.0]])
        print("Initial Matrices")
        print(self.extrinsic_matrix)
        print(self.intrinsic_matrix)
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        self.tag_detections_recent = np.zeros([4,3])
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])
        self.rainbow_color_order = ['red', 'orange', 'yellow', 'green', 'blue']
        self.board_depths = {}
        self.board_depths_arr = np.array([])


    def uvToWorld(self, uv):
        print("Using These Matrices")
        print(self.intrinsic_matrix)
        print(self.extrinsic_matrix)
        u,v = uv
        z = self.DepthFrameRaw[v][u]
        frame_c = z * np.dot(np.linalg.inv(self.intrinsic_matrix), np.array([u, v, 1]).reshape((3, 1)))
        frame_c = np.vstack((frame_c, 1))
        frame_w = np.dot(np.linalg.inv(self.extrinsic_matrix), frame_c)
        return (frame_w[0, 0], frame_w[1, 0], frame_w[2, 0])

    def uvToCamera(self, uv):
        u,v = uv
        z = self.DepthFrameRaw[v][u]
        frame_c = z * np.dot(np.linalg.inv(self.intrinsic_matrix), np.array([u, v, 1]).reshape((3, 1)))
        frame_c = np.vstack((frame_c, 1))
        return (frame_c[0, 0], frame_c[1, 0], frame_c[2, 0])

    def cameraToWorld(self, camera_frame):
        frame_w = np.dot(np.linalg.inv(self.extrinsic_matrix), camera_frame)
        return frame_w

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtBlockFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.BlockFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def retrieve_area_color(self, data, contour, labels):
        mask = np.zeros(data.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean = cv2.mean(data, mask=mask)[:3]
        print("Mean = " + str(mean))
        min_dist = (np.inf, None)
        for label in labels:
            d = np.linalg.norm(label["color"] - np.array(mean))
            print(label['id'] + " = " + str(d))
            if d < min_dist[0]:
                min_dist = (d, label["id"])
        print("Selected " + str(min_dist[1]))
        return min_dist[1]

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        colors = list((
            {'id': 'red', 'color': (84, 87, 93)},
            {'id': 'orange', 'color': (124, 121, 86)},
            {'id': 'yellow', 'color': (144, 190, 83)},
            {'id': 'green', 'color': (47, 143, 125)},
            {'id': 'blue', 'color': (1, 106, 139)},
            # {'id': 'violet', 'color': (26, 94, 132)}
        ))

        # template_match = self.getBoardDepthCalibrationPoints()
        # template_match_array = np.array([list(v) for v in self.template_match.values()])

        lower = 600
        upper = 960
        rgb_image = self.VideoFrame
        # cv2.imwrite('rgb.png', rgb_image)
        cnt_image = self.VideoFrame
        depth_data = self.DepthFrameRaw
        # cv2.imwrite('depth.png', depth_data)
        """mask out arm & outside board"""
        mask = np.zeros_like(depth_data, dtype=np.uint8)
        cv2.rectangle(mask, (250,95),(1160,695), 255, cv2.FILLED)
        cv2.rectangle(mask, (610,390),(800,695), 0, cv2.FILLED)
        cv2.rectangle(cnt_image, (250,95),(1160,695), (255, 0, 0), 2)
        cv2.rectangle(cnt_image, (610,390),(800,695), (255, 0, 0), 2)
        thresh = cv2.bitwise_and(cv2.inRange(depth_data, lower, upper), mask)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite("morph_thresh.png", thresh)
        # cv2.imshow("thresh", thresh)
        # depending on your version of OpenCV, the following line could be:
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cnt_image, contours, -1, (0,255,255), thickness=2)
        blocks = []
        for contour in contours:
            try:
                # hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
                # hsv_image[:, :, 1] = 100
                # hsv_image[:, :, 2] = 100
                color = self.retrieve_area_color(rgb_image, contour, colors)
                min_rect = cv2.minAreaRect(contour)
                w,h = min_rect[1]
                if w < 15 or h < 15 or w > 200 or h > 200: # skip this detection
                    continue
                is_large = (w+h)/2 > 36 # arbitrary, look at typical values of min rect
                theta = min_rect[2]
                theta = int(abs(theta) % 90) # Ensure theta is between 0 and 90 positive
                theta = theta / 180.0 * np.pi
                M = cv2.moments(contour)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                z = depth_data[cy, cx]
                nz = self.getNearestBoardDepth((cy, cx))
                corrected_z = z - nz # Use this or wz??? corrected z uses adjusted z based on nearest board depth reading
                wx, wy, wz = self.uvToWorld((cx, cy))
                wz += 19
                print((cx, cy))
                print((wx, wy, wz, theta))
                print((wx, wy, corrected_z, theta))
                cv2.putText(cnt_image, color, (cx-30, cy+40), font, 1.0, (0,0,0), thickness=2)
                cv2.putText(cnt_image, str(int(theta * 180 / np.pi)), (cx, cy), font, 0.5, (255,255,255), thickness=2)
                # import uuid
                # from collections import Counter
                # print("Contour = " + str(contour))
                block_info = {
                    # 'id':uuid.uuid4(),
                    'color':color,
                    'location':(wx, wy, wz), # changed from (wx, wy, wz)
                    'theta':theta,
                    'is_large':is_large # TODO: determine if large or small
                }
                # See if this is the same block
            #     found_block = False
            #     for block in self.block_detections:
            #         x1, y1, z1 = block['location']
            #         dx = abs(wx - x1)
            #         dy = abs(wy - y1)
            #         dz = abs(z - z1)
            #         if dx < 10 and dy < 10 and dz < 10: # found same block
            #             print("Found same block!")
            #             found_block = True
            #             if len(block['prev_theta']) < 40: # use last 20 measurements of theta for average
            #                 block['prev_theta'] += [theta]
            #             else:
            #                 block['prev_theta'] = block['prev_theta'][1:] + [theta]
            #             updated_theta_array = np.array(block['prev_theta'] + [theta])
            #             # block['theta'] = np.mean(updated_theta_array)
            #             block['theta'] = np.median(updated_theta_array)
            #             print("Averaging " + str(len(block['prev_theta'])) + " measurements: " + ','.join([str(i) for i in block['prev_theta']]))
            #             block_info = block
            #     if not found_block:
            #         block_info['prev_theta'] = [theta]

                blocks += [block_info]
            except Exception as e:
                print(e)
        self.block_detections = np.array(blocks) # Set self block detections. Should we implement this in the other function?
        self.BlockFrame = cnt_image
        print("Number of blocks detected = " + str(len(self.block_detections)))
        return self.block_detections # If > 0, at least one block has been detected. Return True in this case
        # return True

    def get_april_matrix(self):
        for tag in self.tag_detections.detections:
            print(tag.id)
            if tag.id == (1, 2, 3, 4):
                tag_pos = np.array([tag.pose.pose.pose.position.x, tag.pose.pose.pose.position.y, tag.pose.pose.pose.position.z])
                # tag_pos *= 1000.0
                tag_orientation = np.array([tag.pose.pose.pose.orientation.x, 
                                            tag.pose.pose.pose.orientation.y,
                                            tag.pose.pose.pose.orientation.z, 
                                            tag.pose.pose.pose.orientation.w])
                return tag_pos, tag_orientation
                # rot_matrix = np.dot(quaternion_matrix(tag_orientation), np.array([[0, 1, 0, 0],[-1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]]))
                # H_tag2cam = np.vstack((np.hstack((rot_matrix[0:3, 0:3], tag_pos)), np.array([0, 0, 0, 1])))
                # # print("Extirnsic matrix: ")
                # # print(H_tag2cam)
                # self.extrinsic_matrix = H_tag2cam

    def getBoardDepthCalibrationPoints(self):

        def getCumsumIntermediateValues(array, thresh):
            nvals_per_col = [len(j) for j in ''.join([i for i in np.where(array <= thresh, '0', '1')]).split('1')]
            tups = []
            i = 0
            n = 0
            for nvals in nvals_per_col:
                i = n
                n += nvals
                tups += [(i, n)]
                i += 1
                n += 1
            tups
            add = np.array([array[i:j].sum() for i,j in tups]).cumsum()
            return add

        # Use template matching to extract known points on board (this will exclude noise from blocks/robot/background)
        rgb_image = cv2.imread("rgb.png") # self.VideoFrame.copy()
        depth_data = cv2.imread("depth.png", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYDEPTH) # self.DepthFrameRaw.copy()
        template = cv2.imread("board_template.png")
        gs_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        template = template[5:-5, 5:-5]
        h, w = template.shape[:2]
        gs_template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        output_image = cv2.matchTemplate(gs_image, gs_template, cv2.TM_CCOEFF_NORMED)
        cv2.imwrite("output_image.png", output_image)
        ret, thresh_image = cv2.threshold(output_image, 0.9, 1.0, cv2.THRESH_BINARY)
        cv2.imwrite("thresh_image.png", thresh_image)

        # Extract Depth Points from template matches
        max_loc = output_image.shape
        top_pad = h // 2
        left_pad = w // 2
        bottom_pad = h // 2 - 1
        right_pad = w // 2 - 1
        padded_mask = cv2.copyMakeBorder(thresh_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, 0) # thresh image or output image. check if not working
        extracted_depth = padded_mask * depth_data
        y,x = np.where(padded_mask == [1])

        # Get rows and columns of pixels

        # Cols
        x_sorted_indices = x.argsort()
        x_xsort = x[x_sorted_indices]
        x_diff = x_xsort[1:] - x_xsort[:-1] # Get change between adjacent x vals (cols). Indicates a change in row
        x_min = x_xsort[0]
        x_max = x_xsort[-1]
        x_changes = x_diff[x_diff > w//2] # Need to fix to account for 1's that are missed
        x_add = getCumsumIntermediateValues(x_diff, w//2) # Accounts for 1's that are missed!
        x_colpix = np.array([x_min] + (x_changes.cumsum() + x_min + x_add[:-1]).tolist())
        x_cols = np.array([np.abs(xi - x_colpix).argmin() for xi in x]) # Is using 

        # Rows
        y_sorted_indices = y.argsort()
        y_ysort = y[y_sorted_indices]
        y_diff = y_ysort[1:] - y_ysort[:-1]
        y_min = y_ysort[0]
        y_max = y_ysort[-1]
        y_changes = y_diff[y_diff > h//2]
        y_add = getCumsumIntermediateValues(y_diff, h//2) # Accounts for 1's that are missed!
        y_rowpix = np.array([y_min] + (y_changes.cumsum() + y_min + y_add[:-1]).tolist())
        y_rows = np.array([np.abs(yi - y_rowpix).argmin() for yi in y])

        # Put depths in (row,col) indexed dictionary
        depths = {}
        for xdata, ydata in zip(zip(x,x_cols),zip(y,y_rows)):
            xpix,col = xdata
            ypix,row = ydata
            d = depth_data[ypix,xpix]
        depths[(row,col)] = (ypix,xpix,d)

        # Return a dict of (row,col) keys containing uncalibrated (U,V,D) pixels
        print(depths)
        self.board_depths = depths
        self.board_depths_arr = np.array([i for i in depths.values()])

        # Testing
        rgb_image2 = rgb_image.copy()
        for yi,xi,d in depths.values():
            cv2.circle(rgb_image2, (xi, yi), 10, (255,255,0), 1)
        cv2.imwrite("image_with_circles.png", rgb_image2)

        return depths

    def rainbowOrder(self):
        '''Return block detections in rainbow color order
        
        Returns 2 arrays, sorted in rainbow order. First array is large blocks, second array is small blocks'''
        block_list_large = []
        block_list_small = []
        colors = {'red':[], 'orange':[], 'yellow':[], 'green':[], 'blue':[]}
        for block in self.block_detections:
            colors[block['color']] += [block]
        for color in self.rainbow_color_order:
            for block in colors[color]:
                if block['is_large']:
                    block_list_large += [block]
                else:
                    block_list_small += [block]
        return np.array(block_list_large), np.array(block_list_small)

    def largeAndSmall(self):
        block_list_large = []
        block_list_small = []
        for block in self.block_detections:
            if block['is_large']:
                block_list_large += [block]
            else:
                block_list_small += [block]
        return np.array(block_list_large), np.array(block_list_small)

    def sortByDistance(self, block_info_list):
        blocks_distance_sorted = []
        for block in block_info_list:
            x,y, _ = block['location']
            dist = x**2 + y**2
            blocks_distance_sorted += [dist]
        blocks_distance_sorted = np.array(blocks_distance_sorted)
        sorted_indices = blocks_distance_sorted.argsort()
        blocks = block_info_list[sorted_indices]
        return blocks

    def getTopOfStacks(self):
        # cnt_image = self.VideoFrame
        depth_data = self.DepthFrameRaw
        # cv2.imwrite('depth.png', depth_data)
        """mask out arm & outside board"""
        mask = np.zeros_like(depth_data, dtype=np.uint8)
        cv2.rectangle(mask, (240,150),(1100,690), 255, cv2.FILLED)
        cv2.rectangle(mask, (580,100),(770,690), 0, cv2.FILLED)
        # cv2.rectangle(cnt_image, (240,150),(1100,690), (255, 0, 0), 2)
        # cv2.rectangle(cnt_image, (580,100),(770,690), (255, 0, 0), 2)
        depth_data

    def getNearestBoardDepth(self, pixel):
        idx = np.square(np.abs(self.board_depths_arr[:, :2] - pixel)).sum(axis=1).argmin()
        d = self.board_depths_arr[idx, 2]
        return d
    





class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        for detection in data.detections:
            # print(detection.id[0])
            # print(detection.pose.pose.pose.position)
            # print(type(detection.id[0]))
            # print(type(detection.pose.pose.pose.position.x))
            # print(type(detection.pose.pose.pose.position))
            x = detection.pose.pose.pose.position.x
            y = detection.pose.pose.pose.position.y
            z = detection.pose.pose.pose.position.z
            self.camera.tag_detections_recent[detection.id[0]-1] = np.array([x, y, z])



class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        #print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Block window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            block_frame = self.camera.convertQtBlockFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame, block_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                self.camera.detectBlocksInDepthImage()
                cal_points = self.camera.getBoardDepthCalibrationPoints()
                large, small = self.camera.largeAndSmall()
                large_rainbow, small_rainbow = self.camera.rainbowOrder()
                print("Cal Points = " + str(cal_points))
                print("Large Blocks = " + str(large))
                print("Small Blocks = " + str(small))
                print("Large Rainbow = " + str(large_rainbow))
                print("Small Rainbow = " + str(small_rainbow))
                cv2.imwrite("block_colors_blockframe.png", self.camera.BlockFrame)
                cv2.imwrite("block_colors_imageframe.png", self.camera.VideoFrame)
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow(
                    "Block window",
                    cv2.cvtColor(self.camera.BlockFrame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
