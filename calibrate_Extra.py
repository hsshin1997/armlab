        tag_detection_dict = {}
        for tag in self.tag_detections.detections:
            print(tag.id)
            if tag.id[0] in (1, 2, 3, 4): # tag == (1, 2, 3, 4)
                print(" in if statement")
                tag_detection_dict[tag.id[0]] = tag
                if len(list(tag_detection_dict.keys())) == 4:
                    print("breaking loop")
                    break
        # tag_detection_array = np.zeros(())
        for tag_id in [1, 2, 3, 4]:
            tag = tag_detection_dict[tag_id]
            print(tag_id)
            tag_pos = np.array([tag.pose.pose.pose.position.x, tag.pose.pose.pose.position.y, tag.pose.pose.pose.position.z]).reshape((3, 1))
            print(tag_pos)
            tag_pos *= 1000.0
            tag_orientation = np.array([tag.pose.pose.pose.orientation.x, 
                                        tag.pose.pose.pose.orientation.y,
                                        tag.pose.pose.pose.orientation.z, 
                                        tag.pose.pose.pose.orientation.w])
            print(tag_orientation)
            rot_matrix = np.dot(quaternion_matrix(tag_orientation), np.array([[0, 1, 0, 0],[-1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]]))
            H_tag2cam = np.vstack((np.hstack((rot_matrix[0:3, 0:3], tag_pos)), np.array([0, 0, 0, 1])))
            # print("Extirnsic matrix: ")
            # print(H_tag2cam)
            self.extrinsic_matrix = H_tag2cam