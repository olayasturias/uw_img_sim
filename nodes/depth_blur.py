#!/usr/bin/env python3
import sys
import rospy
import os
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TopicConverter(object):
    def __init__(self,depth_topic = '/depth', rgb_topic = '/rgb'):
        self.bridge = CvBridge()
        self.depth_img = []
        self.bridge = CvBridge()

        print('depth topic is ', depth_topic)
        print('rgb topic is ', rgb_topic)


        rgb_sub = rospy.Subscriber(rgb_topic, Image, self.rgb_callback,
                                     queue_size = 1)
        kinect_sub = rospy.Subscriber(depth_topic, Image, self.kinect_callback,
                                      queue_size = 1, buff_size = 52428800)
        self.img_pub = rospy.Publisher('/blurred',
                                       Image,
                                       queue_size=1)



    def rgb_callback(self,msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        alpha = 0.2
        img_res = cv_img[:,:,1]*alpha + self.depth_img*(1-alpha)

        cv2.imwrite('rgb.jpg',cv_img)
        cv2.imwrite('depth.jpg',self.depth_img)


        img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        cvoutput = self.bridge.cv2_to_imgmsg(img, "bgr8")

        self.img_pub.publish(cvoutput)

        cv2.imshow('image windw',self.depth_img.astype(np.uint8))
        cv2.waitKey(3)



    def kinect_callback(self,msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        print (np.amin(cv_img))
        print (np.amax(cv_img))
        self.depth_img = cv_img
        # Create Gaussian kernel according to distance








def main():
    rospy.init_node('convertbag', log_level=rospy.INFO)

    print(rospy.get_param("~depth-topic"))
    print(rospy.get_param("~rgb-topic"))



    TopicConverter(depth_topic = rospy.get_param("~depth-topic"),
                   rgb_topic = rospy.get_param("~rgb-topic"))
    rospy.spin()


if __name__ == "__main__":
    main()
