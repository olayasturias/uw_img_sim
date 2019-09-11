#!/usr/bin/env python3

import sys
import rospy
import os
import re
import datetime
import argparse
import utils
import numpy as np
from torchvision import transforms
from transformer_net import TransformerNet
import torch
import rosbag
import cv2
from sensor_msgs.msg import Image as sImage
from cv_bridge import CvBridge
from PIL import Image


class TopicConverter(object):
    def __init__(self, topic = '/camera',depth_topic = '/rgbd',
                 scale = 2,winsize = 11, attenuation = 0.2 , alpha = 0,
                 model = 'model.model', cuda = 0, input_bag = None,
                 output_bag = None):
        self.scale = scale
        self.model = model
        self.device = torch.device("cuda" if cuda else "cpu")
        self.bridge = CvBridge()
        self.depth_img = np.array([])
        self.winsize = winsize
        self.attenuation = attenuation
        self.alpha = alpha
        self.depth_topic = depth_topic

        if input_bag == 'None' and output_bag == 'None' :
            image_sub = rospy.Subscriber(topic, sImage,self.img_callback,
                                         queue_size = 1, buff_size = 52428800)
            depth_sub = rospy.Subscriber(depth_topic, sImage,self.depth_callback,
                                         queue_size = 1, buff_size = 52428800)
            rospy.loginfo("Subscribed to img topic %s", topic)
            rospy.loginfo("Subscribed to depth topic %s", depth_topic)
            self.img_pub = rospy.Publisher(topic+'/underwater',
                                           sImage,
                                           queue_size=1)
        else:
            self.bag_read(input_bag, topic, output_bag)


    def depth_callback(self,msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        cv_img = np.where(cv_img!=0, cv_img, 200)
        #cv_img = Image.fromarray(cv_img)


        if self.scale is not None:
            cv_img = cv2.resize(cv_img,
                                dsize = (int(cv_img.shape[1] / self.scale),
                                         int(cv_img.shape[0] / self.scale)),
                                interpolation = cv2.INTER_CUBIC)
        self.depth_img = cv_img

    def img_callback(self,msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        if self.scale is not None:
            cv_img = cv2.resize(cv_img,
                                dsize = (int(cv_img.shape[1] / self.scale),
                                         int(cv_img.shape[0] / self.scale)),
                                interpolation = cv2.INTER_CUBIC)
        # Apply blur
        if self.depth_img:
            cv_img = self.depth_blur(cv_img)

        # Apply style transfer
        converted = self.convert(cv_img)


        # Publish
        self.img_pub.publish(converted)

    def bag_read(self, filein, topicin, fileout):

        print('Reading bag file '+ filein + ' topic ' + topicin)
        readbag = rosbag.Bag(filein,'r')


        with rosbag.Bag(fileout,'w') as outbag:
            for topic, msg, dtime in readbag.read_messages():
                if topic == self.depth_topic:
                    self.depth_callback(msg)
                    outbag.write(topic, msg, msg.header.stamp if msg._has_header else dtime)

                elif topic == topicin:
                    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                    #cv_img = Image.fromarray(cv_img)
                    if self.scale is not None:
                        cv_img = cv2.resize(cv_img,
                                            dsize = (int(cv_img.shape[1] / self.scale),
                                                     int(cv_img.shape[0] / self.scale)),
                                            interpolation = cv2.INTER_CUBIC)
                    # Apply blur
                    if self.depth_img.size != 0:
                        cv_img = self.depth_blur(cv_img)
                    else:
                        print('no depth image yet')
                    # Apply style transfer
                    converted = self.convert(cv_img)
                    outbag.write(topic, converted, dtime)
                else:
                    outbag.write(topic, msg, dtime)

        rospy.loginfo('Bag conversion finished')
        readbag.close()


    def convert(self, orig_image):
        #pil_img = Image.fromarray(orig_image)
        pil_img = orig_image

        content_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.mul(255))
                            ])
        content_image = content_transform(pil_img)
        content_image = content_image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(self.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(self.device)

            rospy.loginfo('stylizing image ...')
            output = style_model(content_image).cpu()
            img = output[0].clone().clamp(0, 255).numpy()
            img = img.transpose(1, 2, 0).astype("uint8")

            img = cv2.addWeighted( orig_image, self.alpha,
                                   img[0:orig_image.shape[0],
                                   0:orig_image.shape[1]],1-self.alpha,0.0)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cvoutput = self.bridge.cv2_to_imgmsg(img, "bgr8")
            return cvoutput

    def depth_blur(self,img):
        (winW, winH) = (self.winsize, self.winsize)
        alpha = self.attenuation
        for (x, y, window) in utils.sliding_window(img, stepSize=self.winsize-6,
                                                   windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue


            #print(int(y+winW/3))
            #print(int(x+winH/3))
            img[y:y + winW, x:x + winH] = cv2.GaussianBlur(img[y:y + winW, x:x + winH],
                                                           (winW, winH),
                                                           alpha*self.depth_img[int(y+winW/2),int(x+winH/2)])


        return img







def main():
    rospy.init_node('convertbag', log_level=rospy.INFO)


    if rospy.get_param("~cuda") and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)


    TopicConverter(rospy.get_param("~topic"),rospy.get_param("~depth-topic"),
                   rospy.get_param("~scale"),rospy.get_param("~winsize"),
                   rospy.get_param("~attenuation"),rospy.get_param("~alpha"),
                   rospy.get_param("~model"), rospy.get_param("~cuda"),
                   rospy.get_param("~input-bag"),rospy.get_param("~output-bag"))
    rospy.spin()


if __name__ == "__main__":
    main()
