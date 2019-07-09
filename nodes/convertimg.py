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
    def __init__(self, topic = '/camera', scale = 2, model = 'model.model',
                 cuda = 0, input_bag = None, output_bag = None):
        self.scale = scale
        self.model = model
        self.device = torch.device("cuda" if cuda else "cpu")
        self.bridge = CvBridge()
        print (input_bag)
        print(output_bag)

        if input_bag == 'None' and output_bag == 'None' :
            image_sub = rospy.Subscriber(topic, sImage,self.img_callback,
                                         queue_size = 1, buff_size = 52428800)
            rospy.loginfo("Subscribed to %s topic", topic)
            self.img_pub = rospy.Publisher(topic+'/underwater',
                                           sImage,
                                           queue_size=1)
        else:
            self.bag_read(input_bag, topic, output_bag)



    def img_callback(self,msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        cv_img = Image.fromarray(cv_img)

        if self.scale is not None:
            cv_img = cv_img.resize((int(cv_img.size[0] / self.scale),
                                    int(cv_img.size[1] / self.scale)),
                                    Image.ANTIALIAS)

        content_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.mul(255))
                            ])
        content_image = content_transform(cv_img)
        content_image = content_image.unsqueeze(0).to(self.device)

        converted = self.convert(content_image)
        self.img_pub.publish(converted)

    def bag_read(self, filein, topicin, fileout):
        content_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.mul(255))
                            ])
        print('Reading bag file '+ filein + ' topic ' + topicin)
        readbag = rosbag.Bag(filein,'r')


        with rosbag.Bag(fileout,'w') as outbag:
            for topic, msg, dtime in readbag.read_messages():
                if topic == topicin:
                    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                    cv_img = Image.fromarray(cv_img)
                    if self.scale is not None:
                        cv_img = cv_img.resize((int(cv_img.size[0] / self.scale),
                                                int(cv_img.size[1] / self.scale)),
                                                Image.ANTIALIAS)

                    content_image = content_transform(cv_img)
                    content_image = content_image.unsqueeze(0).to(self.device)

                    converted = self.convert(content_image)
                    outbag.write(topic, converted, msg.header.stamp
                                 if msg._has_header else dtime)
                else:
                    outbag.write(topic, msg, msg.header.stamp if msg._has_header else dtime)

        rospy.loginfo('Bag conversion finished')
        readbag.close()


    def convert(self, content_image):
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
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cvoutput = self.bridge.cv2_to_imgmsg(img, "bgr8")
            return cvoutput






def main():
    rospy.init_node('convertbag', log_level=rospy.INFO)


    if rospy.get_param("~cuda") and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)


    TopicConverter(rospy.get_param("~topic"), rospy.get_param("~scale"),
                   rospy.get_param("~model"), rospy.get_param("~cuda"),
                   rospy.get_param("~input-bag"),rospy.get_param("~output-bag"))
    rospy.spin()


if __name__ == "__main__":
    main()
