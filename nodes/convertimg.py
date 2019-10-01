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
import random
import math
from sensor_msgs.msg import Image as sImage
from cv_bridge import CvBridge
from PIL import Image


class TopicConverter(object):
    def __init__(self, topic = '/camera',depth_topic = '/rgbd',
                 scale = 2,winsize = 11, attenuation = 0.2 , alpha = 0,
                 model = 'model.model', cuda = 0, input_bag = None,
                 output_bag = None, nlights = 2, lightposes = 'DLDR'):
        self.scale = scale
        self.model = model
        self.device = torch.device("cuda" if cuda else "cpu")
        self.bridge = CvBridge()
        self.depth_img = np.array([])
        self.winsize = winsize
        self.attenuation = attenuation
        self.alpha = alpha
        self.depth_topic = depth_topic
        self.lightposes = lightposes
        self.nlights = nlights

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
        print('saving img')
        cv2.imwrite('/home/olaya/catkin_ws/src/uw_img_sim/nodes/depth_panel.jpg',cv_img)


        if self.scale is not None:
            cv_img = cv2.resize(cv_img,
                                dsize = (int(cv_img.shape[1] / self.scale),
                                         int(cv_img.shape[0] / self.scale)),
                                interpolation = cv2.INTER_CUBIC)
        self.depth_img = cv_img

    def img_callback(self,msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        print('saving rgb')
        cv2.imwrite('/home/olaya/catkin_ws/src/uw_img_sim/nodes/rgb_panel.jpg',cv_img)

        if self.scale is not None:
            cv_img = cv2.resize(cv_img,
                                dsize = (int(cv_img.shape[1] / self.scale),
                                         int(cv_img.shape[0] / self.scale)),
                                interpolation = cv2.INTER_CUBIC)
        # Apply blur
        if self.depth_img.any():
            cv_img = self.depth_blur(cv_img)

        # Add lights (if any)
        # first obtain lights pose
        if self.nlights > 0:
            lights = np.empty([self.nlights,2])
            for i in range (self.nlights-1):
                if self.lightposes[i*2] == 'L' or self.lightposes[i*2+1] == 'L':
                    lights[i][0] = 0 #left
                elif self.lightposes[i*2] == 'R' or self.lightposes[i*2+1] == 'R':
                    lights[i][0] = cv_img.shape[1] # right

                elif self.lightposes[i*2] == 'T' or self.lightposes[i*2+1] == 'T':
                    lights[i][1] = 0 #top
                elif self.lightposes[i*2] == 'B' or self.lightposes[i*2+1] == 'B':
                    lights[i][1] = cv_img.shape[0] #bottom
            for l in lights:
                print (lights)
                print (l)
                print(l[0],l[1])
                light1_pose = (l[0],l[1])
                cv_img = self.add_sun_flare(cv_img , flare_center = light1_pose ,radius = 300)
            # cv_img = [self.add_sun_flare(cv_img , flare_center = (l[0],l[1]),radius = 300) for l in lights]

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

    def gaussian_flare(self, shape = (10,10), size_y=None, center = (0,0), radius = 5, rgb = (255,255,255)):
        size = shape[0]/2
        size_y=shape[1]/2
        sigma = radius

        # generate rgb array with light color
        rgbimg = np.ones((shape[0],shape[1],1),dtype = float)

        # generate the alpha channel
        x, y = np.mgrid[-size:size, -size_y:size_y]
        x = x - center[0]/2
        y = y - center[1]/2

        g = np.exp(-(x**2/float(2*sigma**2)+y**2/float(2*sigma**2)))
        g = g / g.sum()
        alpha_channel = np.interp(g, (g.min(), g.max()), (0, 1)).astype(float)


        return cv2.merge((rgbimg*rgb[0], rgbimg*rgb[1],rgbimg*rgb[2], alpha_channel))

    def flare_source(self, image, point,radius,src_color):
        # Gaussian flare
        flare = self.gaussian_flare(shape = image.shape ,center = point, radius = radius)
        alpha_channel = flare[:,:,3]

        # change color space of image to YCrCb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(float)

        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(alpha_channel, flare[:,:,0])

        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - alpha_channel, image[:,:,0])

        # Add the masked foreground and background.
        outImage = cv2.add(foreground, background)

        out = cv2.merge((outImage,image[:,:,1],image[:,:,2])).astype(np.uint8)

        return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

    def add_sun_flare_line(self, flare_center,angle,imshape):
        x=[]
        y=[]
        i=0
        for rand_x in range(0,imshape[1],10):
            rand_y= math.tan(angle)*(rand_x-flare_center[0])+flare_center[1]
            x.append(rand_x)
            y.append(2*flare_center[1]-rand_y)
        return x,y

    def add_sun_process(self, image, no_of_flare_circles,flare_center,radius,x,y,src_color):
        overlay= image.copy()
        output= image.copy()
        imshape=image.shape
        for i in range(no_of_flare_circles):
            alpha=random.uniform(0.05,0.2)
            r=random.randint(0, len(x)-1)
            rad=random.randint(1, imshape[0]//100-2)
            cv2.circle(overlay,(int(x[r]),int(y[r])), rad*rad*rad, (random.randint(max(src_color[0]-50,0), src_color[0]),random.randint(max(src_color[1]-50,0), src_color[1]),random.randint(max(src_color[2]-50,0), src_color[2])), -1)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
        output= self.flare_source(output,(int(flare_center[0]),int(flare_center[1])),radius,src_color)
        return output

    def add_sun_flare(image,flare_center=-1, angle=-1, no_of_circles=8,radius=400, src_color=(255,255,255)):
        """ Adding light flare
        :param image: image
        :param flare_center: center of light in coordinates (x,y)
        :param angle: angle of flare, default random
        :param no_of_circles: number of secondary circles
        :param radius: radius of flare
        :param src_color: color of flare, default white

        Returns flared image"""
        image_RGB=[]
        if(angle!=-1):
            angle=angle%(2*math.pi)
        if not(no_of_circles>=0 and no_of_circles<=20):
            raise Exception(err_flare_circle_count)

        imshape=image.shape
        if(angle==-1):
            angle_t=random.uniform(0,2*math.pi)
            if angle_t==math.pi/2:
                angle_t=0
        else:
            angle_t=angle

        if flare_center==-1:
            flare_center_t=(random.randint(0,imshape[1]),random.randint(0,imshape[0]//2))
        else:
            flare_center_t=flare_center
        x,y= self.add_sun_flare_line(flare_center_t,angle_t,imshape)
        output= self.add_sun_process(image, no_of_circles,flare_center_t,radius,x,y,src_color)
        image_RGB = output
        return image_RGB







def main():
    rospy.init_node('convertbag', log_level=rospy.INFO)


    if rospy.get_param("~cuda") and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)


    TopicConverter(rospy.get_param("~topic"),rospy.get_param("~depth-topic"),
                   rospy.get_param("~scale"),rospy.get_param("~winsize"),
                   rospy.get_param("~attenuation"),rospy.get_param("~alpha"),
                   rospy.get_param("~model"), rospy.get_param("~cuda"),
                   rospy.get_param("~input-bag"),rospy.get_param("~output-bag"),
                   rospy.get_param("~nlights"),rospy.get_param("~light-pose"))
    rospy.spin()


if __name__ == "__main__":
    main()
