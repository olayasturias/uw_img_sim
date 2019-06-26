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
    def __init__(self,args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        image_sub = rospy.Subscriber(args.topic, sImage,self.img_callback,
                                     queue_size = 1, buff_size = 52428800)
        rospy.loginfo("Subscribed to %s topic", args.topic)
        self.img_pub = rospy.Publisher(args.topic+'/underwater',
                                       sImage,
                                       queue_size=1)

    def img_callback(self,msg):
        bridge = CvBridge()
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        cv_img = Image.fromarray(cv_img)

        if self.args.scale is not None:
            cv_img = cv_img.resize((int(cv_img.size[0] / self.args.scale),
                                    int(cv_img.size[1] / self.args.scale)),
                                    Image.ANTIALIAS)

        device = torch.device("cuda" if self.args.cuda else "cpu")
        content_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x.mul(255))
                            ])
        content_image = content_transform(cv_img)
        content_image = content_image.unsqueeze(0).to(device)
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(self.args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if self.args.export_onnx:
                assert self.args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, self.args.input_bag, self.args.export_onnx).cpu()
            else:
                rospy.loginfo('stylizing image ...')
                output = style_model(content_image).cpu()
                img = output[0].clone().clamp(0, 255).numpy()
                img = img.transpose(1, 2, 0).astype("uint8")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cvoutput = bridge.cv2_to_imgmsg(img, "bgr8")
                self.img_pub.publish(cvoutput)





def main():
    rospy.init_node('convertbag', log_level=rospy.INFO)
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")

    eval_arg_parser.add_argument("--topic", type=str, required=False,
                                 help="path to directory with images you want to stylize")
    eval_arg_parser.add_argument("--scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--model", type=str, required=False,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=False,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        TopicConverter(args)
        rospy.spin()


if __name__ == "__main__":
    main()
