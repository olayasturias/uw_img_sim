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

def stylize(args):
    content_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.mul(255))
                        ])


    print('Reading bag file '+ os.getcwd() + '/' + args.input_bag
          + ' topic ' + args.topic)
    readbag = rosbag.Bag(args.input_bag,'r')
    bridge = CvBridge()

    with rosbag.Bag(args.output_bag,'w') as outbag:
        for topic, msg, dtime in readbag.read_messages():
            if topic == args.topic:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                cv_img = Image.fromarray(cv_img)
                if args.scale is not None:
                    cv_img = cv_img.resize((int(cv_img.size[0] / args.scale),
                                            int(cv_img.size[1] / args.scale)),
                                            Image.ANTIALIAS)

                device = torch.device("cuda" if args.cuda else "cpu")
                content_image = content_transform(cv_img)
                content_image = content_image.unsqueeze(0).to(device)

                with torch.no_grad():
                    style_model = TransformerNet()
                    state_dict = torch.load(args.model)
                    # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
                    for k in list(state_dict.keys()):
                        if re.search(r'in\d+\.running_(mean|var)$', k):
                            del state_dict[k]
                    style_model.load_state_dict(state_dict)
                    style_model.to(device)
                    if args.export_onnx:
                        assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                        output = torch.onnx._export(style_model, args.input_bag, args.export_onnx).cpu()
                    else:
                        rospy.loginfo('stylizing image ...')
                        output = style_model(content_image).cpu()
                        img = output[0].clone().clamp(0, 255).numpy()
                        img = img.transpose(1, 2, 0).astype("uint8")
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cvoutput = bridge.cv2_to_imgmsg(img, "bgr8")
                        outbag.write(topic, cvoutput, msg.header.stamp if msg._has_header else dtime)
            else:
                outbag.write(topic, msg, msg.header.stamp if msg._has_header else dtime)

    readbag.close()





def main():
    rospy.init_node('convertbag', log_level=rospy.INFO)
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--input-bag", type=str, required=False,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--topic", type=str, required=False,
                                 help="path to directory with images you want to stylize")
    eval_arg_parser.add_argument("--scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-bag", type=str, required=False,
                                 help="path for saving the output image")
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
        stylize(args)


if __name__ == "__main__":
    main()
