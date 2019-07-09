#!/usr/bin/env python3

import sys
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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    content_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.mul(255))
                        ])


    print('Reading bag file '+ os.getcwd() + '/' + args.content_image
          + ' topic ' + args.topic)
    readbag = rosbag.Bag(args.content_image,'r')
    bridge = CvBridge()
    cv_img = []

    with rosbag.Bag(args.output_image,'w') as outbag:
        for topic, msg, dtime in readbag.read_messages():

            if topic == args.topic:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                (h, w, d) = img.shape

                nh = h/args.content_scale
                nw = w/args.content_scale
                cv_img = cv2.resize((h, w), (int(nh),int(nw)))

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
                        output = torch.onnx._export(style_model, content_image, args.export_onnx).cpu()
                    else:
                        print('stylizing image ...')
                        output = style_model(cv_img).cpu()
            else:
                outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)

    readbag.close()





def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=False,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--topic", type=str, required=False,
                                 help="path to directory with images you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=False,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--output-dir", type=str, required=False,
                                 help="path for saving the output images")
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
