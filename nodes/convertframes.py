# python3
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

def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    content_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.mul(255))
                        ])

    if args.content_image and not args.input_dir:
        print('converting image file '+ args.content_image)
        content_image = utils.load_image(args.content_image, scale=args.content_scale)
        content_image = content_transform(content_image)
        content_image = content_image.unsqueeze(0).to(device)

        if args.model.endswith(".onnx"):
            output = stylize_onnx_caffe2(content_image, args)
        else:
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
                    output = style_model(content_image).cpu()
        utils.save_image(args.output_image, output[0])


    elif not args.content_image and args.input_dir:
        print('converting files from directory /' + args.input_dir)
        imageset  = utils.load_dir(args.input_dir, scale=args.content_scale)
        content_image,image_names = zip(*imageset)

        content_image = [content_transform(img) for img in content_image]
        content_image = [img.unsqueeze(0).to(device) for img in content_image]

        output  = []

        if args.model.endswith(".onnx"):
            output = stylize_onnx_caffe2(content_image, args)
        else:
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
                    output = [style_model(img).cpu() for img in content_image]

        output_vals = zip(output, image_names)
        [utils.save_image(args.output_dir+name,img[0]) for img,name in output_vals]
        print ('saving images into directory ' + args.output_dir)
    else:
        print('ERROR: set directory or image to convert')




def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=False,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--input-dir", type=str, required=False,
                                 help="path to directory with images you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=False,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--output-dir", type=str, required=False,
                                 help="path for saving the output images")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
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
