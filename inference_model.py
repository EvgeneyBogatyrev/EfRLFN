import os
import sys
import torch
import cv2
import argparse
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from code.model import EfRLFN


def init_network(args):
    network = EfRLFN(upscale=args.scale)
    network = network.cuda().float()
    network.eval()

    pretrained_path = args.weights
    ext = os.path.splitext(pretrained_path)[1]
    if ext == '.pt':
        network.load_state_dict(torch.load(pretrained_path), strict=True)
    elif ext == '.ckpt':
        checkpoint = torch.load(pretrained_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('network'):
                new_state_dict[key[len("network."):]] = state_dict[key]
        network.load_state_dict(new_state_dict, strict=True)
    return network


def upscale_image(network, image_path, out_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.PILToTensor()])
    image = transform(image).float()
    
    image /= 255.
    image = torch.tensor(image).unsqueeze(0).cuda().float()
    
    out = network(image).clamp(0, 1)

    out = out.permute(0, 2, 3, 1).squeeze(0)
    out *= 255
    out = out.detach().cpu().numpy().astype(np.uint8)
    out_image = Image.fromarray(out)

    out_image.save(out_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights',
                        type=str, help="Path to model weigths")   
    parser.add_argument('-s', '--scale',
                        type=int, help="Upscale param")   
    parser.add_argument('-i', "--input", 
                        type=str, help="Path to the input image")          
    parser.add_argument('-o', '--output',
                        type=str, help="Path to the output image")
    return parser.parse_args()  


if __name__ == '__main__':
    args = parse_args()
    
    model = init_network(args)
    upscale_image(model, args.input, args.output)
    print("Saved output to", args.output)