'''
 * The Recognize Anything Model (RAM)
 * Written by Xinyu Huang
'''
import argparse
import numpy as np
import random
import time

import torch

from PIL import Image
from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='RAM inference for tagging')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/1641173_2291260800.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 384)')


if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=args.image_size)

    #######load model
    model = ram(pretrained=args.pretrained,
                             image_size=args.image_size,
                             vit='swin_l')
    model.eval()

    model = model.to(device)

    image = transform(Image.open(args.image)).unsqueeze(0).to(device)

    print('image shape: ', image.shape)
    plt.imshow(image.squeeze().permute(1,2,0).cpu().numpy())
    plt.show()

    start_inference_time = time.time()
    res = inference(image, model)
    print('Inference time: ', time.time() - start_inference_time)

    print("Image Tags: ", res[0])
    print("Confidence: ", " | ".join(["{:.3f}".format(conf) for conf in res[1]]))
