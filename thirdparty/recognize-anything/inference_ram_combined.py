'''
 * The Recognize Anything Model (RAM) inference on seen AND unseen classes
'''
import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import ram
from ram import inference_ram_openset as inference
from ram import get_transform

from ram.utils import build_openset_label_embedding
from torch import nn

parser = argparse.ArgumentParser(
    description='RAM inference for tagging')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/openset_example.jpg')
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

    #######set openset interference
    openset_label_embedding, openset_categories = build_openset_label_embedding()

    model.tag_list = np.concatenate(
        (model.tag_list, np.array(openset_categories)))
    
    model.label_embed = nn.Parameter(torch.cat(
        (model.label_embed, openset_label_embedding.float())))

    model.num_class = len(model.tag_list)

    # the threshold for unseen categories is often lower
    openset_class_threshold = torch.ones(len(openset_categories)) * 0.5
    model.class_threshold = torch.cat(
        (model.class_threshold, openset_class_threshold))
    #######

    image = transform(Image.open(args.image)).unsqueeze(0).to(device)

    res = inference(image, model)
    print("Image Tags: ", res)
