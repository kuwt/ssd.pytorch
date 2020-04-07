import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from matplotlib import pyplot as plt
import argparse
import pprint
import yaml

from ssd import build_ssd

######## ###################
# 
# train
# 
#  ############################
parser = argparse.ArgumentParser( description='Single Shot MultiBox Detector Training With Pytorch')

train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--imgPath', default='./data/dog.jpg', type=str)
parser.add_argument('--checkpt', default='./weights/ssd300_mAP_77.43_v2.pth', type=str)
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to test model')
parser.add_argument('--save_result_path', type=str,default='./result.png')
args = parser.parse_args()

######## ###################
# 
# demo
# 
#  ############################
def demo(cfg, imgPath, checkpt):
    if args.cuda:
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

    ##### Build SSD300 in Test Phase #####
    with torch.no_grad():
        net = build_ssd(cfg, 'test', cfg['min_dim'], cfg['num_classes'])    # initialize SSD
        net.load_weights(checkpt)
        net.eval()

    ##### Load image #####
    image = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform
    plt.figure(figsize=(10,10))
    plt.imshow(rgb_image)
    plt.show()

    ##### Pre-process the input #####
    x = cv2.resize(image, (cfg['min_dim'], cfg['min_dim'])).astype(np.float32)

    PREPROCESS_MEAN = cfg['PREPROCESS_MEAN']
    x -= PREPROCESS_MEAN   #subtract mean of the dataset as in the training phase
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    plt.imshow(x)
    plt.show()
    x = torch.from_numpy(x).permute(2, 0, 1)

    ##### SSD Forward Pass #####
    xx = Variable(x.unsqueeze(0))     # add dimension to x to match network input, wrap tensor in Variable
    
    if args.cuda:
        if torch.cuda.is_available():
            xx = xx.cuda()
    y = net(xx)
    print('y = ', y.shape)
    ##### Parse the Detections and View Results #####
    detections = y.data
    print('detections = ', detections.shape)

    plt.figure(figsize=(10,10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):     # i is the class label
        j = 0
        score = detections[0,i,j,0]         #detections is of size 1 x classNumber x topk x 5,  5 corresponding to score, bbox pt coordinate
        while detections[0,i,j,0] >= 0.6:   # for each class, print the topk detection until the score drops below 0.6
            score = detections[0,i,j,0]
            display_txt = 'c:{},s:{}'.format(i,score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j+=1

    plt.show()
    plt.savefig(args.save_result_path)

######## ###################
# 
# main
# 
#  ############################
if __name__ == '__main__':

     with open("./config_voc.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        print('\n=========\nconfig \n==========\n')
        pprint.pprint(cfg)
        demo( cfg, args.imgPath, args.checkpt)