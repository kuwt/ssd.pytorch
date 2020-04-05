import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from matplotlib import pyplot as plt
from ssd import build_ssd

#####
use_cuda = 0
import config
CFG = config.voc
PREPROCESS_MEAN = (104.0, 117.0, 123.0) 
NN_inputSize = 300

#####
def demo(cfg, imgPath = './data/dog.jpg' , checkpt = './weights/ssd300_mAP_77.43_v2.pth'):
    if use_cuda:
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

    ##### Build SSD300 in Test Phase #####
    with torch.no_grad():
        net = build_ssd(cfg, 'test', NN_inputSize, cfg['num_classes'])    # initialize SSD
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
    x = cv2.resize(image, (NN_inputSize, NN_inputSize)).astype(np.float32)
    x -= PREPROCESS_MEAN   #subtract mean of the dataset as in the training phase
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    plt.imshow(x)
    plt.show()
    x = torch.from_numpy(x).permute(2, 0, 1)

    ##### SSD Forward Pass #####
    xx = Variable(x.unsqueeze(0))     # add dimension to x to match network input, wrap tensor in Variable
    
    if use_cuda:
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
    for i in range(detections.size(1)):   # i is the class label
        j = 0
        score = detections[0,i,j,0]     #detections is of size 1 x classNumber x topk x 5,  5 corresponding to score, bbox pt coordinate
        while detections[0,i,j,0] >= 0.6:   # for each class, print the topk detection until the score drops below 0.6
            score = detections[0,i,j,0]
            display_txt = '{}'.format(score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j+=1

    plt.show()
    plt.savefig('result.png')

if __name__ == '__main__':
    demo(CFG)