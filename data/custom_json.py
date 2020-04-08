import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

THE_CLASSES = ('box', )
DATA_ROOT = '/srv/data/kuwingto/SAEcustomData20200224/obj1_bbox'

def transverseImageAnnoDirectory(root, img_ext = ".bmp", anno_ext = ".json"):
    """
    transverse the root directory and return the image paths and annotation as lists.
    """
    image_paths = []
    annotations_paths = []
    
    ### Get all annotations file names  ###
    anno_file_names = []
    for root, dirs, files in os.walk(root):
        for f in files:
            filename, extension = os.path.splitext(f)
            if extension == anno_ext: # found annotation
                anno_file_names.append(filename)
    
    ### For each anno file name, see if the corresonding image is there, if yes, read the anno and image into the list  ###
    for name in anno_file_names:
        image_path = root + "/" + name + img_ext
        if os.path.isfile(image_path):# found image of an annotation
             ### ready image path  ###
            image_paths.append(image_path)

            ### ready annotation path###
            anno_file_path = root + "/" + name + anno_ext
            annotations_paths.append(anno_file_path)
    return image_paths, annotations_paths

class CustomDetection(data.Dataset):
    def __init__(self, 
                cfg,
                root = DATA_ROOT,
                dataset_name='custom'):

        self.root = root
        self.name = dataset_name
        self.input_res = cfg['min_dim']
        self.class_to_ind = dict(zip(THE_CLASSES, range(len(THE_CLASSES))))
        self._imgpaths, self._annopaths = transverseImageAnnoDirectory(self.root, ".bmp", ".json")
    
        print("show 5 samples")
        for i in range(5):
            print("imagePath = " ,self._imgpaths[i])
        for i in range(5):    
            print("annotation = " ,self._annopaths[i])

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self._imgpaths)

    def pull_item(self, index):
        img_path = self._imgpaths[index]
        anno_path = self._annopaths[index]
        #print("img_path = ", img_path)
        #print("anno_path = ", anno_path)
        
        ##### read files #####
        img = cv2.imread(img_path)
        height, width, channels = img.shape

        annofile = open(anno_path, 'r')
        jsonanno = json.load(annofile)

        ##### read files #####
        targets = [] 
        for obj in jsonanno['shapes']:
            name = obj['label']
            x = [obj['points'][0][0], obj['points'][1][0]]
            y = [obj['points'][0][1], obj['points'][1][1]]

            bndbox = []
            bndbox.append(min(x))
            bndbox.append(min(y))
            bndbox.append(max(x))
            bndbox.append(max(y))

            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)    # [xmin, ymin, xmax, ymax, label_ind]
            targets += [bndbox]  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

        ##### resize to map input resolution of the network#####
        img =  cv2.resize(img, (self.input_res, self.input_res), interpolation = cv2.INTER_AREA)
        
        factor_x = float(self.input_res) / float(width)
        factor_y = float(self.input_res) / float(height)
        for t in targets:
            t[0] = factor_x * t[0]
            t[2] = factor_x * t[2]
            t[1] = factor_y * t[1]
            t[3] = factor_y * t[3]
        
        ##### augmentation #####
        bbs = []
        for t in targets:
            bbs.append(BoundingBox(x1=t[0], y1=t[1], x2= t[2], y2=t[3]))
        bbs_oi = BoundingBoxesOnImage(bbs, shape=img.shape)

        seq = iaa.Sequential([
            iaa.Multiply((0.8, 1.1)), # change brightness, doesn't affect keypoints
            iaa.AdditiveGaussianNoise(0,5),
            iaa.Affine(
                scale=(0.9,1),
                rotate=(-5,5)
            )
        ])
        image_aug, bbs_aug = seq(image=img, bounding_boxes=bbs_oi)
        for i in range(len(bbs_oi.bounding_boxes)):
            after = bbs_aug.bounding_boxes[i]
            targets[i][0] =  after.x1
            targets[i][1] =  after.y1
            targets[i][2] =  after.x2
            targets[i][3] =  after.y2
        img = image_aug

        ##### treatment for SSD multibox #####
        for t in targets:
            t[0] = float(t[0]) / (self.input_res)
            t[1] = float(t[1]) / (self.input_res)
            t[2] = float(t[2]) / (self.input_res)
            t[3] = float(t[3]) / (self.input_res)
        img = img.astype('float32') 

        return torch.from_numpy(img).permute(2, 0, 1), targets, height, width



    