import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

THE_CLASSES = ('box', )
DATA_ROOT = '/srv/data/kuwingto/SAEcustomData20200224/obj1_bbox'

class CustomAnnotationTransform(object):

    def __init__(self):
        self.class_to_ind = dict(zip(THE_CLASSES, range(len(THE_CLASSES))))
    
    def __call__(self, target, width, height):

        res = []
        for obj in target['shapes']:
            name = obj['label']
            x = [obj['points'][0][0] / width, obj['points'][1][0] / width]
            y = [obj['points'][0][1] / height, obj['points'][1][1] / height]

            bndbox = []
            bndbox.append(min(x))
            bndbox.append(min(y))
            bndbox.append(max(x))
            bndbox.append(max(y))

            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


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
                transform=None,
                target_transform=CustomAnnotationTransform(),
                dataset_name='custom'):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.input_res = cfg['min_dim']
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
        
        annofile = open(anno_path, 'r')
        target = json.load(annofile)
        
        img = cv2.imread(img_path)


        height, width, channels = img.shape

        if self.target_transform is not None: #get normalized annotation
            target = self.target_transform(target, width, height)

        if self.transform is not None:  #augmentation
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        else:
            img = img.astype('float32') 
            img =  cv2.resize(img, (self.input_res, self.input_res), interpolation = cv2.INTER_AREA)

        #print("b = ", target)
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width



    