#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:36:02 2021

@author: endocv2021@generalizationChallenge
"""

import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import skimage
from skimage import io
from tifffile import imsave
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config
from mask_former import add_mask_former_config

def create_predFolder(task_type):
    directoryName = 'EndoCV2022'
    if not os.path.exists(directoryName):
        os.mkdir(directoryName)
        
    if not os.path.exists(os.path.join(directoryName, task_type)):
        os.mkdir(os.path.join(directoryName, task_type))
        
    return os.path.join(directoryName, task_type)

def detect_imgs(infolder, ext='.tif'):
    import os

    items = os.listdir(infolder)

    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)

def mymodel(cfg_file_path, weights_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(cfg_file_path)
    cfg.MODEL.WEIGHTS = os.path.join(weights_path)
    model = DefaultPredictor(cfg)
    return model, cfg.MODEL.DEVICE
    


        
if __name__ == '__main__':
    '''
     You are not allowed to print the images or visualizing the test data according to the rule. 
     We expect all the users to abide by this rule and help us have a fair challenge "EndoCV2021-Generalizability challenge"
     
     FAQs:
         1) Most of my predictions do not have polyp.
            --> This can be the case as this is a generalisation challenge. The dataset is very different and can produce such results. In general, not all samples 
            have polyp.
        2) What format should I save the predictions.
            --> you can save it in the tif or jpg format. 
        3) Can I visualize the data or copy them in my local computer to see?
            --> No, you are not allowed to do this. This is against challenge rules. No test data can be copied or visualised to get insight. Please treat this as unseen image.!!!
        4) Can I use my own test code?
            --> Yes, but please make sure that you follow the rules. Any visulization or copy of test data is against the challenge rules. We make sure that the 
            competition is fair and results are replicative.
    '''
    model, device = mymodel('/home/mariiak/MaskFormer/output/config.yaml', os.path.join('/home/mariiak/MaskFormer/output', "model_final.pth"))
    task_type = 'PolypGen2.0'

    directoryName = create_predFolder(task_type)    
    root_dir = '/datasets/EndoCV2022_ChallengeDataset/PolypGen2.0'
    subDirs = ['seq1', 'seq10_endocv22']
    
    for j in range(0, len(subDirs)):
        seq_folder = os.path.join(root_dir, subDirs[j])
        imgfolder = os.path.join(seq_folder, 'images')
        
        # set folder to save your checkpoints here!
        saveDir = os.path.join(directoryName , subDirs[j]+'_pred')
    
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)

        imgfiles = os.listdir(imgfolder)
        imgfiles = list(map(lambda x: os.path.join(imgfolder, x), imgfiles))
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        file = open(saveDir + '/'+"timeElaspsed" + subDirs[j] +'.txt', mode='w')
        timeappend = []
    
        for imagePath in imgfiles:
            """plt.imshow(img1[:,:,(2,1,0)])
            Grab the name of the file. 
            """
            filename = (imagePath.split('/')[-1]).split('.jpg')[0]
            print('filename is printing::=====>>', filename)
            images = cv2.imread(imagePath)
            
            #            
            img = skimage.io.imread(imagePath)
            size=img.shape
            start.record()
            #
            outputs = model(images)
            #
            end.record()
            torch.cuda.synchronize()
            print(start.elapsed_time(end))
            timeappend.append(start.elapsed_time(end))

            preds = outputs["sem_seg"][0].detach().unsqueeze(0).cpu()
            preds = (preds.sigmoid() > 0.5).numpy().astype(np.uint8)
            preds = (preds[0] * 255.0).astype(np.uint8)
            imsave(saveDir +'/'+ filename +'_mask.jpg', preds)
            
            file.write('%s -----> %s \n' % 
               (filename, start.elapsed_time(end)))

        file.write('%s -----> %s \n' % 
           ('average_t', np.mean(timeappend)))
