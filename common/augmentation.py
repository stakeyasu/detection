
# coding: utf-8

import sys
import os
import numpy as np
import cv2
import random

def flip(train_X):

    _train_X=[]
    for img in train_X:
 
      if random.randint(0,1) == 0:
         flip_img=cv2.flip(img,1)
         _train_X.append(flip_img) 
      else :
         _train_X.append(img)

    return _train_X


