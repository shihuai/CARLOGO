# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_search_file.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年10月12日 星期四 20时28分29秒
#########################################################################

import cv2
import os
import numpy as np

with open('filelist.txt', 'w') as fw:
    for file in os.listdir('data/train/'):
        img = cv2.imread('data/train/' + file)
        print img.shape
        width, height, ch = img.shape
        fw.write(file + ' ' + str(width) + ' ' + str(height) + '\n')
