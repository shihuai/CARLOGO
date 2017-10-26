# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: py_crop_imgs.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年07月09日 星期日 16时57分19秒
#########################################################################

import os
import numpy as np
import cv2

def load_imgs_coords():
    fr = open('train_coords.txt', 'r')
    rows = fr.readlines()
    fr.close()

    names = []
    for row in rows:
        row = row.strip('\n').split(' ')
        names.append(row[0])
        img = cv2.imread('test/' + row[0])
        #cv2.rectangle(img, (int(row[1]), int(row[2])), (int(row[3]), int(row[4])), (0, 255, 0), 2)
        img = img[int(row[2]):int(row[4]), int(row[1]):int(row[3])]
        cv2.imwrite('head_test/' + row[0], img)

    return names

def remove_no_coords_img(names):
    img_list = os.listdir('test/')

    excluded_imgs = []
    for name in img_list:
        if name not in names:
            excluded_imgs.append(name)

    if not os.path.exists('excluded_imgs/'):
        os.mkdir('excluded_imgs')

    for name in excluded_imgs:
        img = cv2.imread('test/' + name)
        cv2.imwrite('excluded_imgs/' + name, img)

if __name__ == '__main__':

    names = load_imgs_coords()
    remove_no_coords_img(names)


