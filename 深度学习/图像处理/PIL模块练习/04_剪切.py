#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  04_剪切.py
:time  2024/1/15 14:37
:desc  
"""
import numpy as np
from PIL import Image, ImageEnhance
from common_utils import *

from PIL import Image

image_path = f"{DATA_PATH}/data_test/lslm/63.jpg"
img = Image.open(image_path)
w, h = img.size
print(w,h)
box = (100,100,1000,1000)
img = img.crop(box)
img.show()

#x,y,w,h
crops = [(100, 100, w, h), (61, 869, 2429, 1074), (694, 52, 2181, 2033), (423, 24, 1438, 990), (1465, 25, 1231, 1385)]


crop = crops.pop(2)
print(crop)
# img = img.crop((crop[0],crop[1],crop[0]+crop[2],crop[1]+crop[3])).resize(img.size,Image.LANCZOS)
print((crop[0],crop[1],crop[0]+crop[2],crop[1]+crop[3]))
img = img.crop((crop[0],crop[1],crop[0]+crop[2],crop[1]+crop[3]))
img.show()