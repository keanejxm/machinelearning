#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  读取与现实.py
:time  2024/1/15 10:42
:desc  
"""
from common_utils import *

from PIL import Image

image_path = f"{DATA_PATH}/data_test/lslm/63.jpg"
img = Image.open(image_path)
img.show()
