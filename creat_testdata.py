# coding: utf-8
# 生成测试数据集， 用于验证model和train
# creat testdata to check train.ipynb
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import os

root = 'test_img/'
if not os.path.exists(root):
    os.mkdir(root)
for i in range(2,5):
    if not os.path.exists(root+'{}/'.format(i)):
        os.mkdir(root+'{}/'.format(i))

CHARS = list('abcdefghijklmnopqrstuvwxyz0123456789')
def make_data(root='test_img/'):
    """生成不同长度的文本"""
    global CHARS
    length = random.randint(2,4)
    strs = ''.join(random.sample(CHARS, length))
    img = np.zeros([35, 90])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, strs, (5, 28), font, 1, (255, 255, 255), 1)
    cv2.imwrite(root+"{}/{}_{}.png".format(length, strs, random.randint(0,10)), img)

for i in range(64):
    make_data()

