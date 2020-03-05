#encoding:utf-8
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

EPOCHS = 100

def display(fname):
    img = Image.open(fname)
    img.show()

def result(ep):
    with open('save_gimage.pkl', 'rb') as f:
        save_gimage = pkl.load(f)
        fig, axes = plt.subplots(5,5,figsize=(28, 28))
        for img, ax in zip(save_gimage[ep], axes.flatten()):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(img.reshape((28,28)), cmap='gray')
        if ep == -1:
            ep = EPOCHS - 1
