import math

import PIL.ImageShow
from PIL import Image
import numpy as np
import os
import cv2

from distortPoint import *

def undistortImage(image,k,center):
    image=np.dot(image[...,:3], [0.299, 0.587, 0.114]).T
    width,height=image.shape
    undistortImage=np.zeros([width,height])
    for i in range(width):
        for j in range(height):
            point=distortPoint(np.array((i,j)),k,center)
            x=math.floor(point[0])
            y=math.floor(point[1])

            if(  x>=0 and x<width and y>=0 and y<height ):
                undistortImage[i,j]=image[x,y]
    return undistortImage