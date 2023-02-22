import PIL.ImageShow
from PIL import Image
import numpy as np
import os
import cv2



def readParam(path):
    return np.loadtxt(path, dtype=np.float32, delimiter=' ')

def readData(imagePath):
    image=[]
    imageList = os.listdir(imagePath)
    imageList.sort()
    for item in imageList:
        if item.endswith('.jpg'):
            item = imagePath + '/' + item
            image.append(cv2.imread(item))
    K=readParam("../data/K.txt")
    D=readParam("../data/D.txt")
    return image,K,D

def buildMatrix(pose):
    w=np.array(pose[:3])
    t=np.array(pose[3:])
    k=w/np.linalg.norm(w)
    theta = np.linalg.norm(w)
    kx=np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    R=np.eye(3) + np.sin(theta)*kx + (1-np.cos(theta))*kx@kx

    outerParam=np.eye(4)
    outerParam[:3,:3]=R
    outerParam[:3,3]=t.T
    return outerParam

def cornersGen():
    x=9
    y=6
    X,Y=np.meshgrid(np.arange(x),np.arange(y))
    corners=(np.stack([X,Y],axis=-1)*0.04).reshape([x*y,2])
    corners=np.concatenate([corners, np.zeros([x*y, 1])], axis=-1)
    corners = np.concatenate([corners, np.ones([54, 1])], axis=-1)
    return corners



def ProjectPoints(points,K,RT):
    res=(np.matmul(RT,points.T))[:3,:]
    res=(K@res).T
    z = res[:,2]
    x=res[:,0]/z
    y=res[:,1]/z
    return x,y
