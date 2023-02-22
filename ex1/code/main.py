from utility import *

from imageTovideo import *
from distortPoint import *
import matplotlib.pyplot as plt





def main():
    I,K,D=readData("../data/images")
    poses=np.loadtxt("../data/poses.txt")
    center = np.array((K[0, 2], K[1, 2]))
    for i in range(len(poses)):
        RT=buildMatrix(poses[i])
        corners=cornersGen()
        x,y=ProjectPoints(corners,K,RT)
        points=np.array((x,y)).T
        point=[distortPoint(q,D, center ) for q in points]
        x=[i[0] for i in point]
        y = [i[1] for i in point]
        plt.imshow(I[i])
        plt.plot(x,y,'r.')
        plt.savefig("./dotImage/img_{0:04d}.jpg".format(i))
        plt.clf()
        print(i)
    '''''
    makevideo("./dotImage")
    playvideo()
    '''''



if __name__=="__main__":
    main()