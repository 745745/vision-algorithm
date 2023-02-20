from utility import *

import matplotlib.pyplot as plt





def main():
    I,K,D=readData()
    poses=np.loadtxt("../data/poses.txt")
    RT=buildMatrix(poses[0])
    corners=cornersGen()
    x,y=ProjectPoints(corners,K,RT)
    plt.imshow(I)
    plt.plot(x,y,'r+')
    plt.show()





if __name__=="__main__":
    main()