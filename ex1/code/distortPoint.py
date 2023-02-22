import numpy as np
def distortPoint(point,k,center):
    p=point-center
    r_s=np.linalg.norm(p) **2
    return (1+ k[0]*r_s + k[1] * r_s*r_s)*p +center
