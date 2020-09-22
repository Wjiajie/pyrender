import cv2
import numpy as np
def get_pose(pt13, pt2, K):
    """
    input:
        K: camera intrinsic [3x3]
        pt13: point in world cooedinate [Nx3x1]
        pt2: point in image coordinate [Nx2x1]
    return:
        camera_pose: camera_pose
    """
    #matx solve the different coordinate between opencv and opengl
    matx = np.array([[1, 0, 0,0 ],[0, -1, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]]) 
    _,rvec,tvec,inliners = cv2.solvePnPRansac(pt13, pt2, K, None)
    R = cv2.Rodrigues(rvec)[0]
    T = tvec
    R = R.transpose()
    T = -R.dot(T)
    camera_pose = np.eye(4)
    camera_pose[0:3,0:3] = R
    camera_pose[0:3,3:4] = T
    camera_pose = camera_pose.dot(matx)
    return camera_pose


def get2d23d(depth, camera_pose, K, pt1, pt2, znear, zfar, width=480, height=640):
    """
    input: 
        depth: image depth(in coordinate) 
        camera_pose: orign camera_pose
        K: camera intrinsic
        pt1: orign 2d Point
        pt2: matchinfg 2d point
        znear,zfar: depth clip range

    return:
        pt13: point in world coordinate[Nx3x1]
        pt2: part of input pt2
    """
    zn = znear
    zf = zfar
    h = height/2
    w = width/2
    pt13 = []
    pt2new = []
    for i in range(pt1.shape[0]):
        dx = int(pt1[i,1])
        dy = int(pt1[i,0])
        if depth[dx,dy] == 0:
          continue
        d = (depth[dx,dy]*(zn+zf)-2*zf*zn)/((zf-zn)*depth[dx,dy])
    # print(type(x))
        uvd = np.array([[(dy*1.0-w+0.5)/w],[-(dx*1.0-h+0.5)/h],[d], [1.0]])
        uvd = uvd*-depth[dx,dy]
        xyz = np.linalg.inv(K).dot(uvd)
        xyz = camera_pose.dot(xyz)
     # fobj.write('v'+' '+str(xyz[0,0]/xyz[3,0])+' '+str(xyz[1,0]/xyz[3,0])+' '+str(xyz[2,0]/xyz[3,0])+'\n')
        pt13.append([xyz[0,0]/xyz[3,0],xyz[1,0]/xyz[3,0],xyz[2,0]/xyz[3,0]])
        pt2new.append(pt2[i])
    pt13 = np.array(pt13)   
    pt2new = np.array(pt2new)
    return pt13, pt2new

    
