import os
import cv2
import numpy as np
import re
import open3d as o3d
from sklearn.preprocessing import normalize
import scipy

file_name = "star"
image_arr = []
light_vector_arr = []
image_row = 0 
image_col = 0

# input the array shape as (w*h) * 3 , every row is a normal vetor of one pixel
def normal_visualization(N):
    N = np.reshape(N, (image_row, image_col, 3))
    N[:,:,0], N[:,:,2] = N[:,:,2], N[:,:,0].copy()
    N = (N + 1.0) / 2.0
    cv2.imshow('normal map', N)
    cv2.waitKey()
    cv2.destroyAllWindows()

def depth_visualization(D):
    D = np.reshape(D, (image_row,image_col))
    D = np.uint8(D)
    cv2.imshow('depth map', D)
    cv2.waitKey()
    cv2.destroyAllWindows()

# read text file of light source
f = open(os.path.join("test",file_name,"LightSource.txt"))
for line in f:
    light_vector = np.zeros(3)
    # parsing 
    light_str = line.split(" ")[1]
    light_str = re.split(',|\(|\)|\\n',light_str)
    light_str = list(filter(None, light_str))
    for i,n in enumerate(light_str):
        light_vector[i] = float(n)
    # normalize
    light_vector = light_vector / np.linalg.norm(light_vector)
    light_vector_arr.append(light_vector)
    print(light_vector)

# read .bmp image
for i in range(1,7):
    image_name = "pic" + str(i) + ".bmp"
    image_path = os.path.join("test",file_name,image_name)
    print(image_path)
    temp = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    image_row,image_col = temp.shape
    temp = temp.flatten()
    # reshape to 1 * (w*h) matrix
    image_arr.append(temp)

# convert to numpy array
light_vector_arr = np.asarray(light_vector_arr)
image_arr = np.asarray(image_arr)

print("array shape")
print("--------------------------")
print("Light vector : ",light_vector_arr.shape)
print("Image array : ",image_arr.shape)

# normal estimation
Kdn = np.linalg.inv(light_vector_arr.T @ light_vector_arr) @ light_vector_arr.T @ image_arr
Kdn = Kdn.T
print("Kdn array : ",Kdn.shape)
# n is unit vector let P is the Reflection coefficient , 
# Kdn = Pn = 2-norm(n) * n -> n = Kdn / 2-norm(n)
n = np.zeros_like(Kdn)
n = normalize(Kdn, axis=1)
### calculate the normal by numpy
# for i in range(image_row*image_col):
#     v = Kdn[i]
#     norm = np.linalg.norm(v)
#     if norm != 0:
#         n[i] = v / norm
normal_visualization(n)

# Surface Reconstruction
### calc depth Z value : Mz = v 
image_size = image_row * image_col
M = scipy.sparse.lil_matrix((image_size * 2, image_size))
v = np.zeros(image_size * 2,dtype=np.float32)
z_approx = np.zeros((image_row,image_col),dtype=float)
mask = np.ones((image_row,image_col))
nx = n[:,0]
ny = n[:,1]
nz = n[:,2]
### fill v
v[0:nx.shape[0]] = -nx/(nz+1e-8)
v[nx.shape[0]:v.shape[0]] = -ny/(nz+1e-8)
### fill M
offset = image_size
for i in range(image_row):
    for j in range(image_col):
        idx = i * image_col + j
        if j != image_col-1:
            M[idx, idx] = -1
            M[idx, idx+1] = 1
        if i != image_row-1:
            M[idx + offset, idx] = -1
            M[idx + offset, idx + image_col] = 1

MtM = M.transpose().dot(M)
Mtv = M.transpose().dot(v)
z, info = scipy.sparse.linalg.cg(MtM, Mtv)
depth_visualization(z)

# output to ply file
pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(Kdn)
# o3d.io.write_point_cloud("./temp.ply", pcd)