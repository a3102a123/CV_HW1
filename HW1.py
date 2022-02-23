import os
import cv2
import numpy as np
import re
import open3d as o3d
import open3d.core as o3c
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import scipy

file_name = "star"
image_arr = []
light_vector_arr = []
image_row = 0 
image_col = 0
mask = []

# input the array shape as (w*h) * 3 , every row is a normal vetor of one pixel
def normal_visualization(N):
    N = np.reshape(N, (image_row, image_col, 3))
    N[:,:,0], N[:,:,2] = N[:,:,2], N[:,:,0].copy()
    N = (N + 1.0) / 2.0
    cv2.imshow('Normal map', N)
    cv2.waitKey()
    cv2.destroyAllWindows()

def depth_visualization(D):
    D = np.reshape(D, (image_row,image_col))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()

def save_ply(Z):
    Z = np.reshape(Z, (image_row,image_col))
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = i
            data[idx][1] = j
            data[idx][2] = Z[i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud("./temp.ply", pcd,write_ascii=True)

def read_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

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
    if i == 1:
        mask = np.asarray(temp)
        print(mask)
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

# normal_visualization(n)

# Surface Reconstruction

### calc depth Z value : Mz = v
n = np.reshape(n, (image_row, image_col, 3))
obj_h, obj_w = np.where(mask != 0)
num_pix = np.size(obj_h)
print("Valid pixel: ", num_pix)
#fill valid index
full2obj = np.zeros((image_row, image_col))

for idx in range(np.size(obj_h)):
    full2obj[obj_h[idx], obj_w[idx]] = idx

M = scipy.sparse.lil_matrix((num_pix * 2, num_pix))
v = np.zeros(num_pix * 2, dtype=np.float32)
for idx in range(num_pix):
    h = obj_h[idx]
    w = obj_w[idx]
    nx = n[h][w][0]
    ny = n[h][w][1]
    nz = n[h][w][2]

    row_idx = idx * 2
    if mask[h][w + 1]:
        idx_horizon = full2obj[h][w + 1]
        M[row_idx, idx] = -1
        M[row_idx, idx_horizon] = 1
        v[row_idx] = -nx / nz
    elif mask[h][w - 1]:
        idx_horizon = full2obj[h][w - 1]
        M[row_idx, idx] = -1
        M[row_idx, idx_horizon] = 1
        v[row_idx] = -nx / nz

    row_idx = idx * 2 + 1
    if mask[h + 1][w]:
        idx_vert = full2obj[h + 1][w]
        M[row_idx, idx] = 1
        M[row_idx, idx_vert] = -1
        v[row_idx] = -ny / nz
    elif mask[h - 1][w]:
        idx_vert = full2obj[h - 1][w]
        M[row_idx, idx] = 1
        M[row_idx, idx_vert] = -1
        v[row_idx] = -ny / nz


MtM = M.T @ M
Mtv = M.T @ v
z = scipy.sparse.linalg.spsolve(MtM, Mtv)

'''std_z = np.std(z, ddof=1)
mean_z = np.mean(z)
z_score = (z - mean_z) / std_z'''

Z = mask.astype('float')
for idx in range(num_pix):
    h = obj_h[idx]
    w = obj_w[idx]
    #Z[h, w] = (z_score[idx] + 1) / 2
    Z[h,w] = z[idx]


# depth_visualization(Z)

save_ply(Z)
read_ply("./temp.ply")