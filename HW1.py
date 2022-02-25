import os
import cv2
from cv2 import threshold
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

def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][image_col - 1 - j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

def read_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

def print_depth(Z):
    Z_map = np.reshape(Z,(image_row,image_col)).copy()
    for i in range(image_row):
        for j in range(image_col):
            if(Z_map[i][j] != 0):
                print(i," ",j," : ",Z_map[i][j])

def surface_reconstruction_integral(N):
    N = np.reshape(N,(image_row , image_col,3))
    z_approx1 = np.zeros((image_row , image_col))
    z_approx2 = np.zeros((image_row , image_col))
    z_approx3 = np.zeros((image_row , image_col))
    z_approx4 = np.zeros((image_row , image_col))
    z = np.zeros((image_row , image_col))
    for i in range(1,image_row):
        for j in range(image_col):
            n1 = N[i][j][0]
            n2 = N[i][j][1]
            n3 = N[i][j][2]
            if n3 != 0:
                z_approx1[i][j] = n2/n3 + z_approx1[i - 1][j]
    for i in range(image_row -2,-1,-1):
        for j in range(image_col):
            n1 = N[i][j][0]
            n2 = N[i][j][1]
            n3 = N[i][j][2]
            if n3 != 0:
                z_approx2[i][j] = -n2/n3 + z_approx2[i+1][j]

    for j in range(1,image_col):
        for i in range(image_row):
            n1 = N[i][j][0]
            n2 = N[i][j][1]
            n3 = N[i][j][2]
            if n3 != 0:
                z_approx3[i][j] = -n1/n3 + z_approx3[i][j - 1]
    for j in range(image_col - 2,-1,-1):
        for i in range(image_row):
            n1 = N[i][j][0]
            n2 = N[i][j][1]
            n3 = N[i][j][2]
            if n3 != 0:
                z_approx4[i][j] = n1/n3 + z_approx4[i][j + 1]
    
    for i in range(0,image_row):
        for j in range(0,image_col):
            z1 = z_approx1[i][j]
            z2 = z_approx2[i][j]
            z3 = z_approx3[i][j]
            z4 = z_approx4[i][j]
            z[i][j] = (z1*(image_row - i) + z2*i + z3*(image_col - j) + z4*j) / (image_row + image_col) 

    return z

def surface_reconstruction_matrix(N):
    N = np.reshape(N,(image_row , image_col,3))
    num_pix = image_row * image_col
    M = scipy.sparse.lil_matrix((2*num_pix, num_pix))
    v = np.zeros((2*num_pix, 1))
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            x = N[i][j][0]
            y = N[i][j][1]
            z = N[i][j][2]
            if z != 0:
                v[idx] = -x/z
                v[idx + num_pix] = -y/z
            if j != (image_col - 1) :
                M[idx , idx] =  -1
                M[idx , idx+1] = 1
            if i != (image_col - 1):
                M[idx + num_pix , idx] = -1
                M[idx + num_pix , idx + image_col] = 1
    
    MtM = M.T @ M
    Mtv = M.T @ v
    Z, info = scipy.sparse.linalg.cg(MtM, Mtv)
    return Z

# read text file of light source
f = open(os.path.join("test",file_name,"LightSource.txt"))
for line in f:
    light_vector = np.zeros(3)
    # parsing 
    print(line.split(" ")[0])
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
        image_row,image_col = temp.shape
    temp = temp.flatten()
    # reshape to 1 * (w*h) matrix
    image_arr.append(temp)

# convert to numpy array
light_vector_arr = np.asarray(light_vector_arr)
image_arr = np.asarray(image_arr)

print("Image size : ", image_row , " , " , image_col)
print("--------------------------")
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
print(Kdn[int(image_row / 2 * image_col + image_col /2)]," ---> ",n[int(image_row / 2 * image_col + image_col /2)])
# n = Kdn
### calculate the normal by numpy
# for i in range(image_row*image_col):
#     v = Kdn[i]
#     norm = np.linalg.norm(v)
#     if norm != 0:
#         n[i] = v / norm

# Surface Reconstruction

### calc depth Z value : Mz = v
n = np.reshape(n, (image_row, image_col, 3))
# create mask
mask = np.zeros((image_row,image_col))
for i in range(image_row):
    for j in range(image_col):
        if n[i][j][2] != 0:
            mask[i][j] = 1
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
        M[row_idx, idx] = 1
        M[row_idx, idx_horizon] = -1
        v[row_idx] = -nx / nz

    row_idx = idx * 2 + 1
    if mask[h + 1][w]:
        idx_vert = full2obj[h + 1][w]
        M[row_idx, idx] = 1
        M[row_idx, idx_vert] = -1
        v[row_idx] = -ny / nz
    elif mask[h - 1][w]:
        idx_vert = full2obj[h - 1][w]
        M[row_idx, idx] = -1
        M[row_idx, idx_vert] = 1
        v[row_idx] = -ny / nz


MtM = M.T @ M
Mtv = M.T @ v
z = scipy.sparse.linalg.spsolve(MtM, Mtv)

std_z = np.std(z, ddof=1)
mean_z = np.mean(z)
z_zscore = (z - mean_z) / std_z

# 因奇异值造成的异常
outlier_ind = np.abs(z_zscore) > 10
z_min = np.min(z[~outlier_ind])
z_max = np.max(z[~outlier_ind])

Z = np.zeros((image_row,image_col))
for idx in range(num_pix):
    h = obj_h[idx]
    w = obj_w[idx]
    # Z[h, w] = (z[idx] - z_min) / (z_max - z_min) * 255
    Z[h,w] = z[idx]

# other way try to solve the skewness of depth map
# Z = surface_reconstruction_matrix(n)
# Z = surface_reconstruction_integral(n)

# visualizing corresponding parameter
depth_visualization(Z)
normal_visualization(n)
mask_visualization(mask)
plt.show()
save_ply(Z,"./temp.ply")
read_ply("./temp.ply")