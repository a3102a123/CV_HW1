import sys
import os
import open3d as o3d

def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

if len(sys.argv) != 2:
    print("\nPlease enter a folder path (relative path)!\n")
    print("Usage : python check.py [folder path]")
    sys.exit()

file_names = ["bunny.ply" , "star.ply" , "venus.ply" ]
folder_path = sys.argv[1]
for file in file_names:
    path = os.path.join(folder_path,file)
    print(path)
    show_ply(path)
