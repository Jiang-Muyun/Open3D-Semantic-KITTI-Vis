import open3d
import os
import numpy as np
from PIL import Image
import glob
import cv2
import argparse
import json

from src.kitti_base import PointCloud_Vis, Semantic_KITTI_Utils

def init_params():
    parser = argparse.ArgumentParser('vis_velo.py')
    parser.add_argument('--cfg', default = 'config/ego_view.json', type=str)
    parser.add_argument('--root', default = os.environ.get('KITTI_ROOT','~/dataset/KITTI/'), type=str)
    parser.add_argument('--part', default = '00', type=str , help='KITTI part number')
    parser.add_argument('--index', default = 0, type=int, help='start index')
    parser.add_argument('--voxel', default = 0.1, type=float, help='voxel size for down sampleing')
    parser.add_argument('--modify', action = 'store_true', default = False, help='modify an existing view')
    args = parser.parse_args()

    assert os.path.exists(args.root),'Root directory does not exist '+ args.root
    assert os.path.exists(args.cfg), 'Config file does not exist '+ args.cfg

    cfg_data = json.load(open(args.cfg))
    h_fov = cfg_data['h_fov']
    v_fov = cfg_data['v_fov']
    x_range = cfg_data['x_range']
    y_range = cfg_data['y_range']
    z_range = cfg_data['z_range']
    d_range = cfg_data['d_range']

    handle = Semantic_KITTI_Utils(root = args.root)
    handle.set_part(part = args.part)
    handle.set_filter(h_fov, v_fov, x_range, y_range, z_range, d_range)
    vis_handle = PointCloud_Vis(args.cfg, new_config = args.modify)

    return args, handle, vis_handle

if __name__ == "__main__":
    args, handle, vis_handle = init_params()
    cv2.namedWindow('depth');cv2.moveWindow("depth", 600,000)
    cv2.namedWindow('learn_mapping');cv2.moveWindow("learn_mapping", 400,200)
    cv2.namedWindow('semantic');cv2.moveWindow("semantic", 200,400)

    for index in range(args.index, handle.get_max_index()):
        # Load image velodyne points and semantic labels
        handle.load(index)

        # Downsample the point cloud and semantic labels as the same time
        pcd,sem_label = handle.extract_points(voxel_size = args.voxel)
        pts_3d = np.asarray(pcd.points).astype(np.float32)

        print(index,'/',handle.get_max_index(), 'n_pts',pts_3d.shape[0])

        # Project in view 3D points to 2D image using RT matrix
        # Filter out the points that are behind us and keeping the labels consistent
        pts_2d, color, sem_label = handle.project_3d_to_2d(pcd, sem_label)

        # Map the Semantic KITTI labels to KITTI original 19 classes labels
        sem_label_learn_mapping = handle.learning_mapping(sem_label)

        # Showing the point cloud depth
        img_depth = handle.draw_2d_points(pts_2d, color)

        # Showing the Semantic KITTI cloud cloud labels
        img_semantic = handle.draw_2d_sem_points(pts_2d, sem_label)

        # Showing the labels mapped to kitti original 19 classes labels
        img_learn_mapping = handle.draw_2d_sem_points_with_learning_mapping(pts_2d, sem_label_learn_mapping)

        # Update the display
        vis_handle.update(pcd)
        cv2.imshow('depth', img_depth)
        cv2.imshow('semantic', img_semantic)
        cv2.imshow('learn_mapping', img_learn_mapping)

        # Saving the frames
        # vis_handle.capture_screen('tmp/top_view.jpg')
        # cv2.imwrite('tmp/frame_depth.jpg',frame_depth)
        # cv2.imwrite('tmp/frame_semantic.jpg',frame_semantic)
        # cv2.imwrite('tmp/learn_mapping.jpg',learn_mapping)

        if 32 == cv2.waitKey(1):
            break
