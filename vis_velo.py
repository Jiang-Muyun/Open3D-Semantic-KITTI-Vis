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
    parser.add_argument('--root', default='/media/james/MyPassport/James/dataset/KITTI/odometry/dataset/', type=str)
    parser.add_argument('--part', default='00',type=str)
    parser.add_argument('--modify', action='store_true', default = False)
    args = parser.parse_args()

    assert os.path.exists(args.root),'Root directory does not exist '+ args.root
    handle = Semantic_KITTI_Utils(os.path.join(args.root, 'sequences/%s/'%(args.part)))

    if not os.path.exists(args.cfg):
        args.modify = True
        h_fov = [-40, 40]
        v_fov = [-25, 2.0]
        x_range,y_range, z_range, d_range = None, None, None, None
    else:
        cfg_data = json.load(open(args.cfg))
        h_fov = cfg_data['h_fov']
        v_fov = cfg_data['v_fov']
        x_range = cfg_data['x_range']
        y_range = cfg_data['y_range']
        z_range = cfg_data['z_range']
        d_range = cfg_data['d_range']

    handle.set_filter(h_fov, v_fov, x_range, y_range, z_range, d_range)
    vis_handle = PointCloud_Vis(args.cfg, new_config = args.modify)

    return handle, vis_handle

if __name__ == "__main__":
    handle, vis_handle = init_params()
    handle.index = 70
    while handle.next():
        pcd,sem_label = handle.extract_points(voxel_size = 0.1)

        vis_handle.update(pcd)
        print('n_pts',np.asarray(pcd.points).shape[0], sem_label.shape)

        pts_2d, color, sem_label = handle.project_3d_to_2d(pcd, sem_label)
        frame_depth = handle.draw_2d_points(pts_2d, color)
        frame_semantic = handle.draw_2d_sem_points(pts_2d, sem_label)

        # vis_handle.capture_screen('tmp/top_view.jpg')
        # cv2.imwrite('tmp/frame_depth.jpg',frame_depth)
        # cv2.imwrite('tmp/frame_semantic.jpg',frame_semantic)

        cv2.imshow('depth', frame_depth)
        cv2.imshow('semantic', frame_semantic)
        if 32 == cv2.waitKey(1):
            break
