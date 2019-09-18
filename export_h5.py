import open3d
import os
import numpy as np
from PIL import Image
import glob
import cv2
import argparse
import json
import h5py
from src.kitti_base import PointCloud_Vis, Semantic_KITTI_Utils
from vis_velo import init_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser('vis_velo.py')
    parser.add_argument('--cfg', default = 'config/ego_view.json', type=str)
    parser.add_argument('--root', default = os.environ.get('KITTI_ROOT','~/dataset/KITTI/'), type=str)
    parser.add_argument('--voxel', default=0.1, type=float, help='voxel size for down sampleing')

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
    handle.set_part(part = '00')
    handle.set_filter(h_fov, v_fov, x_range, y_range, z_range, d_range)

    fp = h5py.File('tmp/sem_kitti.h5',"a")

    for part in ['00','01','02','03','04','05','06','07','08','09','10']:
        args.part = part
        handle.set_part(part)

        for index in range(0,handle.get_max_index()):
            key_pts = '%s/%06d/pts'%(args.part, handle.index)
            key_sem = '%s/%06d/sem'%(args.part, handle.index)
            if key_pts in fp.keys() or key_sem in fp.keys():
                print('skip', key_pts, key_sem)
                continue
                
            handle.load(index)

            # Downsample the point cloud and semantic labels as the same time
            pcd, sem_label = handle.extract_points(voxel_size = args.voxel)
            pts_3d = np.asarray(pcd.points).astype(np.float32)

            # Project in view 3D points to 2D image using RT matrix
            # Filter out the points that are behind us and keeping the labels consistent
            pts_2d, color, sem_label = handle.project_3d_to_2d(pcd, sem_label)

            # Map the Semantic KITTI labels to KITTI original 19 classes labels
            sem_label_learn_mapping = handle.learning_mapping(sem_label)

            assert pts_3d.shape[0] == sem_label_learn_mapping.shape[0]
            print(key_pts, key_sem, pts_3d.shape[0])

            fp[key_pts] = np.asarray(pcd.points)
            fp[key_sem] = sem_label

    fp.close()