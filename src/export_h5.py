import open3d
import os
import numpy as np
from PIL import Image
import glob
import cv2
import argparse
import json
import h5py
import sys
sys.path.append('.')
from src.kitti_base import PointCloud_Vis, Semantic_KITTI_Utils
from vis_velo import init_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser('vis_velo.py')
    parser.add_argument('--cfg', default = 'config/export_config.json', type=str)
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

    fp = h5py.File('tmp/pts_sem_voxel_%.2f.h5'%(args.voxel),"a")
    parts = ['00','01','02','03','04','05','06','07','08','09','10']

    try:
        for part in parts:
            args.part = part
            handle.set_part(part)

            for index in range(0,handle.get_max_index()):
                key = '%s/%06d'%(args.part, index)
                if key in fp.keys() :
                    print('skip', key)
                    continue
                handle.load(index)

                # Downsample the point cloud and semantic labels as the same time
                pcd, sem_label = handle.extract_points(voxel_size = args.voxel)
                pts_4d = handle.pts

                # Map the Semantic KITTI labels to KITTI 19 classes labels
                # Note: Here the 19 classs are different from the original KITTI 19 classes
                mapped_label = handle.learning_mapping(sem_label)
                mapped_label = mapped_label.reshape((-1,1)).astype(np.uint8)

                fp[key+'/pt'] = pts_4d
                fp[key+'/label'] = mapped_label
                print(index, key, pts_4d.shape)
    
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    fp.close()