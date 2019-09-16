import open3d
import json
import os
import numpy as np
from PIL import Image
import glob
import cv2
import shutil
import yaml

def print_projection_cv2(points, color, image):
    """ project converted velodyne points into camera image """
    assert points.shape[1] == 2, points.shape

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    points = points.astype(np.int32).tolist()
    color = color.astype(np.int32).tolist()

    for (x,y),c in zip(points,color):
        if x < 0 or y <0 or x >= hsv_image.shape[1] or y >= hsv_image.shape[0]:
            continue
        cv2.circle(hsv_image, (x, y), 2, (c, 255, 255), -1)
        # cv2.rectangle(hsv_image,(x-1,y-1),(x+1,y+1),(c,255,255),-1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def print_projection_plt(points, color, image):
    """ project converted velodyne points into camera image """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (int(points[0][i]), int(points[1][i])), 2, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def calib_velo2cam(fn_v2c):
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    for line in open(fn_v2c, "r"):
        (key, val) = line.split(':', 1)
        if key == 'R':
            R = np.fromstring(val, sep=' ')
            R = R.reshape(3, 3)
        if key == 'T':
            T = np.fromstring(val, sep=' ')
            T = T.reshape(3, 1)
    return R, T

def calib_cam2cam(fn_c2c, mode = '02'):
    """
    If your image is 'rectified image' :get only Projection(P : 3x4) matrix is enough
    but if your image is 'distorted image'(not rectified image) :
        you need undistortion step using distortion coefficients(5 : D)
    In this code, only P matrix info is used for rectified image
    """
    # with open(fn_c2c, "r") as f: c2c_file = f.readlines()
    for line in open(fn_c2c, "r"):
        (key, val) = line.split(':', 1)
        if key == ('P_rect_' + mode):
            P = np.fromstring(val, sep=' ')
            P = P.reshape(3, 4)
            P = P[:3, :3]  # erase 4th column ([0,0,0])
    return P

cfg_template = {
    "h_fov":[-40, 40],
    "v_fov":[-25, 2.0],
    "x_range": None,
    "y_range": None,
    "z_range": None,
    "d_range": None,
    "class_name": "PinholeCameraParameters",
    "version_major": 1,
    "version_minor": 0,
    "extrinsic": 
        [0.9957216461094273,0.0021923177689312026,0.09237747134411402,0.0,-0.09216825089269272,-0.04772462536762007,0.9945991019807439,0.0,
        0.006589157496541302,-0.9988581249989011,-0.04731838043683903,0.0,4.5214514284776115,2.791922931313028,112.96160194201362,1.0],
    "intrinsic": {
        "width": 800,
        "height": 800,
        "intrinsic_matrix": [692.820323027551,0.0,0.0,0.0,692.820323027551,0.0,599.5,399.5,1.0],
    },
}

class PointCloud_Vis():
    def __init__(self,cfg, new_config = False, width = 800, height = 800):
        self.vis = open3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=width, height=height, left=200)
        self.vis.get_render_option().load_from_json('config/render_option.json')
        self.cfg = cfg

        self.new_config = new_config
        if not os.path.exists(self.cfg):
            print('[Warn] The config file [%s] does not exist'%(self.cfg))
            print('Crearing a new config file from top_view.json')
            json.dump(cfg_template, open(self.cfg,'w'))
        else:
            print('Load config file [%s]'%(self.cfg))
            # Modify json file or there will be errors when we change window size
            data = json.load(open(self.cfg,'r'))
            data['intrinsic']['width'] = width
            data['intrinsic']['height'] = height
            data['intrinsic']['intrinsic_matrix'][6] = (width-1)/2
            data['intrinsic']['intrinsic_matrix'][7] = (height-1)/2
            json.dump(data, open(self.cfg,'w'),indent=4)
            self.param = open3d.io.read_pinhole_camera_parameters(self.cfg)

        self.pcd = open3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        self.vis.register_key_callback(32, lambda vis: exit())
    
    def __del__(self):
        self.vis.destroy_window()

    def update(self,pcd):
        self.pcd.points = pcd.points
        self.pcd.colors = pcd.colors
        self.vis.remove_geometry(self.pcd)
        self.vis.add_geometry(self.pcd)
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.param)
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

        if self.new_config:
            print('Move the frame to the place you want')
            print('---Press [Q] to save---')
            self.vis.run()
            self.param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
            open3d.io.write_pinhole_camera_parameters(self.cfg,self.param)
            self.new_config = False

            # add our own parameters
            data = json.load(open(self.cfg,'r'))
            data['h_fov'] = cfg_template['h_fov']
            data['v_fov'] = cfg_template['v_fov']
            data['x_range'] = cfg_template['x_range']
            data['y_range'] = cfg_template['y_range']
            data['z_range'] = cfg_template['z_range']
            data['d_range'] = cfg_template['d_range']
            json.dump(data, open(self.cfg,'w'),indent=4)

            print('Saved Please restart using [%s]' % self.cfg)
            exit()

    def capture_screen(self,fn, depth = False):
        if depth:
            self.vis.capture_depth_image(fn, False)
        else:
            self.vis.capture_screen_image(fn, False)

class Semantic_KITTI_Utils():
    def __init__(self,sequence_root):
        self.sequence_root = sequence_root
        self.index = 0
        self.load()

    def load(self):
        # R_vc = Rotation matrix ( velodyne -> camera )
        # T_vc = Translation matrix ( velodyne -> camera )
        self.R_vc, self.T_vc = calib_velo2cam('calib/calib_velo_to_cam.txt')

        # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
        self.P_ = calib_cam2cam('calib/calib_cam_to_cam.txt' ,mode="02")

        # RT_ - rotation matrix & translation matrix ( velodyne coordinates -> camera coordinates )
        #         [r_11 , r_12 , r_13 , t_x ]
        # RT_  =  [r_21 , r_22 , r_23 , t_y ]
        #         [r_31 , r_32 , r_33 , t_z ]
        self.RT_ = np.concatenate((self.R_vc, self.T_vc), axis=1)

        self.sem_cfg = yaml.load(open('config/semantic-kitti.yaml','r'), Loader=yaml.SafeLoader)
        self.sem_class_names = self.sem_cfg['labels']
        self.sem_learning_map = self.sem_cfg['learning_map']
        self.sem_learning_map_inv = self.sem_cfg['learning_map_inv']
        self.sem_learning_ignore = self.sem_cfg['learning_ignore']

        color_map = self.sem_cfg['color_map']
        sem_color_list = []
        for i in range(0,260):
            if i in color_map.keys():
                color = color_map[i]
                sem_color_list.append([color[-1], color[1], color[0]])
            else:
                sem_color_list.append([0,0,0])
        self.sem_color_map = np.array(sem_color_list,dtype=np.uint8)

    def next(self):
        fn_frame = os.path.join(self.sequence_root, 'image_2/%06d.png' % (self.index))
        fn_velo = os.path.join(self.sequence_root, 'velodyne/%06d.bin' %(self.index))
        fn_label = os.path.join(self.sequence_root, 'labels/%06d.label' %(self.index))

        if not os.path.exists(fn_frame) or not os.path.exists(fn_velo):
            print('End of sequence')
            return False
        
        if not os.path.exists(fn_label):
            print('Semantic KITTI label file not found')
            return False

        self.frame = cv2.imread(fn_frame)
        self.points = np.fromfile(fn_velo, dtype=np.float32).reshape(-1, 4)[:,:3]
        self.n_pts = self.points.shape[0]
        label = np.fromfile(fn_label, dtype=np.uint32).reshape((-1))

        if label.shape[0] == self.points.shape[0]:
            self.sem_label = label & 0xFFFF  # semantic label in lower half
            self.inst_label = label >> 16  # instance id in upper half
            assert((self.sem_label + (self.inst_label << 16) == label).all()) # sanity check
        else:
            print("Points shape: ", self.points.shape)
            print("Label shape: ", label.shape)
            raise ValueError("Scan and Label don't contain same number of points")

        self.index += 1
        return True
    
    def set_filter(self, h_fov, v_fov, x_range = None, y_range = None, z_range = None, d_range = None):
        # rough velodyne azimuth range corresponding to camera horizontal fov
        self.h_fov = h_fov if h_fov is not None else (-180, 180)
        self.v_fov = v_fov if v_fov is not None else (-25, 2)
        self.x_range = x_range if x_range is not None else (-10000, 10000)
        self.y_range = y_range if y_range is not None else (-10000, 10000)
        self.z_range = z_range if z_range is not None else (-10000, 10000)
        self.d_range = d_range if d_range is not None else (-10000, 10000)

    def hv_in_range(self, m, n, fov, fov_type='h'):
        """ extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit 
            horizontal limit = azimuth angle limit
            vertical limit = elevation angle limit
        """
        if fov_type == 'h':
            return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), \
                                    np.arctan2(n, m) < (-fov[0] * np.pi / 180))
        elif fov_type == 'v':
            return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), \
                                    np.arctan2(n, m) > (fov[0] * np.pi / 180))
        else:
            raise NameError("fov type must be set between 'h' and 'v' ")

    def box_in_range(self,x,y,z,d, x_range, y_range, z_range, d_range):
        """ extract filtered in-range velodyne coordinates based on x,y,z limit """
        return np.logical_and.reduce((
                x > x_range[0], x < x_range[1],
                y > y_range[0], y < y_range[1],
                z > z_range[0], z < z_range[1],
                d > d_range[0], d < d_range[1]))

    def points_basic_filter(self, points):
        """
            filter points based on h,v FOV and x,y,z distance range.
            x,y,z direction is based on velodyne coordinates
            1. azimuth & elevation angle limit check
            2. x,y,z distance limit
            return a bool array
        """
        assert points.shape[1] == 3, points.shape # [N,3]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) # this is much faster than d = np.sqrt(np.power(points,2).sum(1))

        # extract in-range fov points
        h_points = self.hv_in_range(x, y, self.h_fov, fov_type='h')
        v_points = self.hv_in_range(d, z, self.v_fov, fov_type='v')
        combined = np.logical_and(h_points, v_points)

        # extract in-range x,y,z points
        in_range = self.box_in_range(x,y,z,d, self.x_range, self.y_range, self.z_range, self.d_range)
        combined = np.logical_and(combined, in_range)

        return combined

    def extract_points(self,voxel_size = -1, every_k_points = 0):
        """ extract points corresponding to FOV setting """

        # filter in range points based on fov, x,y,z range setting
        combined = self.points_basic_filter(self.points)
        pts = self.points[combined]
        sem_label = self.sem_label[combined]
        fake_color = np.repeat(sem_label.reshape((-1,1)),3,axis=1).astype(np.float32)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pts)
        pcd.colors = open3d.utility.Vector3dVector(fake_color)

        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
        
        if every_k_points > 0 :
            pcd = pcd.uniform_down_sample(every_k_points)

        sem_label = np.asarray(pcd.colors)[:,0].astype(np.uint16)
        colors = self.sem_color_map[sem_label]
        pcd.colors = open3d.utility.Vector3dVector(colors)

        return pcd,sem_label

    def cvt_pcd(self,pcd, in_view_constraints=True):
        """ 
            Convert open3d.geometry.PointCloud object to [4, N] array
                        [x_1 , x_2 , .. ]
            xyz_v   =   [y_1 , y_2 , .. ]
                        [z_1 , z_2 , .. ]
                        [ 1  ,  1  , .. ]
        """
        # The [N,3] downsampled array
        pts_3d = np.asarray(pcd.points)

        if in_view_constraints:
            h_points = self.hv_in_range(pts_3d[:,0], pts_3d[:,1], [-90,90], fov_type='h')
            pts_3d = pts_3d[h_points]

        # Create a [N,1] array
        one_mat = np.ones((pts_3d.shape[0], 1),dtype=np.float64)
        # Concat and change shape from [N,4] to [4,N]
        xyz_v = np.concatenate((pts_3d, one_mat), axis=1).T
        return xyz_v

    def project_3d_points(self, pcd):
        # convert open3d.geometry.PointCloud object to [4, N] array
        xyz_v = self.cvt_pcd(pcd)

        assert xyz_v.shape[0] == 4, xyz_v.shape

        # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
        for i in range(xyz_v.shape[1]):
            xyz_v[:3, i] = np.matmul(self.RT_, xyz_v[:, i])

        """
        xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
                   [x_1 , x_2 , .. ]
        xyz_c   =  [y_1 , y_2 , .. ]
                   [z_1 , z_2 , .. ]
        """
        xyz_c = np.delete(xyz_v, 3, axis=0)

        # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
        for i in range(xyz_c.shape[1]):
            xyz_c[:, i] = np.matmul(self.P_, xyz_c[:, i])

        """
        xy_i   - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
        pts_2d - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
                    [s_1*x_1 , s_2*x_2 , .. ]
        xy_i    =   [s_1*y_1 , s_2*y_2 , .. ]     pts_2d =   [x_1 , x_2 , .. ]
                    [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
        """
        xy_i = xyz_c[::] / xyz_c[::][2]
        pts_2d = np.delete(xy_i, 2, axis=0)

        points = xyz_v[:3].T
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) # this is much faster than d = np.sqrt(np.power(points,2).sum(1))
        color = self.normalize_data(d, min=1, max=70, scale=120, clip=True)

        return pts_2d.T, color

    def normalize_data(self, val, min, max, scale, depth=False, clip=False):
        """ Return normalized data """
        if clip:
            # limit the values in an array
            np.clip(val, min, max, out=val)
        if depth:
            """
            print 'normalized depth value'
            normalize values to (0 - scale) & close distance value has high value. (similar to stereo vision's disparity map)
            """
            return (((max - val) / (max - min)) * scale).astype(np.uint8)
        else:
            """
            print 'normalized value'
            normalize values to (0 - scale) & close distance value has low value.
            """
            return (((val - min) / (max - min)) * scale).astype(np.uint8)
