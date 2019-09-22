import open3d
import json
import os
import numpy as np
from PIL import Image
import glob
import cv2
import shutil
import yaml
import colorsys

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

class PointCloud_Vis():
    def __init__(self,cfg, new_config = False, width = 800, height = 800):
        self.vis = open3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=width, height=height, left=100)
        self.vis.get_render_option().load_from_json('config/render_option.json')
        self.cfg = cfg

        self.new_config = new_config
        # Modify json file or there will be errors when we change window size
        print('Load config file [%s]'%(self.cfg))
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
            data = json.load(open(self.cfg,'r'))

            self.param = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
            open3d.io.write_pinhole_camera_parameters(self.cfg,self.param)
            self.new_config = False

            # add our own parameters
            cfg = json.load(open(self.cfg,'r'))
            cfg['h_fov'] = data['h_fov']
            cfg['v_fov'] = data['v_fov']
            cfg['x_range'] = data['x_range']
            cfg['y_range'] = data['y_range']
            cfg['z_range'] = data['z_range']
            cfg['d_range'] = data['d_range']
            json.dump(cfg, open(self.cfg,'w'),indent=4)

            print('Saved. Please restart using [%s]' % self.cfg)
            exit()

    def capture_screen(self,fn, depth = False):
        if depth:
            self.vis.capture_depth_image(fn, False)
        else:
            self.vis.capture_screen_image(fn, False)

class Semantic_KITTI_Utils():
    def __init__(self, root):
        self.root = root
        self.init()

    def set_part(self, part='00'):
        length = {
            '00': 4540,'01':1100,'02':4660,'03':800,'04':270,'05':2760,
            '06':1100,'07':1100,'08':4070,'09':1590,'10':1200
        }
        assert part in length.keys(), 'Only %s are supported' %(length.keys())
        self.sequence_root = os.path.join(self.root, 'sequences/%s/'%(part))
        self.index = 0
        self.max_index = length[part]
        return self.max_index
    
    def get_max_index(self):
        return self.max_index

    def init(self):
        # R_vc = Rotation matrix ( velodyne -> camera )
        # T_vc = Translation matrix ( velodyne -> camera )
        self.R_vc, self.T_vc = calib_velo2cam('config/calib_velo_to_cam.txt')

        # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
        self.P_ = calib_cam2cam('config/calib_cam_to_cam.txt' ,mode="02")

        # RT_ - rotation matrix & translation matrix ( velodyne coordinates -> camera coordinates )
        #         [r_11 , r_12 , r_13 , t_x ]
        # RT_  =  [r_21 , r_22 , r_23 , t_y ]
        #         [r_31 , r_32 , r_33 , t_z ]
        self.RT_ = np.concatenate((self.R_vc, self.T_vc), axis=1)

        self.sem_cfg = yaml.load(open('config/semantic-kitti.yaml','r'), Loader=yaml.SafeLoader)
        self.class_names = self.sem_cfg['labels']
        self.learning_map = self.sem_cfg['learning_map']
        self.learning_map_inv = self.sem_cfg['learning_map_inv']
        self.learning_ignore = self.sem_cfg['learning_ignore']
        self.sem_color_map = self.sem_cfg['color_map']

        self.kitti_color_map = [[0,0,0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                        [153, 153, 153], [250, 170, 30], [220, 220, 0],[107, 142, 35], [152, 251, 152], [0, 130, 180],
                        [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230],[119, 11, 32]]

    def load(self,index = None):
        """  Load the frame, point cloud and semantic labels from file """

        self.index = index
        if self.index == self.max_index:
            print('End of sequence')
            return False

        fn_frame = os.path.join(self.sequence_root, 'image_2/%06d.png' % (self.index))
        fn_velo = os.path.join(self.sequence_root, 'velodyne/%06d.bin' %(self.index))
        fn_label = os.path.join(self.sequence_root, 'labels/%06d.label' %(self.index))

        assert os.path.exists(fn_frame), 'Broken dataset %s' % (fn_frame)
        assert os.path.exists(fn_velo), 'Broken dataset %s' % (fn_velo)
        assert os.path.exists(fn_label), 'Broken dataset %s' % (fn_label)

        self.frame = cv2.imread(fn_frame)
        assert self.frame is not None, 'Broken dataset %s' % (fn_frame)
            
        self.points = np.fromfile(fn_velo, dtype=np.float32).reshape(-1, 4)
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

        return True
    
    def set_filter(self, h_fov, v_fov, x_range = None, y_range = None, z_range = None, d_range = None):
        # rough velodyne azimuth range corresponding to camera horizontal fov
        self.h_fov = h_fov if h_fov is not None else (-180, 180)
        self.v_fov = v_fov if v_fov is not None else (-25, 2)
        self.x_range = x_range if x_range is not None else (-10000, 10000)
        self.y_range = y_range if y_range is not None else (-10000, 10000)
        self.z_range = z_range if z_range is not None else (-10000, 10000)
        self.d_range = d_range if d_range is not None else (-10000, 10000)

        self.min_bound = [self.x_range[0], self.y_range[0], self.z_range[0]]
        self.max_bound = [self.x_range[1], self.y_range[1], self.z_range[1]]

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
        assert points.shape[1] == 4, points.shape # [N,3]
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

    def extract_points(self,voxel_size = 0.01):
        # filter in range points based on fov, x,y,z range setting
        combined = self.points_basic_filter(self.points)
        points = self.points[combined]
        label = self.sem_label[combined]

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points[:,:3])

        # approximate_class must be set to true
        # see this issue for more info https://github.com/intel-isl/Open3D/issues/1085
        pcd,trace = pcd.voxel_down_sample_and_trace(voxel_size,self.min_bound,self.max_bound,approximate_class=True)
        to_index_org = np.max(trace, 1)

        pts = points[to_index_org]
        sem_label = label[to_index_org]
        self.pts = pts
        colors = np.array([self.sem_color_map[x] for x in sem_label])
        pcd.colors = open3d.utility.Vector3dVector(colors/255.0)

        return pcd,sem_label

    def get_in_view_pts(self, pcd, sem_label):
        """ 
            Convert open3d.geometry.PointCloud object to [4, N] array
                        [x_1 , x_2 , .. ]
            xyz_v   =   [y_1 , y_2 , .. ]
                        [z_1 , z_2 , .. ]
                        [ 1  ,  1  , .. ]
        """
        # The [N,3] downsampled array
        pts_3d = np.asarray(pcd.points)

        # finter out the points not in view
        h_points = self.hv_in_range(pts_3d[:,0], pts_3d[:,1], [-50,50], fov_type='h')
        pts_3d = pts_3d[h_points]
        sem_label = sem_label[h_points]

        return pts_3d, sem_label

    def project_3d_to_2d(self, pts_3d):
        assert pts_3d.shape[1] == 3, pts_3d.shape

        # Create a [N,1] array
        one_mat = np.ones((pts_3d.shape[0], 1),dtype=np.float64)

        # Concat and change shape from [N,3] to [N,4] to [4,N]
        xyz_v = np.concatenate((pts_3d, one_mat), axis=1).T

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
        pts = np.delete(xy_i, 2, axis=0)
        pts_2d = pts.T
        assert pts_2d.shape[1] == 2

        points = xyz_v[:3].T
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        d_normalize = (d - d.min()) / (d.max() - d.min())
        color = [[int(x*255) for x in colorsys.hsv_to_rgb(hue,1,1)] for hue in d_normalize]

        return pts_2d, color

    def draw_2d_points(self, pts_2d, color):
        """ draw 2d points in camera image """
        assert pts_2d.shape[1] == 2, pts_2d.shape

        image = self.frame.copy()
        pts_2d = pts_2d.astype(np.int32).tolist()

        for (x,y),c in zip(pts_2d,color):
            cv2.circle(image, (x, y), 2, [c[2],c[1],c[0]], -1)
            
        return image

    def draw_2d_sem_points(self, pts_2d, sem_label):
        """ draw 2d points in camera image """
        assert pts_2d.shape[1] == 2, pts_2d.shape
        assert pts_2d.shape[0] == sem_label.shape[0], str(pts_2d.shape) + ' '+  str(sem_label.shape)

        image = self.frame.copy()
        pts_2d = pts_2d.astype(np.int32).tolist()
        colors = [self.sem_color_map[x] for x in sem_label.tolist()]

        for (x,y),c in zip(pts_2d,colors):
            cv2.circle(image, (x, y), 2, [c[2],c[1],c[0]], -1)
        return image

    def draw_2d_sem_points_with_learning_mapping(self, pts_2d, sem_label_learn):
        """ draw 2d points in camera image """
        assert pts_2d.shape[1] == 2, pts_2d.shape
        assert pts_2d.shape[0] == sem_label_learn.shape[0], str(pts_2d.shape) + ' '+  str(sem_label_learn.shape)
        
        image = self.frame.copy()
        pts_2d = pts_2d.astype(np.int32).tolist()
        colors = [self.kitti_color_map[x] for x in sem_label_learn.tolist()]

        for (x,y),c in zip(pts_2d,colors):
            cv2.circle(image, (x, y), 2, [c[2],c[1],c[0]], -1)
        return image

    def learning_mapping(self,sem_label):
        # Note: Here the 19 classs are different from the original KITTI 19 classes
        num_classes = 20
        class_names = [
            'unlabelled',     # 0
            'car',            # 1
            'bicycle',        # 2
            'motorcycle',     # 3
            'truck',          # 4
            'other-vehicle',  # 5
            'person',         # 6
            'bicyclist',      # 7
            'motorcyclist',   # 8
            'road',           # 9
            'parking',        # 10
            'sidewalk',       # 11
            'other-ground',   # 12
            'building',       # 13
            'fence',          # 14
            'vegetation',     # 15
            'trunk',          # 16
            'terrain',        # 17
            'pole',           # 18
            'traffic-sign'    # 19
        ]
        sem_label_learn = [self.learning_map[x] for x in sem_label]
        sem_label_learn = np.array(sem_label_learn, dtype=np.uint8)
        return sem_label_learn

    def inv_learning_mapping(self,sem_label_learn):
        sem_label = [self.learning_map_inv[x] for x in sem_label_learn]
        sem_label = np.array(sem_label, dtype=np.uint16)
        return sem_label