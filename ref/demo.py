import os
import numpy as np
import glob
import cv2
import sys
sys.path.append('.')
from ref.kitti_foundation import KITTI,KITTI_Util

root = '/media/james/MyPassport/James/dataset/KITTI/raw/2011_09_26/'
if not os.path.exists(root):
    raise Exception('Root directory does not exist')
velo_path = os.path.join(root, '2011_09_26_drive_0005_sync/velodyne_points/data')
image_path = os.path.join(root, '2011_09_26_drive_0005_sync/image_02/data')
v2c_filepath = os.path.join(root,'calib_velo_to_cam.txt')
c2c_filepath = os.path.join(root,'calib_cam_to_cam.txt')


def print_projection_cv2(points, color, image):
    """ project converted velodyne points into camera image """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    points = points.astype(np.int32).transpose((1,0)).tolist()
    color = color.astype(np.int32).tolist()

    for pt,c in zip(points,color):
        cv2.circle(hsv_image, (pt[0], pt[1]), 2, (c, 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def print_projection_plt(points, color, image):
    """ project converted velodyne points into camera image """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (int(points[0][i]), int(points[1][i])), 2, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def pano_example1():
    """ save one frame image about velodyne dataset converted to panoramic image  """
    v_fov, h_fov = (-10.5, 2.0), (-60, 80)
    velo = KITTI_Util(frame=89, velo_path=velo_path)

    frame = velo.velo_2_pano_frame(h_fov, v_fov, depth=False)

    cv2.imshow('panoramic result', frame)
    cv2.waitKey(0)

def pano_example2():
    """ save video about velodyne dataset converted to panoramic image  """
    v_fov, h_fov = (-24.9, 2.0), (-180, 160)

    velo2 = KITTI_Util(frame='all', velo_path=velo_path)
    pano = velo2.velo_2_pano(h_fov, v_fov, depth=False)

    velo = KITTI_Util(frame=0, velo_path=velo_path)
    velo.velo_2_pano_frame(h_fov, v_fov, depth=False)
    size = velo.surround_size

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid = cv2.VideoWriter('experiment/demo_pano.avi', fourcc, 25.0, size, False)

    for frame in pano:
        vid.write(frame)
        cv2.imshow('panoramic result', frame)
        cv2.waitKey(1)

    vid.release()

def topview_example1():
    """ save one frame image about velodyne dataset converted to topview image  """
    x_range, y_range, z_range = (-15, 15), (-10, 10), (-2, 2)
    velo = KITTI_Util(frame=89, velo_path=velo_path)

    frame = velo.velo_2_topview_frame(x_range=x_range, y_range=y_range, z_range=z_range)

    cv2.imshow('topview', frame)
    cv2.waitKey(0)

def topview_example2():
    """ save video about velodyne dataset converted to topview image  """
    x_range, y_range, z_range, scale = (-50, 50), (-50, 50), (-4, 4), 10
    size = (int((max(y_range) - min(y_range)) * scale), int((max(x_range) - min(x_range)) * scale))

    velo2 = KITTI_Util(frame='all', velo_path=velo_path)
    topview = velo2.velo_2_topview(x_range=x_range, y_range=y_range, z_range=z_range, scale=scale)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid = cv2.VideoWriter('experiment/demo_topview.avi', fourcc, 25.0, size, False)

    try:
        while True:
            frame = next(topview)
            vid.write(frame)
            cv2.imshow('topview', frame)
            if 27 == cv2.waitKey(1):
                break
    except StopIteration:
        pass

    vid.release()

def projection_example1():
    """ save one frame about projecting velodyne points into camera image """
    # v_fov, h_fov = (-24.9, 2.0), (-90, 90)
    h_fov = (-40.5, 40.5)
    v_fov = (-25, 2.0)

    res = KITTI_Util(frame=89, camera_path=image_path, velo_path=velo_path, \
                    v2c_path=v2c_filepath, c2c_path=c2c_filepath)

    img, points_2d, color = res.velo_projection_frame(v_fov=v_fov, h_fov=h_fov)
    print(res.num_frame, img.shape, points_2d.shape, color.shape)
    result = print_projection_cv2(points_2d, color, img)

    cv2.imshow('projection result', result)
    cv2.waitKey(0)

def projection_example2():
    """ save video about projecting velodyne points into camera image """

    h_fov = (-40, 40)
    v_fov = (-25, 2.0)
    temp = KITTI(frame=0, camera_path=image_path)
    img = temp.camera_file
    size = (img.shape[1], img.shape[0])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid = cv2.VideoWriter('experiment/demo_projection.avi', fourcc, 25.0, size)
    test = KITTI_Util(frame='all', camera_path=image_path, velo_path=velo_path,v2c_path=v2c_filepath, c2c_path=c2c_filepath)

    res = test.velo_projection(v_fov=v_fov, h_fov=h_fov)
    try:
        while True:
            frame, point, color = next(res)
            image = print_projection_cv2(point, color, frame)
            vid.write(image)
            cv2.imshow('projection', image)
            if 27 == cv2.waitKey(1):
                break
    except StopIteration:
        pass
    vid.release()

def xml_example():
    xml_path = "./tracklet_labels.xml"
    xml_check = KITTI_Util(xml_path=xml_path)

    tracklet_, type_ = xml_check.tracklet_info
    print(tracklet_[0])

if __name__ == "__main__":
    # pano_example1()
    # pano_example2()
    
    # topview_example1()
    # topview_example2()

    # projection_example1()
    projection_example2()

