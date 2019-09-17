import os
import numpy as np
from PIL import Image
import glob
import cv2
import argparse
import json
import imageio
from tqdm import tqdm

def make_gif():
    images = []
    for index in tqdm(range(50,70,1)):
        fn1 = 'tmp/img/%06d.png'%(index)
        fn2 = 'tmp/pcd/%06d.png'%(index)
        img1 = Image.open(fn1)
        img1.thumbnail((800,300),Image.ANTIALIAS)
        img2 = Image.open(fn2).crop((0,120,800,460))

        img_np = np.vstack((np.array(img2), np.array(img1)))
        img = Image.fromarray(img_np)

        img.thumbnail((400,400))
        images.append(img)
    imageio.mimsave('assets/semantic-kitti.gif', images, fps=5)

if __name__ == "__main__":
    img = Image.open('tmp/frame_semantic.jpg')
    img.thumbnail((800,800),Image.ANTIALIAS)
    img.save('assets/frame_semantic.jpg')

    img = Image.open('tmp/frame_depth.jpg')
    img.thumbnail((800,800),Image.ANTIALIAS)
    img.save('assets/frame_depth.jpg')

    img = Image.open('tmp/ego_view.jpg')
    img = img.crop((0,350,800,670))
    img.save('assets/ego_view.jpg')

    img = Image.open('tmp/top_view.jpg')
    img = img.crop((0,350,800,550))
    img.save('assets/top_view.jpg')