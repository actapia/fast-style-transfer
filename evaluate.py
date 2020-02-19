from __future__ import print_function

import os
import cv2
import sys
import subprocess
sys.path.insert(0, 'src')

import warnings
import transform
import scipy.misc
import numpy as np
import tensorflow as tf
from utils import save_img, get_img
from skimage.transform import resize
import matplotlib
import matplotlib.pyplot as plt

import threading, multiprocessing

from PyQt5 import QtCore

import time

from screeninfo import get_monitors

import argparse

user_image_path = 'examples/user_image.png'
style_ckpts = ['ckpts/la_muse.ckpt',
'ckpts/rain_princess.ckpt',
'ckpts/scream.ckpt',
'ckpts/wreck.ckpt',
'ckpts/wave.ckpt',
'ckpts/udnie.ckpt',
'ckpts/fns.ckpt']

default_backend = "Qt5Agg"

# Disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
       
def transfer(img, style_ckpt, crop=True):
  g = tf.Graph()
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True

  with g.as_default(), g.device('/gpu:0'), tf.Session(config=config) as sess:

    # Resize image
    h, w, _ = img.shape
    mx, my = h//2, w//2

    if float(w) / float(h) < 2:
      h = w // 2
	  
      img = img[mx-h//2:mx+h//2, :, :]

    else:
      w = 2 * h
      img = img[:, my-w//2:my+w//2, :]

    if not crop:
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          img = resize(img, (400, 800))      

    # Build inference network
    img = np.reshape(img, (1,) + img.shape)
    batch_shape = img.shape
    img_holder = tf.placeholder(tf.float32, shape=batch_shape)
    styled_img = transform.net(img_holder)

    # Restore style weights
    saver = tf.train.Saver()
    if style_ckpt == "ckpts/fns.ckpt":
        tf.train.import_meta_graph(style_ckpt + ".meta")
    saver.restore(sess, style_ckpt)

    styled_img_ = sess.run(styled_img, feed_dict={img_holder: img})

  return styled_img_
  
 
def show_image(styled_img):
    matplotlib.use(default_backend)
    plt.ion()
    plt.clf()
    plt.imshow(styled_img)
    plt.axis('off')
    plt.tight_layout()
    plt.pause(0.001)
    f = plt.gcf()
    monitors = get_monitors()
    f.canvas.manager.window.move(monitors[-1].x, 0)
    f.canvas.manager.window.showMaximized()
    plt.show(block=True)
    

def clear_screen():
    pass
    # subprocess.call(["powershell.exe","Clear-Host"],shell=True)
    
def show_options(options, first=1, prompt="Selection: "):
    for index, option in enumerate(options):
        print("  [{0}] {1}".format(index+first, option))
    return int(input(prompt))
    
def style_image(img, ckpt, crop=True):
    styled_img = transfer(img, ckpt, crop)[0]
    return np.clip(styled_img, 0, 255).astype(np.uint8)

# After each image, have the option of saving the image for email later.
# Also have the option to do a different style
#
# After the person leaves, then have the option of resetting
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--camera","-c",type=int,help="camera to use",default=0,required=False)
  parser.add_argument("--no-crop",help="don't crop images",action="store_true",required=False)
  args = parser.parse_args()
  clear_screen()
  multiprocessing.set_start_method("spawn")
  if matplotlib.get_backend() != default_backend:
      print("Warning: Will use {0} as matplotlib backend.".format(default_backend))
  img = None
  print('Source image:')
  source = show_options(["Choose from examples","Take your own","Load from file","Load last image from webcam"])

  if source == 1:
    print('\nChoose from examples:')
    example_idx = show_options(["Chicago","Modern Building"])
    if example_idx != 0:
        example_images = ['examples/content/chicago.jpg',
        'examples/content/stata.jpg']
        img_fname = example_images[example_idx-1]
        img = get_img(img_fname)
  elif source == 2:
    print('\nTaking a picture with webcam {0}.'.format(args.camera))

    cam = cv2.VideoCapture(args.camera)
    cam.set(3, 1280)
    cam.set(4, 720)
    cv2.namedWindow('camera')

    while True:
      ret, frame = cam.read()
      cv2.startWindowThread()
      cv2.imshow('camera', frame)
      if not ret:
        break
      k = cv2.waitKey(1)
      
      # Space is pressed
      if k % 256 == 32:
        cv2.imwrite(user_image_path, frame)
        img = get_img(user_image_path)
        print('Picture taken!')
        break

    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1) # Needed to actually destroy windows in macOS, apparently.
  elif source == 3:
    print('\nLoading image from a file.')
    img = get_img(input("File path: "))
  elif source == 4:
    print('\nLoading last image taken.')
    img = get_img(user_image_path)

  if not img is None:
      job = None
      while True:
        clear_screen()
        print('Source style')
        
        style_idx = show_options(["La Muse",
        "Rain Princess",
        "The Scream",
        "The Shipwreck of the Minotaur",
        "Wave",
        "Udnie"])

        # Quit if we enter 0
        if style_idx == 0:
          if job:
            job.terminate()
          save_option = show_options(["Quit","Save styles."],first=0)
          if save_option == 1:
            # Save all styles.
            out_dir = input("Output directory: ")
            try:
                os.mkdir(out_dir)
            except FileExistsError:
                pass
            for style_index, style_ckpt in enumerate(style_ckpts):
                styled_img = style_image(img, style_ckpt)
                save_img(os.path.join(out_dir,"style{0}.png".format(style_index)),styled_img)
          break
        
        style_ckpt = style_ckpts[style_idx-1]
        
        styled_img = style_image(img, style_ckpt, not args.no_crop)

        if job:
            job.terminate()
        job = multiprocessing.Process(target=show_image,args=(styled_img,))
        job.start()
