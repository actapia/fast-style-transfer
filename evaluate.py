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
import matplotlib.pyplot as plt

import threading, multiprocessing

from PyQt5 import QtCore

import time

from screeninfo import get_monitors

user_image_path = 'examples/user_image.png'

# class CallInQTMainLoop(QtCore.QObject):
    # signal = QtCore.pyqtSignal()

    # def __init__(self, func):
        # super().__init__()
        # self.func = func
        # self.args = list()
        # self.kwargs = dict()
        # self.signal.connect(self._target)

    # def _target(self):
        # self.func(*self.args, **self.kwargs)

    # def __call__(self, *args, **kwargs):
        # self.args = args
        # self.kwargs = kwargs
        # self.signal.emit()

# Disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
# class ChooseThread(threading.Thread):
    # def __init__(self):
        # threading.Thread.__init__(self)
        
    # def run(self):
        # while True:
          # os.system('clear')
          # print('Source style')
          # print('  [1] La Muse')
          # print('  [2] Rain Princess')
          # print('  [3] The Scream')
          # print('  [4] The Shipwreck of the Minotaur')
          # print('  [5] Udnie')
          # print('  [6] Wave')
      
          # style_idx = int(input('Selection: '))

          # # Quit if we enter 0
          # if style_idx == 0:
            # break

          # style_dict = {
            # 1: 'ckpts/la_muse.ckpt',
            # 2: 'ckpts/rain_princess.ckpt',
            # 3: 'ckpts/scream.ckpt',
            # 4: 'ckpts/wreck.ckpt',
            # 5: 'ckpts/udnie.ckpt',
            # 6: 'ckpts/wave.ckpt',
          # }
          # style_ckpt = style_dict[style_idx]

          # styled_img = transfer(img, style_ckpt)[0]
          # styled_img = np.clip(styled_img, 0, 255).astype(np.uint8)
          # show_image(styled_img)
        
    
   

def transfer(img, style_ckpt):
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

    # with warnings.catch_warnings():
      # warnings.simplefilter("ignore")
      # # img = resize(img, (400, 800))
      
    # print(img.shape)

    # Build inference network
    img = np.reshape(img, (1,) + img.shape)
    batch_shape = img.shape
    img_holder = tf.placeholder(tf.float32, shape=batch_shape)
    styled_img = transform.net(img_holder)

    # Restore style weights
    saver = tf.train.Saver()
    saver.restore(sess, style_ckpt)

    styled_img_ = sess.run(styled_img, feed_dict={img_holder: img})

  return styled_img_
  
 
def show_image(styled_img):
    plt.ion()
    plt.clf()
    #plt.pause(0.001)
    plt.imshow(styled_img)
    #plt.pause(0.001)
    plt.axis('off')
    #plt.pause(0.001)
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
    
def style_image(img, ckpt):
    styled_img = transfer(img, ckpt)[0]
    return np.clip(styled_img, 0, 255).astype(np.uint8)

# After each image, have the option of saving the image for email later.
# Also have the option to do a different style
#
# After the person leaves, then have the option of resetting
if __name__ == '__main__':
  #os.system('clear')
  clear_screen()
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
    print('\nTaking a picture with the webcam')

    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)
    cam.set(4, 720)
    cv2.namedWindow('camera')

    while True:
      ret, frame = cam.read()
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
  elif source == 3:
    print('\nLoading image from a file.')
    img = get_img(input("File path: "))
  elif source == 4:
    print('\nLoading last image taken.')
    img = get_img(user_image_path)

  # plt.figure(figsize=(10,6))
  # plt.ion()
  # plt.pause(0.001)
  # choose_thread = ChooseThread()
  # choose_thread.start()
  if not img is None:
      style_ckpts = ['ckpts/la_muse.ckpt',
        'ckpts/rain_princess.ckpt',
        'ckpts/scream.ckpt',
        'ckpts/wreck.ckpt',
        'ckpts/wave.ckpt',
        'ckpts/udnie.ckpt']
      job = None
      while True:
        clear_screen()
        print('Source style')
        #print("Selection: ",end="")
        
        #thread = threading.Thread(target=input)
        #thread.start()
        #plt.ion()
        #plt.pause(0.001)
        style_idx = show_options(["La Muse",
        "Rain Princess",
        "The Scream",
        "The Shipwreck of the Minotaur",
        "Wave",
        "Udnie"])
        #thread.join()
        #style_idx = 4

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
        
        styled_img = style_image(img, style_ckpt)

        if job:
            job.terminate()
        job = multiprocessing.Process(target=show_image,args=(styled_img,))
        job.start()
        #show_image(styled_img)
        # plt.ion()
        # plt.show()

        # plt.show()
        # print("Shown")
        # plt.pause(0.001)
        # plt.show(block=False)
