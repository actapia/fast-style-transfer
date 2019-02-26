from __future__ import print_function

import os
import sys
sys.path.insert(0, 'src')

import glob
import warnings
import transform
import scipy.misc
import numpy as np
import tensorflow as tf
from utils import save_img, get_img
from skimage.transform import resize
import matplotlib.pyplot as plt

# Disable tensorflow debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def transfer(img, style_ckpt):
  g = tf.Graph()
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True

  with g.as_default(), g.device('/gpu:0'), tf.Session(config=config) as sess:

    # Resize image
    h, w, _ = img.shape
    mx, my = h/2, w/2

    if float(w) / float(h) < 2:
      h = w / 2
      img = img[mx-h/2:mx+h/2, :, :]

    else:
      w = 2 * h
      img = img[:, my-w/2:my+w/2, :]

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
    saver.restore(sess, style_ckpt)

    styled_img_ = sess.run(styled_img, feed_dict={img_holder: img})

  return styled_img_

def main():
  style_dict = {
    1: 'ckpts/la_muse.ckpt',
    2: 'ckpts/rain_princess.ckpt',
    3: 'ckpts/scream.ckpt',
    4: 'ckpts/wreck.ckpt',
    5: 'ckpts/udnie.ckpt',
    6: 'ckpts/wave.ckpt',
  }

  img_fnames = glob.glob('examples/eday/*.jpg')

  for img_f in img_fnames:
    email = '.'.join(img_f.split('.')[:-1])

if __name__ == '__main__':
  main()
