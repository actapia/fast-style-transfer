from __future__ import print_function

# import cv2
import sys
sys.path.insert(0, 'src')

import transform
import scipy.misc
import numpy as np
import tensorflow as tf
from utils import save_img, get_img
from skimage.transform import resize
import matplotlib.pyplot as plt

def transfer(img, style_ckpt):
  g = tf.Graph()
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True

  with g.as_default(), g.device('/gpu:0'), tf.Session(config=config) as sess:

    # Resize image
    h, w, _ = img.shape
    mx, my = h/2, w/2

    if float(w) / float(h) < 2:
      h = w/2
      img = img[mx-h/2:mx+h/2, :, :]
      img = resize(img, (400, 800))

    else:
      w = 2 * h
      img = img[:, my-w/2:my+w/2, :]
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

# After each image, have the option of saving the image for email later.
# Also have the option to do a different style
#
# After the person leaves, then have the option of resetting
if __name__ == '__main__':
  print('Source image:')
  print('  [1] Choose from examples')
  print('  [2] Take your own!')
  source = int(input('Selection: '))

  if source == 1:
    print('\nChoose from examples:')
    print('  [1] Chicago')
    print('  [2] Modern Building')
    example_idx = int(input('Selection: '))

    example_dict = {
      1: 'examples/content/chicago.jpg',
      2: 'examples/content/stata.jpg',
      3: 'examples/content/uk.jpg',
      4: 'examples/content/basketball.jpg',
    }
    img_fname = example_dict[example_idx]
    img = get_img(img_fname)

  elif source == 2:
    print('\nTaking a picture with the webcam')

    cam = cv2.VideoCapture(0)
    cv2.namedWindow('camera')

    while True:
      print('test')
      ret, frame = cam.read()
      cv2.imshow('camera', frame)
      if not ret:
        break
      k = cv2.waitKey(1)

      # Space is pressed
      if k % 256 == 32:
        img_fname = 'examples/user_image.png'
        cv2.imwrite(img_name, frame)
        print('Picture taken!')
        break

      cam.release()
      cv2.destroyAllWindows()

  print('\nSource style')
  print('  [1] La Muse')
  print('  [2] Rain Princess')
  print('  [3] The Scream')
  print('  [4] The Shipwreck of the Minotaur')
  print('  [5] Udnie')
  print('  [6] Wave')
  style_idx = int(input('Selection: '))

  style_dict = {
    1: 'ckpts/la_muse.ckpt',
    2: 'ckpts/rain_princess.ckpt',
    3: 'ckpts/scream.ckpt',
    4: 'ckpts/wreck.ckpt',
    5: 'ckpts/udnie.ckpt',
    6: 'ckpts/wave.ckpt',
  }
  style_ckpt = style_dict[style_idx]

  styled_img = transfer(img, style_ckpt)[0]
  styled_img = np.clip(styled_img, 0, 255).astype(np.uint8)

  plt.imshow(styled_img)
  plt.axis('off')
  plt.tight_layout()
  plt.show()
