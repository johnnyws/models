from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time
import sys

import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception

from flask import Flask
from flask import request
app = Flask(__name__)

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, sess):
    # Create a single Session to run all image coding calls.
    #self._sess = tf.Session()
    self._sess = sess

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  return '.png' in filename

def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
  """
  # Read the image file.
  image_data = tf.gfile.FastGFile(filename, 'r').read()

  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)

  return image_data

def _process_image_data(image_data, coder, is_png):
  if is_png:
    print('Converting PNG to JPEG')
    image_data = coder.png_to_jpeg(image_data)

  return image_data


# load the model
def setup_app(app):
  tf.Graph().as_default()

  # a tensor of size [batch_size, height, width, channels]
  images_ph = tf.placeholder(tf.float32, [1, 299, 299, 3])
  num_classes = 196 + 1

  logits, _ = inception.inference(images_ph, num_classes)

  # Restore the moving average version of the learned variables for eval.
  variable_averages = tf.train.ExponentialMovingAverage(
      inception.MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

  sess = tf.Session()

  ckpt_path = "/home/ubuntu/src/tensorflow-test/cars-train/"
  ckpt = tf.train.get_checkpoint_state(ckpt_path)
  if ckpt and ckpt.model_checkpoint_path:
    if os.path.isabs(ckpt.model_checkpoint_path):
      # Restores from checkpoint with absolute path.
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      # Restores from checkpoint with relative path.
      saver.restore(sess, os.path.join(ckpt_path,
                                       ckpt.model_checkpoint_path))

    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/imagenet_train/model.ckpt-0,
    # extract global_step from it.
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    print('Succesfully loaded model from %s at step=%s.' %
          (ckpt.model_checkpoint_path, global_step))
  else:
    print('No checkpoint file found')
    sys.exit(1)

  return sess, images_ph, logits

sess, images_ph, logits = setup_app(app)
coder = ImageCoder(sess)

@app.route("/image", methods = ['POST'])
def image():
  image_data = request.data
  print(request.headers.get("Content-Type"))
  return str(request.headers)

@app.route("/classify", methods = ['POST'])
def classify():
  start = time.time()
  # for testing
  #filename = "/home/ubuntu/src/tensorflow-models/inception/tmp/raw/1/bentley-continental-gt-matte-orange1.jpg"

  image_data = request.data
  image_type = request.headers.get("Content-Type")
  image_data = _process_image_data(image_data, coder, image_type == "image/png")

  # image is a tensor
  image = image_processing.decode_jpeg(image_data)
  image = image_processing.eval_image(image, 299, 299)
  image = tf.sub(image, 0.5)
  image = tf.mul(image, 2.0)

  image = sess.run(image)

  logits_result = sess.run(logits, feed_dict={images_ph:[image]})[0]
  end = time.time()
  sorted_indexes = [i[0] for i in sorted(enumerate(logits_result), key=lambda x:x[1], reverse=True)]
  for i in range(0, 5):
    logit = logits_result[sorted_indexes[i]]
    prob = 1 / (1 + math.exp(-logit))
    print('%d: %f' % (sorted_indexes[i], prob))

  return "Hello World!"

if __name__ == '__main__':
  app.run(debug=True, use_reloader=False)
