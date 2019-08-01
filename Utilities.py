import numpy as np
from PIL import Image
import math
import os
import glob
import tensorflow as tf


def resize_images_png(height, width):
    for img in glob.glob("data/*.png"):
        image = Image.open(img)
        image.thumbnail([height, width])
        image.save(img, "PNG")


def create_grid(input, height, width, batch_size):
    shape_x = int(math.sqrt(batch_size))
    shape_y = int(math.sqrt(batch_size))
    grid = Image.new("RGB", (width*shape_x, height*shape_y), (0, 0, 0))
    count = 0
    for i in range(shape_x):
        for j in range(shape_y):
            pos = (i*width, j*height, i*width+width, j*height+height)
            image = Image.fromarray(input[count], "RGB")
            grid.paste(image, box=pos, mask=None)
            count += 1
    return np.array(grid)


def save_params(output, lr, bs, dr, do, ep, d):
    if not os.path.exists(output):
        os.makedirs(output)
    f = open(output+"/params.txt", "w")
    f.write("Learning rate: " + str(lr))
    f.write("\nBatch Size: " + str(bs))
    f.write("\nData Repeats: " + str(dr))
    f.write("\nDropout Rate: " + str(do))
    f.write("\nEpochs: " + str(ep))
    f.write("\nDistorted: "+ str(d))
    f.close()


def read_images(dr, m, test=False):
    dir = os.getcwd()
    if not test:
        data = os.path.join(dir, "data")
    else:
        data = os.path.join(dir, "test")
    imgs = []  # String filenames stored here
    for i in range(dr):
        for path, _, files in os.walk(data):
            for img in files:
                ext = os.path.splitext(img)[1].lower()
                if m == 0:
                    if ext == ".png":
                        imgs.append(path + '\\' + img)
                elif m == 1:
                    if ext == ".jpg":
                        imgs.append(path + '\\' + img)
    size = len(imgs)
    imgs = tf.convert_to_tensor(imgs)
    dataset = tf.data.Dataset.from_tensor_slices(imgs)
    return dataset, size


def parse_image(path, m, h, w, c, d, corrupt=False):
    image = tf.read_file(path)
    if m == 0:
        image_decoded = tf.image.decode_png(image, c)
    elif m == 1:
        image_decoded = tf.image.decode_jpeg(image, c)
    image = tf.image.resize_images(image_decoded, [h, w])
    if d:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.cast(image, tf.float32)
    image = image/255.0
    if corrupt:
        image = image + np.random.normal(loc=0.5, scale=0.5, size=3)
    return image
