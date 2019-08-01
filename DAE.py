import tensorflow as tf
import tensorflow.contrib.layers as tfcl
import os
import numpy as np
import imageio
import datetime
import Utilities as util
import Layers_autoencoders as lay

LEARNING_RATE = 0.001
BATCH_SIZE = 64
HEIGHT = 128
WIDTH = 128
CHANNEL = 3
DATA_REPEATS = 10
EPOCHS = 10
VERSION = "DAE_1"
DATE = datetime.datetime.now().strftime("%d-%m-%H-%M-%S")
SAVE_PATH = "./Project Out/" + VERSION + "/saved/"
MODE = 1  # 0 For PNG, 1 for JPEG
DROPOUT_RATE = 0.5
DISTORTED = False
RESTORE = True


def encoder(input, is_train):
    input = tf.nn.dropout(input, DROPOUT_RATE)
    enc1 = lay.conv(input, 64, "enc1", is_train)
    enc2 = lay.conv(enc1, 128, "enc2", is_train)
    return lay.conv(enc2, 256, "enc3", is_train)


def decoder(encoding, is_train):
    dec1 = lay.conv_t(encoding, 256, "dec1", is_train)
    dec2 = lay.conv_t(dec1, 128, "dec2", is_train)
    return lay.conv_t(dec2, 3, "dec3", is_train)

def train(output, restore):
    dataset, size = util.read_images(DATA_REPEATS, MODE)
    size = size * EPOCHS
    dataset = dataset.map(lambda x: util.parse_image(x, MODE, HEIGHT, WIDTH, CHANNEL, DISTORTED))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.batch(BATCH_SIZE)

    iterator = dataset.make_one_shot_iterator()
    image_batch = iterator.get_next()

    iterator_test = dataset.make_one_shot_iterator()
    test_batch = iterator_test.get_next()

    dataset_out, _ = util.read_images(1, MODE)
    dataset_out = dataset_out.map(lambda x: util.parse_image(x, MODE, HEIGHT, WIDTH, CHANNEL, DISTORTED))
    dataset_out = dataset_out.batch(BATCH_SIZE)

    iterator_out = dataset_out.make_one_shot_iterator()
    out_image = iterator_out.get_next()

    input = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
    is_train = tf.placeholder(tf.bool)
    out = decoder(encoder(input, is_train), is_train)

    loss = tf.reduce_mean(tf.square(input-out))
    optimiser = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists("./log/" + VERSION + DATE):
        os.makedirs("./log/" + VERSION + DATE)
    writer = tf.summary.FileWriter("./log/" + VERSION + " " + DATE)

    loss_summary = tf.summary.scalar("loss", loss)
    writer.add_graph(sess.graph)
    merged = tf.summary.merge_all()
    i = 0

    while True:
        try:
            print("Running iteration {}/{}...".format(i, size//BATCH_SIZE+1))
            if restore:
                checkpoint = tf.train.latest_checkpoint(SAVE_PATH)
                if checkpoint is not None:
                    saver.restore(sess=sess, save_path=checkpoint)
                restore = False
            train_image = sess.run(image_batch)
            dae_loss = sess.run([loss, optimiser, merged], feed_dict={input: train_image, is_train: True})
            writer.add_summary(dae_loss[2], i)
            if i % 50 == 0:
                test_image = sess.run(test_batch)
                test_out = sess.run(out, feed_dict={input: test_image, is_train: False})
                test_out = ((test_out + 1.) / 2.)  # Keep same scale and floats but remove negatives
                test_out = test_out * 255
                test_out = tf.cast(test_out, dtype=tf.uint8)
                test_out = sess.run(test_out)
                n = 0
                if not os.path.exists(output):
                    os.makedirs(output)
                for img in test_out:
                    path = output + "/" + str(i) + "-" + str(n) + ".png"
                    imageio.imwrite(path, img)
                    n += 1
            if i % 250 == 0:
                if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH)
                saver.save(sess=sess, save_path=SAVE_PATH + "VAE", global_step=i)

            i += 1
        except tf.errors.OutOfRangeError:
            print("Training Complete")
            break
    print("Generating Final Output")
    output = output + "/final_DAE"
    if not os.path.exists(output):
        os.makedirs(output)
    while True:
        try:
            out_batch = sess.run(out_image)
            final_out = sess.run(out, feed_dict={input: out_batch, is_train: False})
            final_out = ((final_out + 1.) / 2.)  # Keep same scale and floats but remove negatives
            final_out = final_out * 255
            final_out = tf.cast(final_out, dtype=tf.uint8)
            final_out = sess.run(final_out)
            n = 0
            for img in final_out:
                path = output + "/" + str(i) + "-" + str(n) + ".png"
                imageio.imwrite(path, img)
                n += 1
        except tf.errors.OutOfRangeError:
            print("Finished Generating output")
            break


if __name__ == "__main__":
    output = "./Project Out/" + VERSION + "/" + DATE
    util.save_params(output, LEARNING_RATE, BATCH_SIZE, DATA_REPEATS, DROPOUT_RATE, EPOCHS, DISTORTED)
    train(output, restore=RESTORE)
