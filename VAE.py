import tensorflow as tf
import tensorflow.contrib.layers as tfcl
import os
import numpy as np
import imageio
import datetime
import Utilities as util
import Layers_autoencoders as lay

LEARNING_RATE = 2e-4
BATCH_SIZE = 64
HEIGHT = 128
WIDTH = 128
CHANNEL = 3
DATA_REPEATS = 4
EPOCHS = 1000
LATENT_DIM = 64
VERSION = "VAE"
DATE = datetime.datetime.now().strftime("%d-%m-%H-%M-%S")
SAVE_PATH = "./Project Out/" + VERSION + "/saved/"
MODE = 1    # 0 For PNG, 1 for JPEG
DROPOUT_RATE = 0.9
DISTORTED = True
RESTORE = True


def encoder(input, is_train):
    with tf.variable_scope("encoder"):
        input = tf.nn.dropout(input, DROPOUT_RATE)
        enc1 = lay.conv(input, 64, "enc1", is_train)
        enc1 = tf.nn.dropout(enc1, DROPOUT_RATE)
        enc2 = lay.conv(enc1, 128, "enc2", is_train)
        enc2 = tf.nn.dropout(enc2, DROPOUT_RATE)
        enc3 = lay.conv(enc2, 256, "enc3", is_train)
        flat = tf.layers.flatten(enc3)
        mean = tfcl.fully_connected(flat, num_outputs=LATENT_DIM, activation_fn=None)
        std_dev = tf.exp(tfcl.fully_connected(flat, num_outputs=LATENT_DIM, activation_fn=None))
        return mean, std_dev


def generateLatentSpace(mean, std_dev):
    with tf.variable_scope("latent"):
        eps = tf.random_normal(shape=tf.shape(std_dev))
        return mean + std_dev * eps


def decoder(encoding, is_train):
    with tf.variable_scope("decoder"):
        enc = tf.layers.dense(encoding, units=65536)
        enc = tf.reshape(enc, [-1, 16, 16, 256])
        dec1 = lay.conv_t(enc, 256, "dec1", is_train)
        dec2 = lay.conv_t(dec1, 128, "dec2", is_train)
        return lay.conv_t(dec2, 3, "dec3", is_train)


def train(output, restore):
    dataset, size = util.read_images(DATA_REPEATS, MODE)
    size = size * EPOCHS
    dataset = dataset.map(lambda path: util.parse_image(path, MODE, HEIGHT, WIDTH, CHANNEL, DISTORTED))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    image_batch = iterator.get_next()

    input = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL])
    is_train = tf.placeholder(tf.bool)

    mean, std_dev = encoder(input, is_train)
    latent_space = generateLatentSpace(mean, std_dev)
    out = decoder(latent_space, is_train)

    # reconstruction_loss = tf.reduce_
    reconstruction_loss = tf.reduce_sum(tf.square(input-out))
    k_loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.square(std_dev) - 1. - tf.log(tf.square(std_dev)))
    loss = tf.reduce_mean(k_loss + reconstruction_loss)
    optimiser = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if not os.path.exists("./log/" + VERSION + DATE):
        os.makedirs("./log/" + VERSION + DATE)
    writer = tf.summary.FileWriter("./log/" + VERSION + " " + DATE)

    loss_summary = tf.summary.scalar("loss", loss)
    writer.add_graph(sess.graph)
    merged = tf.summary.merge_all()
    success = True
    i = 0
    while True:
        try:
            print("Running iteration {}/{}...".format(i, size // BATCH_SIZE + 1))
            if restore:
                checkpoint = tf.train.latest_checkpoint(SAVE_PATH)
                if checkpoint is not None:
                    saver.restore(sess=sess, save_path=checkpoint)
                restore = False
            train_image = sess.run(image_batch)
            vae_loss = sess.run([loss, optimiser, merged], feed_dict={input: train_image, is_train: True})
            writer.add_summary(vae_loss[2], i)
            print(vae_loss[0])
            if i % 50 == 0:
                test_enc = np.random.standard_normal(size=(64, 128, 128, 3))
                test_out = sess.run(out, feed_dict={input: test_enc, is_train: False})
                test_out = ((test_out + 1.) / 2.)  # Keep same scale and floats but remove negatives
                test_out = test_out * 255
                test_out = tf.cast(test_out, dtype=tf.uint8)
                test_out = sess.run(test_out)
                n = 0
                for img in test_out:
                    path = output + "/" + str(i) + "-" + str(n) + ".png"
                    imageio.imwrite(path, img)
                    n += 1
                if not os.path.exists(output + "/grid"):
                    os.makedirs(output + "/grid")
                path = output + "/grid/" + str(i) + ".png"
                imageio.imwrite(path, util.create_grid(test_out, HEIGHT, WIDTH, BATCH_SIZE))

            if i % 250 == 0:
                if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH)
                saver.save(sess=sess, save_path=SAVE_PATH+"VAE", global_step=i)

            i += 1
        except tf.errors.OutOfRangeError:
            print("Training Complete")
            break
        except tf.errors.InvalidArgumentError:
            print("Training Ended: Weights have exploded")
            success = False
            break
    if success:
        print("Generating Final Output")
        output = output + "/final_" + VERSION
        if not os.path.exists(output):
            os.makedirs(output)
        for x in range(5):
            try:
                out_enc = np.random.standard_normal(size=(64, 128, 128, 3))
                final_out = sess.run(out, feed_dict={input: out_enc, is_train: False})
                final_out = ((final_out + 1.) / 2.)  # Keep same scale and floats but remove negatives
                final_out = final_out * 255
                final_out = tf.cast(final_out, dtype=tf.uint8)
                final_out = sess.run(final_out)
                n = 0
                for img in final_out:
                    path = output + "/" + str(i) + "-" + str(n) + ".png"
                    imageio.imwrite(path, img)
                    n += 1
                if not os.path.exists(output + "/grid"):
                    os.makedirs(output + "/grid")
                path = output + "/grid/" + str(i) + ".png"
                imageio.imwrite(path, util.create_grid(test_out, HEIGHT, WIDTH, BATCH_SIZE))
            except tf.errors.OutOfRangeError:
                print("Finished Generating output")
                break


if __name__ == "__main__":
    output = "./Project Out/" + VERSION + "/" + DATE
    util.save_params(output, LEARNING_RATE, BATCH_SIZE, DATA_REPEATS, DROPOUT_RATE, EPOCHS, DISTORTED)
    train(output, RESTORE)
