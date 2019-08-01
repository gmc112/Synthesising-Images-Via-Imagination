import tensorflow as tf
import tensorflow.contrib.layers as tfcl
import os
import numpy as np
import imageio
import datetime
import Utilities as util
import Layers_gan as lay


LEARNING_RATE = 2e-4
BATCH_SIZE = 64
HEIGHT = 128
WIDTH = 128
CHANNEL = 3
DATA_REPEATS = 4
EPOCHS = 1000
VERSION = "GAN_Extra_Layer"
DATE = datetime.datetime.now().strftime("%d-%m-%H-%M-%S")
SAVE_PATH = "./Project Out/" + VERSION + "/saved/"
MODE = 1  # 0 For PNG, 1 for JPEG
DROPOUT_RATE = 0.9
DISTORTED = True
RESTORE = True


# Same model as Gan 1 but restructured with dropout added
# MODELS ARE TAKEN FROM EXAMPLE https://www.youtube.com/watch?v=yz6dNf7X7SA
def generator(input, random_dim, is_train, reuse=False):
    c2, c4, c8, c16, c32, c64 = 1024, 512, 256, 128, 64, 32  # channel num
    s4 = 2
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()

        w1 = tf.get_variable('w0', shape=[random_dim, s4 * s4 * c2], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b0', shape=[c2 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv0 = tf.add(tf.matmul(input, w1), b1, name='flat_conv0')
        # Convolution, bias, activation, repeat!
        with tf.name_scope("gen0"):
            conv0 = tf.reshape(flat_conv0, shape=[-1, s4, s4, c2], name='conv0')
            bn0 = tfcl.batch_norm(conv0, is_training=is_train, epsilon=1e-5, decay=0.9,
                                               updates_collections=None, scope='bn0')
            act0 = tf.nn.relu(bn0, name='act0')
            tf.summary.histogram("conv_t_act", act0)
        act1 = lay.conv_t(act0, c4, "gen1", is_train, False)
        # 8*8*256
        # Convolution, bias, activation, repeat!
        act2 = lay.conv_t(act1, c8, "gen2", is_train, False)
        # 16*16*128
        act3 = lay.conv_t(act2, c16, "gen3", is_train, False)
        # 32*32*64
        act4 = lay.conv_t(act3, c32, "gen4", is_train, False)
        # 64*64*32
        act5 = lay.conv_t(act4, c64, "gen5", is_train, False)
        # 128*128*3
        return lay.conv_t(act5, output_dim,"gen6", is_train, True)





def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        input = lay.dropout(input, DROPOUT_RATE, is_train)
        act1 = lay.conv(input, c2, "dis1", is_train)

        act1 = lay.dropout(act1, DROPOUT_RATE, is_train)
        act2 = lay.conv(act1, c4, "dis2", is_train)

        act2 = lay.dropout(act2, DROPOUT_RATE, is_train)
        act3 = lay.conv(act2, c8, "dis3", is_train)

        act3 = lay.dropout(act3, DROPOUT_RATE, is_train)
        act4 = lay.conv(act3, c16, "dis4", is_train)

        act4 = lay.dropout(act4, DROPOUT_RATE, is_train)
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')

        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')

        return logits


def train(output, restore):
    dataset, size = util.read_images(DATA_REPEATS, MODE)
    size = size * EPOCHS
    dataset = dataset.map(lambda path: util.parse_image(path, MODE, HEIGHT, WIDTH, CHANNEL, DISTORTED))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    image_batch = iterator.get_next()
    iterator_sum = dataset.make_one_shot_iterator()
    im_sum_batch = iterator_sum.get_next()
    random_dim = 100
    with tf.variable_scope('input'):
        real_image = tf.placeholder('float', shape=[None, HEIGHT, WIDTH, CHANNEL], name='real_img')
        random_input = tf.placeholder('float', shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder('bool', name='is_train')
    fake_image = generator(random_input, random_dim, is_train)

    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)

    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    dloss_summary = tf.summary.scalar("dis loss", d_loss)
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.
    gloss_summary = tf.summary.scalar("gen loss", g_loss)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # continue training

    if not os.path.exists("./log/" + VERSION + DATE):
        os.makedirs("./log/" + VERSION + DATE)
    writer = tf.summary.FileWriter("./log/" + VERSION + " " + DATE)

    writer.add_graph(sess.graph)
    merged = tf.summary.merge_all()

    i = 0
    while True:
        try:
            print("Running iteration {}/{}...".format(i, (size // BATCH_SIZE + 1)//6))
            d_iters = 5
            g_iters = 1
            if restore:
                checkpoint = tf.train.latest_checkpoint(SAVE_PATH)
                if checkpoint is not None:
                    saver.restore(sess=sess, save_path=checkpoint)
                restore = False
            for k in range(d_iters):
                train_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, random_dim]).astype(np.float32)
                # sess.run(iterator.initializer)
                train_image = sess.run(image_batch)
                # wgan clip weights
                sess.run(d_clip)

                # Update the discriminator
                _, dloss, m = sess.run([trainer_d, d_loss, merged],
                                       feed_dict={random_input: train_noise, real_image: train_image, is_train: True})
                writer.add_summary(m, i)
            # Update the generator
            for k in range(g_iters):
                train_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, random_dim]).astype(np.float32)
                img = sess.run(im_sum_batch)
                _, gloss, m = sess.run([trainer_g, g_loss, merged],
                                       feed_dict={random_input: train_noise, real_image: img, is_train: True})
                writer.add_summary(m, i)
            # print 'train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss)

            if i % 50 == 0:
                # save images
                print("saving samples")
                if not os.path.exists(output):
                    os.makedirs(output)
                sample_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, random_dim]).astype(np.float32)
                imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})

                imgtest = ((imgtest + 1.) / 2.)  # Keep same scale and floats but remove negatives
                imgtest = imgtest * 255
                imgtest = tf.cast(imgtest, dtype=tf.uint8)
                imgtest = sess.run(imgtest)
                n = 0
                for img in imgtest:
                    path = output + "/" + str(i) + "-" + str(n) + ".png"
                    imageio.imwrite(path, img)
                    n += 1
                if not os.path.exists(output + "/grid"):
                    os.makedirs(output + "/grid")
                path = output + "/grid/" + str(i) + ".png"

                imageio.imwrite(path, util.create_grid(imgtest, HEIGHT, WIDTH, BATCH_SIZE))

                # save_images(imgtest, [8, 8], newPoke_path + '/epoch' + str(i) + '.jpg')
                #
                # print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
                if i % 250 == 0:
                    if not os.path.exists(SAVE_PATH):
                        os.makedirs(SAVE_PATH)
                    saver.save(sess=sess, save_path=SAVE_PATH + "VAE", global_step=i)

            i += 1
        except tf.errors.OutOfRangeError:
            print("Training Complete")
            break
    print("Generating Final Output")
    output = output + "/final_" + VERSION
    if not os.path.exists(output):
        os.makedirs(output)
    for x in range(5):
        try:
            out_enc = np.random.standard_normal(size=(64, 128, 128, 3))
            final_out = sess.run(fake_image, feed_dict={input: out_enc, is_train: False})
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
            imageio.imwrite(path, util.create_grid(imgtest, HEIGHT, WIDTH, BATCH_SIZE))
        except tf.errors.OutOfRangeError:
            print("Finished Generating output")
            break


if __name__ == "__main__":
    output = "./Project Out/" + VERSION + "/" + DATE
    util.save_params(output, LEARNING_RATE, BATCH_SIZE, DATA_REPEATS, DROPOUT_RATE, EPOCHS, DISTORTED)
    train(output, RESTORE)

