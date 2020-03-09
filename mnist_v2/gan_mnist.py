#encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime as dt
import numpy as np
import pickle as pkl
from functools import partial

EPOCHS = 3
BATCH_SIZE = 100
LEARNING_RATE = 0.0001
ALPHA = 0.2

def generator(randomData, alpha, reuse=False):
    with tf.variable_scope('GAN/generator', reuse=reuse):
        h1 = tf.layers.dense(randomData, 256, activation=partial(tf.nn.leaky_relu, alpha=alpha))
        d1 = tf.layers.dropout(inputs=h1, rate=0.4)
        h2 = tf.layers.dense(d1, 512, activation=partial(tf.nn.leaky_relu, alpha=alpha))
        d2 = tf.layers.dropout(inputs=h2, rate=0.4)
        o1 = tf.layers.dense(d2, 784, activation=None)
        img = tf.tanh(o1)
        return img

def discriminator(img, alpha, reuse=False):
    with tf.variable_scope('GAN/discriminator', reuse=reuse):
        h1 = tf.layers.dense(img, 128, activation=partial(tf.nn.leaky_relu, alpha=alpha))
        d1 = tf.layers.dropout(inputs=h1, rate=0.4)
        h2 = tf.layers.dense(d1, 64, activation=partial(tf.nn.leaky_relu, alpha=alpha))
        d2 = tf.layers.dropout(inputs=h2, rate=0.4)
        D_logits = tf.layers.dense(d2, 1, activation=None)
        D = tf.nn.sigmoid(D_logits)
        return D, D_logits

if __name__ == '__main__':
    tstamp_s = dt.datetime.now().strftime("%H:%M:%S")
    mnist = input_data.read_data_sets('MNIST_DataSet')
    ph_realData = tf.placeholder(tf.float32, (BATCH_SIZE, 784))
    ph_randomData = tf.placeholder(tf.float32, (None, 100))
    gimage = generator(ph_randomData, ALPHA)
    real_D, real_D_logits = discriminator(ph_realData, ALPHA)
    fake_D, fake_D_logits = discriminator(gimage, ALPHA, reuse=True)

    d_real_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits, labels=tf.ones_like(real_D))
    loss_real = tf.reduce_mean(d_real_xentropy)

    d_fake_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits, labels=tf.zeros_like(fake_D))
    loss_fake = tf.reduce_mean(d_fake_xentropy)

    d_loss = loss_real + loss_fake
    g_xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_D_logits, labels=tf.ones_like(fake_D))
    g_loss = tf.reduce_mean(g_xentropy)

    d_training_parameter = [trainVar for trainVar in tf.trainable_variables() if 'GAN/discriminator/' in trainVar.name]
    g_training_parameter = [trainVar for trainVar in tf.trainable_variables() if 'GAN/generator/' in trainVar.name]
    d_optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_training_parameter)
    g_optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, var_list=g_training_parameter)
    batch = mnist.train.next_batch(BATCH_SIZE)
    save_gimage = []
    save_loss = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(EPOCHS):
            for i in range(mnist.train.num_examples//BATCH_SIZE):
                batch = mnist.train.next_batch(BATCH_SIZE)
                batch_images = batch[0].reshape((BATCH_SIZE, 784))
                batch_images = batch_images * 2 - 1
                batch_z = np.random.uniform(-1,1,size=(BATCH_SIZE, 100))
                sess.run(d_optimize, feed_dict={ph_realData:batch_images, ph_randomData:batch_z})
                sess.run(g_optimize, feed_dict={ph_randomData:batch_z})
            train_loss_d = sess.run(d_loss, {ph_randomData: batch_z, ph_realData:batch_images})
            train_loss_g = g_loss.eval({ph_randomData:batch_z})
            print('{0} Epoch={1}/{2}, DLoss={3:.4F}, GLoss={4:.4F}'.format(dt.datetime.now().strftime("%H:%M:%S"), e+1, EPOCHS, train_loss_d, train_loss_g))
            save_loss.append((train_loss_d, train_loss_g))
            randomData = np.random.uniform(-1,1,size=(25,100))
            gen_samples = sess.run(generator(ph_randomData, ALPHA, True), feed_dict={ph_randomData:randomData})
            save_gimage.append(gen_samples)
    with open('save_gimage.pkl', 'wb') as f:
        pkl.dump(save_gimage, f)
    with open('save_loss.pkl', 'wb') as f:
        pkl.dump(save_loss, f)
    tstamp_e = dt.datetime.now().strftime("%H:%M:%S")
    time1 = dt.datetime.strptime(tstamp_s, "%H:%M:%S")
    time2 = dt.datetime.strptime(tstamp_e, "%H:%M:%S")
    print("開始:{0}, 終了:{1}, 処理時間:{2}".format(tstamp_s, tstamp_e, (time2 - time1)))
