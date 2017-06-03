from __future__ import print_function

import numpy as np
import tensorflow as tf

from models.model import Model

from utils.data_iterator2 import data_iter, val_iter
from utils.data_utils import get_ex_paths, get_shape
from utils.dice_score import dice_score

from skimage.measure import label
from skimage.morphology import binary_erosion, binary_dilation, ball

import os
import nibabel as nib
from nibabel.testing import data_path

class FCNmodel: #Model):

    def __init__(self, batch_size): #, config):
        # self.config = config
        
        self.batch_size = batch_size
        self.load_data()
        self.add_placeholders()
        self.add_model()
        self.add_loss_op()
        self.add_train_op()
        self.add_pred_op()
        

    def load_data(self):
        self.train_ex_paths = '../' #get_ex_paths(self.config.train_path)
        self.val_ex_paths = '../' #get_ex_paths(self.config.val_path)
        
    def add_placeholders(self):
        self.image_placeholder = tf.placeholder(tf.float32,
                                                shape=[self.batch_size, 160, 160, 144, 4])
        self.label_placeholder = tf.placeholder(tf.int32,
                                                shape=[self.batch_size, 160, 160, 144, 5])
        self.dropout_placeholder = tf.placeholder(tf.float32,
                                                  shape=[])

    def add_model(self):

        
        with tf.variable_scope('conv1') as scope:
            x_downsample = tf.layers.average_pooling3d(self.image_placeholder, pool_size = (2,2,2),
                                            strides = (2,2,2), padding='valid',name=None)
            conv1 = tf.layers.conv3d(inputs=x_downsample, filters=8, 
                                     kernel_size=[5, 5, 5],padding="same", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling3d(inputs = conv1, pool_size = (2,2,2),
                                            strides = (2,2,2), padding='valid',name=None)
        
        with tf.variable_scope('conv2') as scope:
            conv2 = tf.layers.conv3d(inputs=pool1, filters=8, 
                                     kernel_size=[5, 5, 5],padding="same", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling3d(inputs = conv2, pool_size = (2,2,2),
                                            strides = (2,2,2), padding='valid',name=None)
        
        with tf.variable_scope('conv3') as scope:
            conv3 = tf.layers.conv3d(inputs=pool2, filters=32, 
                                     kernel_size=[3, 3, 3],padding="same", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling3d(inputs = conv3, pool_size = (2,2,2),
                                            strides = (2,2,2), padding='valid',name=None)
        
        with tf.variable_scope('conv4') as scope:
            conv4 = tf.layers.conv3d(inputs=pool3, filters=128, 
                                     kernel_size=[3, 3, 3],padding="same", activation=tf.nn.relu)

        with tf.variable_scope('deconv1') as scope:
            W4 = tf.Variable(tf.truncated_normal([3, 3, 3, 16, 128], stddev=0.1))
            deconv4 = tf.nn.conv3d_transpose(conv4, filter = W4, output_shape = [self.batch_size,20, 20, 18, 16], 
                                             strides = [1,2,2,2,1])

            b4 = tf.Variable(tf.constant(0.1, shape=[16]))
            relu4 = tf.nn.relu(deconv4 + b4)
        
        with tf.variable_scope('deconv2') as scope:
            W3 = tf.Variable(tf.truncated_normal([3, 3, 3, 8, 16], stddev=0.1))
            deconv3 = tf.nn.conv3d_transpose(relu4, filter = W3, output_shape = [self.batch_size,40, 40, 36, 8], 
                                             strides = [1,2,2,2,1])
            b3 = tf.Variable(tf.constant(0.1, shape=[8]))
            relu3 = tf.nn.relu(deconv3 + b3)
        
        with tf.variable_scope('deconv3') as scope:
            W2 = tf.Variable(tf.truncated_normal([3, 3, 3, 8, 8], stddev=0.1))
            deconv2 = tf.nn.conv3d_transpose(relu3, filter = W2, output_shape = [self.batch_size,80, 80, 72, 8], 
                                             strides = [1,2,2,2,1])
            b2 = tf.Variable(tf.constant(0.1, shape=[8]))
            relu2 = tf.nn.relu(deconv2 + b2)
        
        with tf.variable_scope('deconv4') as scope:
            W1 = tf.Variable(tf.truncated_normal([3, 3, 3, 5, 8], stddev=0.1))
            deconv1 = tf.nn.conv3d_transpose(relu2, filter = W1, output_shape = [self.batch_size, 160, 160, 144, 5], 
                                             strides = [1,2,2,2,1])
            b1 = tf.Variable(tf.constant(0.1, shape=[5]))
            
            self.score = tf.reshape(deconv1 + b1,[self.batch_size,5*160*160*144])
            

    def add_pred_op(self):
        probs = tf.nn.softmax(tf.reshape(self.score, [-1, 5*160*160*144]))
        reshape_probs = tf.reshape(probs, [-1, 160,160,144,5]) #tf.shape(self.score))

        self.pred = tf.argmax(reshape_probs, 4)
        self.prob = reshape_probs

    def add_loss_op(self):
        # logits = tf.reshape(self.score, [-1, 2])
        # labels = tf.reshape(self.label_placeholder, [-1])
        logits = self.score
        labels = tf.reshape(self.label_placeholder, tf.constant([-1, 5*160*160*144]))
        #ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #          self.score, labels))
        ce_loss = 0
        
        with tf.variable_scope('conv1', reuse=True) as scope:
            w1 = tf.get_variable('weights')
        with tf.variable_scope('conv2', reuse=True) as scope:
            w2 = tf.get_variable('weights')
        with tf.variable_scope('conv3', reuse=True) as scope:
            w3 = tf.get_variable('weights')
        with tf.variable_scope('conv4', reuse=True) as scope:
            w4 = tf.get_variable('weights')
        with tf.variable_scope('deconv1', reuse=True) as scope:
            w1 = tf.get_variable('weights')
        with tf.variable_scope('deconv2', reuse=True) as scope:
            w2 = tf.get_variable('weights')
        with tf.variable_scope('deconv3', reuse=True) as scope:
            w3 = tf.get_variable('weights')
        with tf.variable_scope('deconv4', reuse=True) as scope:
            w4 = tf.get_variable('weights')
            
        reg_loss = self.config.l2 * (tf.nn.l2_loss(w1)
                                   + tf.nn.l2_loss(w2)
                                   + tf.nn.l2_loss(w3)
                                   + tf.nn.l2_loss(w4)
                                   + tf.nn.l2_loss(wfc1)
                                   + tf.nn.l2_loss(wfc2))

        self.loss = ce_loss + reg_loss

    def add_train_op(self):
        self.train = tf.train.AdamOptimizer(
                     learning_rate=self.config.lr).minimize(self.loss) 

    def _train(self, ex_path, sess):
        # train for one epoch
        losses = []
        bdices = []

        bs = self.config.batch_size
        nb = self.config.num_train_batches
        
        # batch_size = 5
        # numIters = np.int(np.ceil(256 / bs))
        
        for batch in range(nb):
            x, y = data_iter(ex_path, bs)
            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: 0.5}

            pred, loss, _ = sess.run([self.pred, self.loss, self.train],
                            feed_dict=feed)

            losses.append(loss)
            bdice = dice_score(y, pred)
            bdices.append(bdice)
            
        return losses, bdices

    def _validate(self, ex_path, sess):
        
        bdices = []

        bs = self.config.batch_size
        nb = self.config.num_val_batches

        for batch in range(nb):
            x, y = val_iter(ex_path, bs)
            
            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: 1.0}

            pred = sess.run(self.pred, feed_dict=feed)

            bdice = dice_score(y, pred)
            bdices.append(bdice)

        return bdices

    def _segment(self, ex_path, sess):
        
        fpred = np.zeros(get_shape(ex_path))
        fy = np.zeros(get_shape(ex_path))
        fprob = np.zeros(get_shape(ex_path) + (2,))

        bs = self.config.batch_size

        for batch, (i, j, k, x, y) in enumerate(data_iter(ex_path, bs)):

            feed = {self.image_placeholder: x,
                    self.label_placeholder: y,
                    self.dropout_placeholder: 1.0}

            pred, prob = sess.run([self.pred, self.prob], feed_dict=feed)

            for idx, _ in enumerate(i):
                fy[i[idx]-4:i[idx]+5,
                   j[idx]-4:j[idx]+5,
                   k[idx]-4:k[idx]+5] = y[idx, :, :, :]
                fpred[i[idx]-4:i[idx]+5,
                      j[idx]-4:j[idx]+5,
                      k[idx]-4:k[idx]+5] = pred[idx, :, :, :]
                fprob[i[idx]-4:i[idx]+5,
                      j[idx]-4:j[idx]+5,
                      k[idx]-4:k[idx]+5, :] = prob[idx, :, :, :, :]

        fdice = dice_score(fy, fpred)

        return fy, fpred, fprob, fdice
