# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the C3D network.

Implements the inference pattern for model building.
inference_c3d(): Builds the model as far as is required for running the network
forward to make predictions.
"""

import tensorflow as tf

"-----------------------------------------------------------------------------"


def conv3d(name, l_input, w, b):
    return tf.nn.bias_add(
        tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
        b
    )


def max_pool(name, l_input, k):
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1],
                            strides=[1, k, 2, 2, 1], padding='SAME', name=name)


def inference_r3dcnn(_X, _dropout, batch_size, num_clips, hidden_cells, init_state, _weights, _biases):
    """
    r3dcnn network implementation
    c3d part from: https://github.com/hx173149/C3D-tensorflow
    """
    # C3d
    # Convolution Layer
    conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
    conv1 = tf.nn.relu(conv1, 'relu1')
    pool1 = max_pool('pool1', conv1, k=1)

    # Convolution Layer
    conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
    conv2 = tf.nn.relu(conv2, 'relu2')
    pool2 = max_pool('pool2', conv2, k=2)

    # Convolution Layer
    conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
    conv3 = tf.nn.relu(conv3, 'relu3a')
    conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
    conv3 = tf.nn.relu(conv3, 'relu3b')
    pool3 = max_pool('pool3', conv3, k=2)

    # Convolution Layer
    conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
    conv4 = tf.nn.relu(conv4, 'relu4a')
    conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
    conv4 = tf.nn.relu(conv4, 'relu4b')
    pool4 = max_pool('pool4', conv4, k=2)

    # Convolution Layer
    conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
    conv5 = tf.nn.relu(conv5, 'relu5a')
    conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
    conv5 = tf.nn.relu(conv5, 'relu5b')
    pool5 = max_pool('pool5', conv5, k=2)

    # Fully connected layer
    pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
    # Reshape conv3 output to fit dense layer inputs
    dense1 = tf.reshape(pool5, [batch_size*num_clips,
                                _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

    dense1 = tf.nn.relu(dense1, name='fc1')  # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout)
    # Relu activation
    dense2 = tf.nn.relu(tf.matmul(
        dense1, _weights['wd2']) + _biases['bd2'], name='fc2')
    dense2 = tf.nn.dropout(dense2, _dropout)

    # output is on continous stream of image features
    # Feed into RNN

    # recurrent network
    # input should have shape [ batch_size, num_steps, x 4096]

    # reshape to 3D for rnn
    # keep it as one stream
    """ batch major:
    [batch x num_steps x clip]
    along the first dimension we have in each batch all time steps for this
    video if it is reshaped to batch*num_size all elements are stacked after
    each other
    [b1clip1 b1clip2 b1clip3 b2clip1 b2clip2 b2clip3]
    this is still batch major!
    """
    """ time major:
    if we transpose batch and time dimension, we get time major
    [num_steps x batch x clip]
    [[b1clip1, b2clip1], [b1clip2, b2clip2], [b1clip3, b2clip3]]
    unrolled this becomes
    [b1clip1, b2clip1, b1clip2, b2clip2, b1clip3, b2clip3]
    this is clearly not the same as the unrolled batch major
    """
    dense2 = tf.reshape(dense2, [batch_size, num_clips,
                                 _biases['bd2'].get_shape().as_list()[0]])

    # time major transpose here, will save some time
    dense2 = tf.transpose(dense2, (1, 0, 2))

    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_cells, state_is_tuple=True)

    # if is_training and config.keep_prob < 1:
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
        lstm_cell, output_keep_prob=_dropout)
    LAYERS = 1
    # Stacking rnn cells
    stack = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * LAYERS,
                                        state_is_tuple=True)

    # this is still batch major
    outputs, final_state = tf.nn.dynamic_rnn(stack, dense2,
                                             dtype=tf.float32,
                                             time_major=True,
                                             initial_state=init_state)
    # this has to be set for live training


    # unroll again to 2D for last layer
    outputs = tf.reshape(outputs, [-1, hidden_cells])
    logit = tf.matmul(outputs, _weights['out']) + _biases['out']

    return logit, final_state
