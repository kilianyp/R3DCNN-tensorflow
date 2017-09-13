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

# https://github.com/hx173149/C3D-tensorflow
# pylint: disable=missing-docstring
import tensorflow as tf
import sys
import numpy as np
import os


# turn off info logging
# tf.logging.set_verbosity(tf.logging.ERROR)

# numpy pretty printing
np.set_printoptions(suppress=True, precision=5)


class Model(object):

    @property
    def inputs(self):
        """Placeholder for the images."""
        return self._inputs

    @property
    def targets(self):
        """Placeholder for labels."""
        return self._targets

    @property
    def state(self):
        """Placeholder for the state of the RNN."""
        return self._state

    @property
    def final_state(self):
        """Final state of the RNN."""
        return self._final_state

    @property
    def loss(self):
        """A 0-D float32 Tensor."""
        return self._loss

    @property
    def summary(self):
        """A summary Tensor containing information for logging."""
        return self._summary

    @property
    def train_op(self):
        """A list of tensor operations that are executed when training."""
        return self._train_op

    @property
    def ler(self):
        """"Levenshtein distance between input and targets."""
        return self._ler

    @property
    def weights(self):
        """The weights of the C3D network."""
        return self._weights

    @property
    def biases(self):
        """The biases of the C3D network."""
        return self._biases

    @property
    def decoded(self):
        """The decoded tensor through the greedy decoder for the CTC."""
        return self._decoded

    @property
    def options(self):
        """The options this model was initialized with."""
        return self._options

    @property
    def norm_score(self):
        """The probabilites for each label according to the input data."""
        return self._norm_score

    @property
    def name(self):
        """The name of this model."""
        return self._name

    def restore(self, location, sess, variables=None):
        """Restore the variables the model from a specific location."""
        saver = tf.train.Saver(variables)
        if os.path.isfile(location + '.index'):
            saver.restore(sess, location)
            print("\n\n\n", file=sys.stderr)
            print("Restored %s" % location, file=sys.stderr)
        else:
            print("Restore Model %s could not be found. Exiting"
                  % location, file=sys.stderr)
            sys.exit()
        # get last thre parts and substitute
        self._name = "_".join(location.split("/")[-3:])

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def tower_acc(self, logit, labels):
        correct_pred = tf.equal(tf.argmax(logit, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    def _variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, wd):
        var = self._variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var




if __name__ == '__main__':
    tf.app.run()
