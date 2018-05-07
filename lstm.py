import glob
import sys
import tensorflow as tf
import numpy as np
import time
import math
import os
import shutil
import itertools
from datetime import datetime
from random import shuffle
from preprocess import *

from collections import defaultdict

BATCH_SIZE = 1000
VALIDATION_FREQUENCY = 10
CHECKPOINT_FREQUENCY = 50
BATCHES = 10000
SEQ_LENGTH = QNLS_PER_PHRASE * TOKENS_PER_QNL


## Model class is adatepd from model.py found here
## https://github.com/monikkinom/ner-lstm/
class Model:
    def __init__(self, output_dim, hidden_state_size=150):
        self._output_dim = output_dim
        self._hidden_state_size = hidden_state_size
        self._optimizer = tf.train.AdamOptimizer(1e-4)

    # Adapted from https://github.com/monikkinom/ner-lstm/blob/master/model.py __init__ function
    def create_placeholders(self):
        self._input_feats = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LENGTH, NFEATURES])
        self._output_tags = tf.placeholder(tf.int32, [BATCH_SIZE, Composers.max])

    def set_input_output(self, input_, output):
        self._input_feats = input_
        self._output_tags = output

    def get_mask(self, t):
        t = tf.reduce_max(t, axis=1)
        return tf.cast(tf.not_equal(t, 0), tf.int32)
    
    def create_graph(self):
        self.create_placeholders()

        ## Create forward and backward cell
        forward_cell = tf.contrib.rnn.LSTMCell(self._hidden_state_size, state_is_tuple=True)

        ## Embedd the very large input vector into a smaller dimension
        ## This is for computational tractability
        with tf.variable_scope("lstm_input"):
            lstm_input = tf.cast(self._input_feats, tf.float32)
        
        ## Apply bidrectional dyamic rnn to get a tuple of forward
        ## and backward outputs. Using dynamic rnn instead of just 
        ## an rnn avoids the task of breaking the input into 
        ## into a list of tensors (one per time step)
        with tf.variable_scope("lstm"):
            outputs_, _ = tf.nn.dynamic_rnn(
                forward_cell,
                lstm_input,
                dtype=tf.float32,
            )
            outputs = outputs_[:, -1, :]
        
        with tf.variable_scope("lstm_output"):
            
            ## Apply linear transformation to get logits(unnormalized scores)
            logits = self.compute_logits(outputs)

            ## Get the normalized probabilities
            ## Note that this a rank 3 tensor
            ## It contains the probabilities of 
            ## different POS tags for each batch 
            ## example at each time step
            self._probabilities = tf.nn.softmax(logits)

        mask = self.get_mask(self._output_tags)
        length = tf.cast(tf.reduce_sum(mask), tf.int32)

        self._total_length = length
        self._loss = self.cost(mask, self._output_tags, self._probabilities)
        self._accuracy = self.compute_accuracy(mask, self._output_tags, self._probabilities)
        self._composer_accuracy = self.compute_composer_accuracy(mask, self._output_tags, self._probabilities)
        self._composer_lengths = self.compute_composer_lengths(mask, self._output_tags)
        self._avg_composer_accuracy = self._composer_accuracy/self._composer_lengths
        self._average_accuracy = self._accuracy/tf.cast(length, tf.float32)
        self._average_loss = self._loss/tf.cast(length, tf.float32)
        self._grouped_accuracy = self.compute_grouped_accuracy(
            mask,
            self._output_tags[0], 
            self._probabilities
        )

    # Taken from https://github.com/monikkinom/ner-lstm/blob/master/model.py weight_and_bias function
    ## Creates a fully connected layer with the given dimensions and parameters
    def initialize_fc_layer(self, row_dim, col_dim, stddev=0.01, bias=0.1):
        weight = tf.truncated_normal([row_dim, col_dim], stddev=stddev)
        bias = tf.constant(bias, shape=[col_dim])
        return tf.Variable(weight, name='weight'), tf.Variable(bias, name='bias')

    # Taken from https://github.com/monikkinom/ner-lstm/blob/master/model.py __init__ function
    def compute_logits(self, outputs):
        softmax_input_size = int(outputs.get_shape()[1])
        
        W, b = self.initialize_fc_layer(softmax_input_size, self._output_dim)
        
        logits = tf.matmul(outputs, W) + b

        return logits

    def add_loss_summary(self):
        tf.summary.scalar('Loss', self._average_loss)

    def add_accuracy_summary(self):
        tf.summary.scalar('Accuracy', self._average_accuracy)

    def add_composer_summary(self):
        for i, c in enumerate(Composers.objs):
            tf.summary.scalar(c + ' Accuracy', self._avg_composer_accuracy[i])

    # Taken from https://github.com/monikkinom/ner-lstm/blob/master/model.py __init__ function
    def get_train_op(self, loss, global_step):
        training_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, training_vars), 10)
        apply_gradient_op = self._optimizer.apply_gradients(zip(grads, training_vars),
            global_step)
        return apply_gradient_op


    # Adapted from https://github.com/monikkinom/ner-lstm/blob/master/model.py cost function
    def cost(self, mask, composers, probabilities):
        composers = tf.cast(composers, tf.float32)
        ## masking not needed since pos class vector will be zero for 
        ## padded time steps

        cross_entropy = composers*tf.log(probabilities)
        return -tf.reduce_sum(cross_entropy)

        # Adapted from https://github.com/monikkinom/ner-lstm/blob/master/model.py cost function
    def compute_accuracy(self, mask, composers, probabilities):
        # mask = tf.expand_dims(mask, -1)
        mask = tf.cast(mask, tf.float32)
        predicted_classes = tf.cast(tf.argmax(probabilities, dimension=1), tf.int32)
        actual_classes = tf.cast(tf.argmax(composers, dimension=1), tf.int32)
        correct_predictions = tf.cast(tf.equal(predicted_classes, actual_classes), tf.float32)
        return tf.cast(tf.reduce_sum(tf.multiply(mask, correct_predictions)), tf.float32)

    def compute_composer_accuracy(self, mask, composers, probabilities):
        mask = tf.expand_dims(mask, -1)
        predicted_classes = tf.cast(tf.argmax(probabilities, dimension=1), tf.int32)
        predicted_classes = tf.cast(tf.one_hot(predicted_classes, Composers.max), tf.float32)
        composers = tf.cast(composers, tf.float32)
        correct_predictions = tf.cast(tf.equal(2 * predicted_classes - composers, 1), tf.int32)
        correct_predictions = tf.multiply(mask, correct_predictions)

        return tf.cast(tf.reduce_sum(correct_predictions, axis=0), tf.float32)

    def compute_composer_lengths(self, mask, composers):
        mask = tf.expand_dims(mask, -1)
        composers = tf.multiply(mask, composers)
        return tf.cast(tf.reduce_sum(composers, axis=0), tf.float32)

    def compute_grouped_accuracy(self, mask, composer, probabilities):
        mask = tf.cast(tf.expand_dims(mask, -1), tf.float32)
        predicted_classes = tf.cast(tf.argmax(probabilities, dimension=1), tf.int32)
        predicted_classes = tf.cast(tf.one_hot(predicted_classes, Composers.max), tf.int32)
        sum_probability = tf.cast(tf.reduce_sum(tf.multiply(probabilities, mask), axis=0), tf.float32)
        predicted_composer = tf.cast(tf.argmax(sum_probability), tf.int32)
        actual_composer = tf.cast(tf.argmax(composer), tf.int32)
        return tf.cast(tf.equal(predicted_composer, actual_composer), tf.int32)

    @property
    def input_feats(self):
        return self._input_feats

    @property
    def output_tags(self):
        return self._output_tags

    @property
    def loss(self):
        return self._loss

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def grouped_accuracy(self):
        return self._grouped_accuracy

    @property
    def total_length(self):
        return self._total_length

    @property
    def composer_accuracy(self):
        return self._composer_accuracy

    @property
    def composer_lengths(self):
        return self._composer_lengths
    
    

# Adapted from http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html

def flatten(pieces):
    ex = [x for p in pieces for x in p.examples]
    shuffle(ex)
    return ex

def generate_batches(training, nbatches):
    X = {c: flatten(training[c]) for c in training}
    for c in training:
        print "%s: %d examples" % (c, len(X[c]))
    I = {c: 0 for c in training}
    length = BATCH_SIZE // len(training)
    for i in range(nbatches):
        x_batch = np.zeros((BATCH_SIZE, SEQ_LENGTH, NFEATURES))
        y_batch = np.zeros((BATCH_SIZE, Composers.max))
        for cname in training:
            c = Composers.getIndex(cname)
            x_batch[c*length: (c+1)*length, : , :] = X[cname][I[cname]: I[cname]+length]
            y_batch[c*length: (c+1)*length, :] = np.tile(training[cname][0].labelVec(), (length, 1))

            I[cname] += length
            if I[cname]+length > len(X[cname]):
                shuffle(X[cname])
                I[cname] = 0

        yield x_batch, y_batch

def compute_summary_metrics(sess, m, validation):
    accuracy, loss, grouped_accuracy, total_len, piece_ct = 0.0, 0.0, 0.0, 0, 0
    composer_accuracy, composer_lengths = defaultdict(int), defaultdict(int)
    for composer in validation:
        for piece in validation[composer]:
            X = piece.examples
            if len(X) == 0: 
                continue
            y = np.tile(piece.labelVec(), (len(X), 1))
            if len(X) > BATCH_SIZE:
                X = X[:BATCH_SIZE]
                y = y[:BATCH_SIZE]
            elif len(X) < BATCH_SIZE:
                nops = BATCH_SIZE - len(X)
                X = np.r_[X, np.zeros((nops, SEQ_LENGTH, NFEATURES))]
                y = np.r_[y, np.zeros((nops, Composers.max))]

            batch_loss, batch_accuracy, batch_gaccuracy, batch_length = sess.run(
                [m.loss, m.accuracy, m.grouped_accuracy, m.total_length],
                feed_dict={m.input_feats:X, m.output_tags: y}
            )
            grouped_accuracy += batch_gaccuracy
            accuracy += batch_accuracy
            loss += batch_loss
            total_len += batch_length
            composer_accuracy[composer] += batch_accuracy
            composer_lengths[composer] += batch_length
            piece_ct += 1

    grouped_accuracy = grouped_accuracy / piece_ct
    loss = loss/total_len
    accuracy = accuracy/total_len
    for c in composer_accuracy:
        composer_accuracy[c] /= composer_lengths[c]

    return loss, accuracy, grouped_accuracy, composer_accuracy

## train and test adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
## models/image/cifar10/cifar10_train.py and cifar10_eval.py

def train(training, validation, train_dir):
    print "Begin Training"
    m = Model(Composers.max)
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        m.create_placeholders()
        m.create_graph()
        train_op = m.get_train_op(m.loss, global_step)

        ## create saver object which helps in checkpointing
        ## the model
        saver = tf.train.Saver(tf.global_variables()+tf.local_variables())

        ## add scalar summaries for loss, accuracy
        m.add_accuracy_summary()
        m.add_loss_summary()
        m.add_composer_summary()
        summary_op = tf.summary.merge_all()

        ## Initialize all the variables
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto())
        sess.run(init)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
        j = 0
        for (X, y) in generate_batches(training, BATCHES):
            _, summary_value = sess.run([train_op, summary_op], feed_dict=
                                         {m.input_feats:X, m.output_tags:y})
            j += 1
            if j % VALIDATION_FREQUENCY == 0:
                val_loss, val_accuracy, group_accuracy, composer_accuracy = compute_summary_metrics(sess, m, validation)
                summary = tf.Summary()
                summary.ParseFromString(summary_value)
                summary.value.add(tag='Validation Loss', simple_value=val_loss)
                summary.value.add(tag='Validation Accuracy', simple_value=val_accuracy)
                summary.value.add(tag='Group Accuracy', simple_value=group_accuracy)
                for c in composer_accuracy:
                    summary.value.add(tag=c + ' Accuracy', simple_value=composer_accuracy[c])
                summary_writer.add_summary(summary, j)
                log_string = '{} batches ====> Validation Accuracy {:.3f}, Group Accuracy {:.3f}, Validation Loss {:.3f}'
                print log_string.format(j, val_accuracy, group_accuracy, val_loss)
            else:
                summary_writer.add_summary(summary_value, j)

            if j % CHECKPOINT_FREQUENCY == 0:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=j)


## Check performance on held out test data
## Loads most recent model from train_dir
## and applies it on test data
def test(testing, 
         train_dir):
    m = Model(Composers.max)
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        m.create_placeholders()
        m.create_graph()
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            test_loss, test_accuracy, _, _ = compute_summary_metrics(sess, m, testing
                                                               )
            print 'Test Accuracy: {:.3f}'.format(test_accuracy)
            print 'Test Loss: {:.3f}'.format(test_loss)



if __name__ == '__main__':
    dataset_path = sys.argv[1]
    train_dir = sys.argv[2]
    experiment_type = sys.argv[3]

    training, validation, testing = process_dataset(dataset_path)

    if experiment_type == 'train':
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.mkdir(train_dir)
        train(training, validation, train_dir)
    else:
        test(testing, train_dir)
