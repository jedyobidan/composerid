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

MAX_LENGTH = 100
BATCH_SIZE = 128
ORTHO_FEATURES = 46
VALIDATION_FREQUENCY = 10
CHECKPOINT_FREQUENCY = 50
NO_OF_EPOCHS = 8
PRE_ORTHO = False
POST_ORTHO = False
SEQ_LENGTH = QNLS_PER_PHRASE * TOKENS_PER_QNL


## Model class is adatepd from model.py found here
## https://github.com/monikkinom/ner-lstm/
class Model:
    def __init__(self, output_dim, hidden_state_size=300):
        self._output_dim = output_dim
        self._hidden_state_size = hidden_state_size
        self._optimizer = tf.train.AdamOptimizer(0.0005)

    # Adapted from https://github.com/monikkinom/ner-lstm/blob/master/model.py __init__ function
    def create_placeholders(self):
        self._input_feats = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LENGTH, NFEATURES])
        self._output_tags = tf.placeholder(tf.int32, [BATCH_SIZE, Composers.max])

    def set_input_output(self, input_, output):
        self._input_feats = input_
        self._output_tags = output

    def get_mask(self, t):
        t = tf.reduce_max(t, axis=1)
        return tf.cast(tf.not_equal(t, -1), tf.int32)
    
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
                sequence_length=tf.cast(SEQ_LENGTH*np.ones(BATCH_SIZE), tf.int32))
            outputs = outputs_[:,-1,:]
        
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

    # Taken from https://github.com/monikkinom/ner-lstm/blob/master/model.py __init__ function
    def get_train_op(self, loss, global_step):
        training_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, training_vars), 10)
        apply_gradient_op = self._optimizer.apply_gradients(zip(grads, training_vars),
            global_step)
        return apply_gradient_op

        # Adapted from https://github.com/monikkinom/ner-lstm/blob/master/model.py cost function
    def compute_accuracy(self, mask, composers, probabilities):
        # mask = tf.expand_dims(mask, -1)
        predicted_classes = tf.cast(tf.argmax(probabilities, dimension=1), tf.int32)
        actual_classes = tf.cast(tf.argmax(composers, dimension=1), tf.int32)
        correct_predictions = tf.cast(tf.equal(predicted_classes, actual_classes), tf.int32)
        return tf.cast(tf.reduce_sum(tf.multiply(mask, correct_predictions)), tf.float32)

    def compute_grouped_accuracy(self, mask, composer, probabilities):
        mask = tf.cast(tf.expand_dims(mask, -1), tf.float32)
        predicted_classes = tf.cast(tf.argmax(probabilities, dimension=1), tf.int32)
        predicted_classes = tf.cast(tf.one_hot(predicted_classes, Composers.max), tf.int32)
        sum_probability = tf.cast(tf.reduce_sum(tf.multiply(probabilities, mask), axis=0), tf.int32)
        predicted_composer = tf.cast(tf.argmax(sum_probability), tf.int32)
        actual_composer = tf.cast(tf.argmax(composer), tf.int32)
        return tf.cast(tf.equal(predicted_composer, actual_composer), tf.int32)


    # Adapted from https://github.com/monikkinom/ner-lstm/blob/master/model.py cost function
    def cost(self, mask, composers, probabilities):
        mask = tf.expand_dims(mask, -1)
        composers = tf.cast(composers, tf.float32)
        ## masking not needed since pos class vector will be zero for 
        ## padded time steps
        cross_entropy = composers*tf.log(probabilities)
        cross_entropy = tf.multiply(tf.cast(mask, tf.float32), cross_entropy)
        return -tf.reduce_sum(cross_entropy)

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
    
    

# Adapted from http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
def generate_batch(X, y):
    for i in xrange(0, len(X), BATCH_SIZE):
        yield X[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]

def shuffle_data(X, y):
    ran = range(len(X))
    shuffle(ran)
    return [X[num] for num in ran], [y[num] for num in ran]

# Adapted from http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
def generate_epochs(X, y, no_of_epochs):
    lx = len(X)
    lx = (lx//BATCH_SIZE)*BATCH_SIZE
    X = X[:lx]
    y = y[:lx]
    for i in range(no_of_epochs):
        X, y = shuffle_data(X, y)
        yield generate_batch(X, y)

## Compute overall loss and accuracy on dev/test data
def compute_summary_metrics(sess, m, music_feature_val, music_label_val):
    loss, accuracy, total_len = 0.0, 0.0, 0
    for i, epoch in enumerate(generate_epochs(music_feature_val, music_label_val, 1)):
        for step, (X, y) in enumerate(epoch):
            batch_loss, batch_accuracy = sess.run(
                [m.loss, m.accuracy], 
                feed_dict={m.input_feats:X, m.output_tags:y}
            )
            loss += batch_loss
            accuracy += batch_accuracy
            total_len += BATCH_SIZE

    loss = loss/total_len if total_len != 0 else 0
    accuracy = accuracy/total_len if total_len != 0 else 1
    return loss, accuracy

def compute_summary_metrics2(sess, m, music_feature_val, music_label_val):
    accuracy, loss, grouped_accuracy, total_len = 0.0, 0.0, 0.0, 0
    for X, y in zip(music_feature_val, music_label_val):
        if len(X) > BATCH_SIZE:
            X = X[:BATCH_SIZE]
            y = y[:BATCH_SIZE]
        elif len(X) < BATCH_SIZE:
            nops = BATCH_SIZE - len(X)
            X = np.r_[X, np.zeros((nops, SEQ_LENGTH, NFEATURES))]
            y = np.r_[y, -np.ones((nops, Composers.max))]

        batch_loss, batch_accuracy, batch_gaccuracy, batch_length = sess.run(
            [m.loss, m.accuracy, m.grouped_accuracy, m.total_length],
            feed_dict={m.input_feats:X, m.output_tags: y}
        )
        grouped_accuracy += batch_gaccuracy
        accuracy += batch_accuracy
        loss += batch_loss
        total_len += batch_length

    grouped_accuracy = grouped_accuracy / len(music_feature_val)
    loss = loss/total_len
    accuracy = accuracy/total_len

    return loss, accuracy, grouped_accuracy

## train and test adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
## models/image/cifar10/cifar10_train.py and cifar10_eval.py

def train(music_feature_train, music_label_train, music_feature_val, music_label_val, train_dir):
    print "Begin Training"
    m = Model(50)
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        
        ## Add input/output placeholders
        m.create_placeholders()
        ## create the model graph
        m.create_graph()
        ## create training op
        train_op = m.get_train_op(m.loss, global_step)

        ## create saver object which helps in checkpointing
        ## the model
        saver = tf.train.Saver(tf.global_variables()+tf.local_variables())

        ## add scalar summaries for loss, accuracy
        m.add_accuracy_summary()
        m.add_loss_summary()
        summary_op = tf.summary.merge_all()

        ## Initialize all the variables
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto())
        sess.run(init)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
        j = 0
        for i, epoch in enumerate(generate_epochs(music_feature_train, music_label_train, NO_OF_EPOCHS)):
            start_time = time.time()
            for step, (X, y) in enumerate(epoch):
                _, summary_value = sess.run([train_op, summary_op], feed_dict=
                                         {m.input_feats:X, m.output_tags:y})
                duration = time.time() - start_time
                j += 1
                if j % VALIDATION_FREQUENCY == 0:
                    val_loss, val_accuracy, group_accuracy = compute_summary_metrics2(sess, m, music_feature_val, music_label_val)
                    summary = tf.Summary()
                    summary.ParseFromString(summary_value)
                    summary.value.add(tag='Validation Loss', simple_value=val_loss)
                    summary.value.add(tag='Validation Accuracy', simple_value=val_accuracy)
                    summary.value.add(tag='Group Accuracy', simple_value=group_accuracy)
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
def test(sentence_words_test, sentence_tags_test,
         vocab_size, no_pos_classes, train_dir):
    m = Model(vocab_size, MAX_LENGTH, no_pos_classes)
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
            test_loss, test_accuracy, test_oov_accuracy = compute_summary_metrics(sess, m, sentence_words_test,
                                                               sentence_tags_test)
            print 'Test Accuracy: {:.3f}'.format(test_accuracy)
            print 'Test Loss: {:.3f}'.format(test_loss)
            print 'Test OoV Accuracy {:.3f}'.format(test_oov_accuracy)


def pieces2Mat(pieces):
    pieces = [v for p in pieces for v in p.getTrainingExamples()]
    return zip(*pieces)

def pieces2Mat2(pieces):
    pieces = [p.getTrainingExamples() for p in pieces]
    Xs = []
    ys = []
    for p in pieces:
        X, y = zip(*p)
        Xs.append(list(X))
        ys.append(list(y))

    return Xs, ys


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    train_dir = sys.argv[2]
    experiment_type = sys.argv[3]

    training, validation, test = process_dataset(dataset_path)

    X_train, y_train = pieces2Mat(training)
    X_val, y_val = pieces2Mat2(validation)
    # X_test, y_test= pieces2Mat(test)

    print "Training Examples: %d" % len(X_train)

    if experiment_type == 'train':
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.mkdir(train_dir)
        train(X_train, y_train, X_val, y_val, train_dir)
    else:
        test(X_test, y_test, train_dir)
