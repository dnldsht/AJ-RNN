# -*- coding: utf-8 -*-

from statistics import mode
import tensorflow as tf
import utils
import numpy as np
import argparse
from tensorflow.keras import layers
from utils import MISSING_VALUE

from keras.models import load_model

tf.config.set_visible_devices([], 'GPU')


class Config(object):
    layer_num = 1
    hidden_size = 100
    learning_rate = 1e-3
    cell_type = 'GRU'
    lamda = 1
    D_epoch = 1
    GPU = '0'
    '''User defined'''
    batch_size = None  # batch_size for train
    epoch = None  # epoch for train
    lamda_D = None  # epoch for training of Discriminator
    G_epoch = None  # epoch for training of Generator
    train_data_filename = None
    test_data_filename = None
    save = None


def RNN_cell(type, hidden_size, keep_prob):
    if type == 'LSTM':
        cell = tf.keras.layers.LSTMCell(hidden_size)
    elif type == 'GRU':
        cell = tf.keras.layers.GRUCell(hidden_size)
    return cell


class Generator(layers.Layer):

    def __init__(self, config, *args, **kwargs):
        super().__init__(name='Generator_LSTM', *args, **kwargs, )
        self.batch_size = config.batch_size  # configurable
        self.hidden_size = config.hidden_size  # congigurable for GRU/LSTM
        self.num_steps = config.num_steps  # length of input array
        # dimension of input eleemnt array univariate 1
        self.input_dimension_size = config.input_dimension_size
        self.cell_type = config.cell_type  # RNN Cell type
        self.lamda = config.lamda  # coefficient that balances the prediction loss
        self.class_num = config.class_num  # number of targes
        self.layer_num = config.layer_num  # number of layers of AJRNN

        with tf.compat.v1.variable_scope(self.name):
            # project layer weight W and bias
            self.W = tf.Variable(tf.random.truncated_normal(
                [self.hidden_size, self.input_dimension_size], stddev=0.1), dtype=tf.float32, name='Project_W')
            self.bias = tf.Variable(tf.constant(
                0.1, shape=[self.input_dimension_size]), dtype=tf.float32, name='Project_bias')
            # lstm_keep_prob, classfication_keep_prob = 1.0, 1.0
            # construct cells with the specific layer_num
            self.mulrnn_cell = tf.keras.layers.StackedRNNCells([RNN_cell(
                type=self.cell_type, hidden_size=self.hidden_size, keep_prob=1.0) for _ in range(self.layer_num)])

        self.dense1 = layers.Dense(self.class_num)

    def __call__(self, input, prediction_target, mask, label_target):

        with tf.compat.v1.variable_scope(self.name):

            # initialize state to zero
            init_state = self.mulrnn_cell.get_initial_state(
                batch_size=self.batch_size, dtype=tf.float32)
            state = init_state

            outputs = list()

            # makes cell run
            # outputs has list of 'num_steps' with each element's shape (batch_size, hidden_size)
            with tf.compat.v1.variable_scope("RNN"):
                for time_step in range(self.num_steps):
                    if time_step > 0:
                        tf.compat.v1.get_variable_scope().reuse_variables()
                        #pass
                    if time_step == 0:
                        (cell_output, state) = self.mulrnn_cell(
                            input[:, time_step, :], state)
                        outputs.append(cell_output)
                    else:
                        # comparison has shape (batch_size, self.input_dimension_size) with elements 1 (means missing) when equal or 0 (not missing) otherwise
                        comparison = tf.equal(
                            input[:, time_step, :], tf.constant(MISSING_VALUE))
                        current_prediction_output = tf.matmul(
                            outputs[time_step - 1], self.W) + self.bias
                        # change the current_input, select current_prediction_output when 1 (missing) or use input when 0 (not missing)
                        current_input = tf.where(
                            comparison, current_prediction_output, input[:, time_step, :])
                        (cell_output, state) = self.mulrnn_cell(
                            current_input, state)
                        outputs.append(cell_output)

            # label_target_hidden_output has the last_time_step of shape (batch_size, hidden_size)
            label_target_hidden_output = outputs[-1]

            # prediction_target_hidden_output has list of 'num_steps - 1' with each element's shape (batch_size, hidden_size)
            prediction_target_hidden_output = outputs[:-1]

            # unfolded outputs into the [batch, hidden_size * (numsteps-1)], and then reshape it into [batch * (numsteps-1), hidden_size]
            prediction_hidden_output = tf.reshape(tensor=tf.concat(
                values=prediction_target_hidden_output, axis=1), shape=[-1, self.hidden_size])

            # prediction has shape (batch * (numsteps - 1), self.input_dimension_size)
            prediction = tf.add(
                tf.matmul(prediction_hidden_output, self.W), self.bias, name='prediction')

            # reshape prediction_target and corresponding mask  into [batch * (numsteps-1), hidden_size]
            prediction_targets = tf.reshape(
                prediction_target, [-1, self.input_dimension_size])
            masks = tf.reshape(mask, [-1, self.input_dimension_size])

            #  softmax for the label_prediction, label_logits has shape (batch_size, self.class_num)
            with tf.compat.v1.variable_scope('Softmax_layer'):
                label_logits = self.dense1(label_target_hidden_output)
                loss_classficiation = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.stop_gradient(label_target), logits=label_logits, name='loss_classficiation')

        # use mask to use the observer values for the loss_prediction
        with tf.compat.v1.name_scope("loss_prediction"):
            loss_prediction = tf.reduce_mean(
                tf.square((prediction_targets - prediction) * masks)
                ) / (self.batch_size)

        regularization_loss = 0.0
        # TODO better regularization_loss
        for i in self.trainable_weights:
            regularization_loss += tf.nn.l2_loss(i)

        with tf.compat.v1.name_scope("loss_total"):
            loss = loss_classficiation + self.lamda * \
                loss_prediction #+ 1e-4 * regularization_loss

        # for get the classfication accuracy, label_predict has shape (batch_size, self.class_num)
        label_predict = tf.nn.softmax(label_logits, name='test_probab')
        correct_predictions = tf.equal(
            tf.argmax(input=label_predict, axis=1), tf.argmax(input=label_target, axis=1))
        accuracy = tf.cast(correct_predictions, tf.float32, name='accuracy')

        loss_tensors = {
            'loss_prediction': loss_prediction,
            'loss_classficiation': loss_classficiation,
            'regularization_loss': regularization_loss,
            'loss': loss
        }

        prediction = tf.reshape(
            prediction, [-1, (self.num_steps - 1)*self.input_dimension_size])
        M = tf.reshape(mask, [-1, (self.num_steps - 1)
                       * self.input_dimension_size])
        prediction_target = tf.reshape(
            prediction_targets, [-1, (self.num_steps - 1)*self.input_dimension_size])

        return loss_tensors, accuracy, prediction, M, label_predict, prediction_target, label_target_hidden_output


class Discriminator(layers.Layer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(name='Discriminator', *args, **kwargs, )
        units = (config.num_steps - 1) * config.input_dimension_size
        self.l1 = layers.Dense(units, activation='tanh')
        self.l2 = layers.Dense(int(units)//2, activation='tanh')
        self.l3 = layers.Dense(units, activation='sigmoid')

    def __call__(self, x):
        out = self.l1(x)
        out = self.l2(out)
        # predict_mask
        return self.l3(out)


class AJRNN(tf.keras.Model):

    def __init__(self, config, *args, **kwargs):
        super().__init__(name='AJRNN', *args, **kwargs, )
        self.config = config
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

    def call(self, inputs, training=None):
        pass

    def compile(self):
        super(AJRNN, self).compile()
        self.generator_optimizer = tf.keras.optimizers.Adam(
            config.learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            config.learning_rate)
    

    def discriminator_step(self, input, prediction_target, mask, label_target, training=True):
        with tf.GradientTape() as tape:
            loss_tensors, accuracy, prediction, M, label_predict, prediction_target, last_hidden_output = self.generator(
                input, prediction_target, mask, label_target)
            real_pre = prediction * (1 - M) + prediction_target * M
            real_pre = tf.reshape(
                real_pre, [config.batch_size, (config.num_steps-1)*config.input_dimension_size])

            predict_M = self.discriminator(real_pre)

            predict_M = tf.reshape(
                predict_M, [-1, (config.num_steps-1)*config.input_dimension_size])

            D_loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_M, labels=M))

        if training:
            gradients_of_discriminator = tape.gradient(
                D_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def generator_step(self, input, prediction_target, mask, label_target, training=True):
        with tf.GradientTape() as tape:
            loss_tensors, accuracy, prediction, M, label_predict, prediction_target, last_hidden_output = self.generator(
                input, prediction_target, mask, label_target)
                
            real_pre = prediction * (1 - M) + prediction_target * M
            real_pre = tf.reshape(
                real_pre, [config.batch_size, (config.num_steps-1)*config.input_dimension_size])

            predict_M = self.discriminator(real_pre)

            predict_M = tf.reshape(
                predict_M, [-1, (config.num_steps-1)*config.input_dimension_size])

            G_loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                logits=predict_M, labels=1 - M) * (1-M))

            total_G_loss = loss_tensors['loss'] + config.lamda_D * G_loss
        if training:
            gradients_of_generator = tape.gradient(
                total_G_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))
        return loss_tensors['loss'], accuracy

    def train_step(self, data, training=True):
        input, prediction_target, mask, label_target = data
        for _ in range(self.config.D_epoch):
            self.discriminator_step(
                input, prediction_target, mask, label_target, training=training)

        for _ in range(self.config.G_epoch):
            batch_loss, batch_accuracy = self.generator_step(
                input, prediction_target, mask, label_target, training=training)

        return {'loss': batch_loss, 'accuracy': batch_accuracy}

    def test_step(self, data):
        return self.train_step(data, training=False)


def main(config:Config):

    print(f"Training w/ {config.train_data_filename}")

    train_dataset, val_dataset, test_dataset, num_classes, num_steps, num_bands = utils.load(config.train_data_filename, config.test_data_filename)

    config.num_steps = num_steps
    config.input_dimension_size = num_bands
    config.class_num = num_classes


    model = AJRNN(config)
    model.compile()

    train_dataset = train_dataset.shuffle(10).batch(
        config.batch_size, drop_remainder=True)

    validation_dataset = val_dataset.batch(
        config.batch_size, drop_remainder=True)
    model.build(input_shape=(config.batch_size, config.num_steps, config.input_dimension_size))
    model.summary()
    
    history = model.fit(train_dataset, 
            epochs=config.epoch,
            validation_data=validation_dataset,
            verbose=2,
            validation_freq=10)


    model.summary()
    
    print(history.history)
    model.save_weights('sits_weights.h5')  # creates a HDF5 file 'my_model.tf'
    
    
    model.load_weights('sits_weights.h5', by_name = True, skip_mismatch = True)
    
    #print('model.trainable_variables', model.trainable_variables)

    if test_dataset is not None:
        test_dataset = test_dataset.batch(
            config.batch_size, drop_remainder=True)
        print()
        print(f"Test Set:")

        history = model.evaluate(test_dataset, verbose=2, return_dict=True)
        print(history)

    #print('model.trainable_variables', model.trainable_variables)
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--lamda_D', type=float, required=True,
                        help='coefficient that adjusts gradients propagated from discriminator')
    parser.add_argument('--G_epoch', type=int, required=True,
                        help='frequency of updating AJRNN in an adversarial training epoch')
    parser.add_argument('--train_data_filename', type=str, required=False, default="SITS")
    parser.add_argument('--test_data_filename', type=str, required=False, default=None)

    parser.add_argument('--layer_num', type=int, required=False,
                        default=1, help='number of layers of AJRNN')
    parser.add_argument('--hidden_size', type=int, required=False,
                        default=100, help='number of hidden units of AJRNN')
    parser.add_argument('--learning_rate', type=float,
                        required=False, default=1e-3)
    parser.add_argument('--cell_type', type=str, required=False,
                        default='GRU', help='should be "GRU" or "LSTM" ')
    parser.add_argument('--lamda', type=float, required=False, default=1,
                        help='coefficient that balances the prediction loss')
    parser.add_argument('--D_epoch', type=int, required=False, default=1,
                        help='frequency of updating dicriminator in an adversarial training epoch')
    parser.add_argument('--GPU', type=str, required=False,
                        default='0', help='GPU to use')
    parser.add_argument('--save', type=str, required=False,
                        default=None, help='Path where to save model')

    config = parser.parse_args()
    main(config)
