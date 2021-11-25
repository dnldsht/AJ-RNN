import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import select
import tf_slim as slim
import utils
import os
import numpy as np
import argparse
from tensorflow import keras
#import keras
from layers import Generator, Discriminator




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

class AJRNN(keras.Model):
  def __init__(
        self,
        config: Config,
    ):
      super(AJRNN, self).__init__()
      self.config = config 
      self.generator = Generator(config)
      self.discriminator = Discriminator()

  def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(AJRNN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

  def train_step(self, data):
    # how do we fucking split??
    inputs, prediction_target, mask, labels = data
    
   # Train the discriminator
    with tf.GradientTape() as tape:
        predictions = self.discriminator(inputs)
        d_loss = self.loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    self.d_optimizer.apply_gradients(
        zip(grads, self.discriminator.trainable_weights)
    )

    # Train the generator (note that we should *not* update the weights
    # of the discriminator)!
    with tf.GradientTape() as tape:
        predictions = self.discriminator(self.generator(random_latent_vectors))
        g_loss = self.loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, self.generator.trainable_weights)
    self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
    return {"d_loss": d_loss, "g_loss": g_loss}



def main(cfg: Config):

    print('Loading data && Transform data--------------------')
    print(cfg.train_data_filename)
    train_data, train_label = utils.load_data(cfg.train_data_filename)

    cfg.num_steps = train_data.shape[1]
    cfg.input_dimension_size = 1

    train_label, num_classes = utils.transfer_labels(train_label)
    cfg.class_num = num_classes

    ajrnn = AJRNN(cfg)

    ajrnn.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    
    ajrnn.fit(train_data, train_label, cfg.batch_size, cfg.epoch)

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--lamda_D', type=float, required=True,
                        help='coefficient that adjusts gradients propagated from discriminator')
    parser.add_argument('--G_epoch', type=int, required=True,
                        help='frequency of updating AJRNN in an adversarial training epoch')
    parser.add_argument('--train_data_filename', type=str, required=True)
    parser.add_argument('--test_data_filename', type=str, required=True)

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
