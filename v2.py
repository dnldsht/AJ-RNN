from statistics import mode
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import utils
import argparse
from utils import MISSING_VALUE

tf.config.set_visible_devices([], 'GPU')

#
# https://medium.com/dive-into-ml-ai/customization-of-model-fit-to-write-a-combined-discriminator-generator-gan-trainer-in-keras-524bce10cf66

class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

    layer_num = 1 # number of layers of AJRNN
    hidden_size = 100
    learning_rate = 1e-3
    cell_type = 'GRU'
    lamda = 1 # coefficient that balances the prediction loss
    D_epoch = 1
    GPU = '0'
    '''User defined'''
    batch_size = None  # batch_size for train
    epoch = None  # epoch for train
    lamda_D = None  # epoch for training of Discriminator
    G_epoch = None  # epoch for training of Generator
    batches = None
    train_data_filename = None
    test_data_filename = None
    checkpoint_path = None
    save_checkpoint = False
    load_checkpoint = False
    smaller_dataset = False
    verbose = 2


class Discriminator(keras.Sequential):
    def __init__(self, config, *args, **kwargs):
        super().__init__(name='Discriminator', *args, **kwargs, )
        units = (config.num_steps - 1) * config.input_dimension_size

        self.add(keras.Input(shape=(units)))
        self.add(layers.Dense(units, activation='tanh'))
        self.add(layers.Dense(int(units)//2, activation='tanh'))
        self.add(layers.Dense(units, activation='sigmoid'))

class Classifier(keras.Sequential):
    def __init__(self, config, *args, **kwargs):
        super().__init__(name='Classifier', *args, **kwargs, )
        units = config.class_num

        self.add(keras.Input(config.hidden_size))
        self.add(layers.Dense(units))


def RNNCell(type, hidden_size):
    if type == 'LSTM':
        cell = layers.LSTMCell(hidden_size)
    elif type == 'GRU':
        cell = layers.GRUCell(hidden_size)
    return cell


class Generator(layers.Layer):

    def __init__(self, config, *args, **kwargs):
        super().__init__(name='Generator', *args, **kwargs)
        self.batch_size = config.batch_size  # configurable
        self.hidden_size = config.hidden_size  # congigurable for GRU/LSTM
        self.num_steps = config.num_steps  # length of input array
        # dimension of input eleemnt array univariate 1
        self.input_dimension_size = config.input_dimension_size
        self.cell_type = config.cell_type  # RNN Cell type
        self.layer_num = config.layer_num  

        cells = [RNNCell(type=self.cell_type, hidden_size=self.hidden_size) for _ in range(self.layer_num)]
        self.mulrnn_cell = layers.StackedRNNCells(cells)

    def build(self, input_shape):
        
        self.W = self.add_weight(
            "kernel",
            dtype=tf.float32,
            shape=[self.hidden_size, input_shape[-1]],
            initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        )

        self.bias = self.add_weight(
            "bias",
            dtype=tf.float32,
            shape=[input_shape[-1]],
            initializer=tf.keras.initializers.Constant(value=0.1) # not 0?
        )
        

    def call(self, input):


        # initialize state to zero
        init_state = self.mulrnn_cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        state = init_state

        outputs = list()

        # makes cell run
        # outputs has list of 'num_steps' with each element's shape (batch_size, hidden_size)
        
        for time_step in range(self.num_steps):
            if time_step == 0:
                (cell_output, state) = self.mulrnn_cell(input[:, time_step, :], state)
                outputs.append(cell_output)
            else:
                # comparison has shape (batch_size, self.input_dimension_size) with elements 1 (means missing) when equal or 0 (not missing) otherwise
                comparison = tf.equal(input[:, time_step, :], tf.constant(MISSING_VALUE))
                current_prediction_output = tf.matmul(outputs[time_step - 1], self.W) + self.bias

                # change the current_input, select current_prediction_output when 1 (missing) or use input when 0 (not missing)
                current_input = tf.where(comparison, current_prediction_output, input[:, time_step, :])
                (cell_output, state) = self.mulrnn_cell(current_input, state)
                outputs.append(cell_output)

        # last_cell has the last_time_step of shape (batch_size, hidden_size)
        last_cell = outputs[-1]

        # prediction_target_hidden_output has list of 'num_steps - 1' with each element's shape (batch_size, hidden_size)
        prediction_target_hidden_output = outputs[:-1]

        # unfolded outputs into the [batch, hidden_size * (numsteps-1)], and then reshape it into [batch * (numsteps-1), hidden_size]
        prediction_hidden_output = tf.reshape(tensor=tf.concat(values=prediction_target_hidden_output, axis=1), shape=[-1, self.hidden_size])

        # prediction has shape (batch * (numsteps - 1), self.input_dimension_size)
        prediction = tf.add(tf.matmul(prediction_hidden_output, self.W), self.bias)

        return prediction, last_cell


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate):
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
     return self.initial_learning_rate / (step + 1)


class AJRNN(tf.keras.Model):

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(name='AJRNN', *args, **kwargs, )
        self.config = config
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.classifier = Classifier(config)
   
        self.built = True

    def compile(self):
        super(AJRNN, self).compile()

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #         initial_learning_rate=1e-4,
        #         decay_steps=config.batches * config.G_epoch,
        #         decay_rate=0.96)

        self.g_optimizer = tf.keras.optimizers.Adam(0)
        self.d_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.classifier_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.discriminator_loss = tf.keras.metrics.Mean(name="discriminator_loss")
        self.classifier_loss = tf.keras.metrics.Mean(name="classifier_loss")
        self.generator_loss = tf.keras.metrics.Mean(name="generator_loss")
        self.imputation_loss = tf.keras.metrics.Mean(name="imputation_loss")
        self.regularization_loss = tf.keras.metrics.Mean(name="regularization_loss")

        self.accuracy = tf.keras.metrics.Accuracy(name="accuracy")
    
    @property
    def metrics(self):
        return [self.discriminator_loss, self.generator_loss, self.classifier_loss, self.imputation_loss, self.regularization_loss, self.accuracy]
    

    def discriminator_step(self, inputs,mask, training=True):
        for _ in range(self.config.D_epoch):

            dim_size = self.config.input_dimension_size
            num_steps = self.config.num_steps

            prediction, _ = self.generator(inputs)
        
            prediction = tf.reshape(prediction, [-1, (num_steps - 1) * dim_size])
            M = tf.reshape(mask, [-1, (num_steps - 1) * dim_size])


            with tf.GradientTape() as tape:
                predict_M = self.discriminator(prediction)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_M, labels=M)
            
            if training:
                grads = tape.gradient(loss, self.discriminator.trainable_variables)
                self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        
        self.discriminator_loss.update_state(loss)


    def generator_step(self, inputs, prediction_target, mask, label_target, training=True):
        for _ in range(self.config.G_epoch if training else 1):
            dim_size = self.config.input_dimension_size
            num_steps = self.config.num_steps
            batch_size = self.config.batch_size


            with tf.GradientTape() as tape:
                prediction, last_cell = self.generator(inputs)
                
                # reshape prediction_target and corresponding mask  into [batch * (numsteps-1), hidden_size]
                prediction_targets = tf.reshape(prediction_target, [-1, dim_size])
                masks = tf.reshape(mask, [-1, dim_size])

                loss_imputation = tf.square( (prediction_targets - prediction) * masks ) / (self.config.batch_size)
                regularization_loss = 1e-4 * sum(tf.nn.l2_loss(i) for i in self.generator.trainable_weights)

            
                prediction = tf.reshape(prediction, [-1, (num_steps - 1) * dim_size])
                M = tf.reshape(mask, [-1, (num_steps - 1) * dim_size])
                prediction_target = tf.reshape(prediction_target, [-1, (num_steps - 1) * dim_size])
                
                real_pre = prediction * (1 - M) + prediction_target * M
                real_pre = tf.reshape(real_pre, [batch_size, (num_steps-1)*dim_size])

                predict_M = self.discriminator(real_pre)
                predict_M = tf.reshape(predict_M, [-1, (num_steps-1)*dim_size])
                

                G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_M, labels=1 - M) * (1-M)

                loss_imputation = tf.reshape(loss_imputation, [-1, (num_steps-1) * dim_size])

                #total_G_loss = loss_imputation + G_loss + regularization_loss
                total_G_loss = G_loss + regularization_loss

            if training:
                # update classifier
                # grads = classifier_tape.gradient(loss_classification, self.classifier.trainable_variables)
                # self.classifier_optimizer.apply_gradients(zip(grads, self.classifier.trainable_variables))

                # update generator
                grads = tape.gradient(total_G_loss, self.generator.trainable_variables)
                self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        with tf.GradientTape() as classifier_tape:
                    label_logits = self.classifier(last_cell)
                    loss_classification = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(label_target), logits=label_logits, name='loss_classification')

        if training:
                # update classifier
                grads = classifier_tape.gradient(loss_classification, self.classifier.trainable_variables)
                self.classifier_optimizer.apply_gradients(zip(grads, self.classifier.trainable_variables))

        self.regularization_loss.update_state(regularization_loss)
        self.imputation_loss.update_state(loss_imputation)   
        self.generator_loss.update_state(G_loss)
        self.classifier_loss.update_state(loss_classification)
        
        # for get the classfication accuracy, label_predict has shape (batch_size, self.class_num)
        label_predict = tf.nn.softmax(label_logits, name='test_probab')
        self.accuracy.update_state(tf.argmax(label_target, axis=1), tf.argmax(label_predict, axis=1))
       

        

    def train_step(self, data, training=True):
        inputs, prediction_target, mask, label_target = data

        self.discriminator_step(inputs, mask, training=training)
        self.generator_step(inputs, prediction_target, mask, label_target, training=training)


        return {
            "accuracy": self.accuracy.result(),
            "c_loss": self.classifier_loss.result(),
            "d_loss": self.discriminator_loss.result(),
            "g_loss": self.generator_loss.result(),
            "imp_loss": self.imputation_loss.result(),
            "reg_loss": self.regularization_loss.result()
        }

    def test_step(self, data):
        inputs, prediction_target, mask, label_target = data

        self.generator_step(inputs, prediction_target, mask, label_target, training=False)

        return {"accuracy": self.accuracy.result()}


def main(config: Config):

    print(f"Training w/ {config.train_data_filename}")
    
    train_dataset, val_dataset, test_dataset, num_classes, num_steps, num_bands = utils.load(config.train_data_filename, config.test_data_filename, config.smaller_dataset)

    config.num_steps = num_steps
    config.input_dimension_size = num_bands
    config.class_num = num_classes

    train_dataset = train_dataset.batch(
        config.batch_size, drop_remainder=True)

    validation_dataset = val_dataset.batch(
        config.batch_size, drop_remainder=True)

    config.batches = train_dataset.cardinality().numpy()

    print(f"Config {config.__dict__}")


    model = AJRNN(config)
    model.compile()

    callbacks = []

    if config.save_checkpoint:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=config.checkpoint_path,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1,
                save_best_only=True)
        )

    if config.load_checkpoint:
        print(f"loading weights from {config.checkpoint_path}")
        model.load_weights(config.checkpoint_path)
    
    history = model.fit(train_dataset, 
            epochs=config.epoch,
            validation_data=validation_dataset,
            verbose=config.verbose,
            callbacks=callbacks,
            validation_freq=1)

    print()
    print("History training")
    print(history.history)

    if test_dataset is not None:
        test_dataset = test_dataset.batch(
            config.batch_size, drop_remainder=True)
        print()
        print(f"Test Set:")

        history = model.evaluate(test_dataset, verbose=config.verbose, return_dict=True)
        print(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--lamda_D', type=float, required=True,help='coefficient that adjusts gradients propagated from discriminator')
    parser.add_argument('--G_epoch', type=int, required=True, help='frequency of updating AJRNN in an adversarial training epoch')

    parser.add_argument('--train_data_filename', type=str, required=False, default="SITS")
    parser.add_argument('--test_data_filename', type=str, required=False, default=None)

    parser.add_argument('--layer_num', type=int, required=False, default=1, help='number of layers of AJRNN')
    parser.add_argument('--hidden_size', type=int, required=False, default=100, help='number of hidden units of AJRNN')
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-3)
    parser.add_argument('--cell_type', type=str, required=False, default='GRU', help='should be "GRU" or "LSTM" ')
    parser.add_argument('--lamda', type=float, required=False, default=1, help='coefficient that balances the prediction loss')
    parser.add_argument('--D_epoch', type=int, required=False, default=1, help='frequency of updating dicriminator in an adversarial training epoch')
    parser.add_argument('--GPU', type=str, required=False, default='0', help='GPU to use')

    parser.add_argument('-path', '--checkpoint_path', type=str, required=False, default=None, help='Path of checkpoint model')
    parser.add_argument('-save', '--save_checkpoint', default=False, action='store_true', help='Save model in checkpoint_path')
    parser.add_argument('-load', '--load_checkpoint', default=False, action='store_true', help='Load model from checkpoint')
    parser.add_argument('-small', '--smaller_dataset', default=False, action='store_true', help='Load smaller dataset')
    parser.add_argument('-v', '--verbose', nargs='?', type=int, const=1, default=2, help='Verbose mode')

    config = parser.parse_args()
    main(config)
