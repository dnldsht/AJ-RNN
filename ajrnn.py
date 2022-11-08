from config import Config
import tensorflow as tf
from utils import MISSING_VALUE


class Discriminator(tf.keras.Sequential):
    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(name='Discriminator', *args, **kwargs, )
        units = (config.num_steps - 1) * config.input_dimension_size
        
        self.add(tf.keras.Input(shape=(units)))
        self.add(tf.keras.layers.Dense(units, activation='tanh'))
        self.add(tf.keras.layers.Dense(int(units)//2, activation='tanh'))
        self.add(tf.keras.layers.Dense(units, activation='sigmoid'))

class Classifier(tf.keras.Sequential):
    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(name='Classifier', *args, **kwargs, )
        units = config.class_num

        self.add(tf.keras.Input(config.hidden_size))
        self.add(tf.keras.layers.Dense(units))

def RNNCell(type, hidden_size):
    if type == 'LSTM':
        cell = tf.keras.layers.LSTMCell(hidden_size)
    elif type == 'GRU':
        cell = tf.keras.layers.GRUCell(hidden_size)
    return cell

class Generator(tf.keras.layers.Layer):

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(name='Generator', *args, **kwargs)
        self.batch_size = config.batch_size  # configurable
        self.hidden_size = config.hidden_size  # congigurable for GRU/LSTM
        self.num_steps = config.num_steps  # length of input array
        # dimension of input eleemnt array univariate 1
        self.input_dimension_size = config.input_dimension_size
        self.cell_type = config.cell_type  # RNN Cell type
        self.layer_num = config.layer_num  

        cells = [RNNCell(type=self.cell_type, hidden_size=self.hidden_size) for _ in range(self.layer_num)]
        self.mulrnn_cell = tf.keras.layers.StackedRNNCells(cells)

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
        #       initial_learning_rate=1e-5,
        #       decay_steps=config.batches * config.G_epoch,
        #       decay_rate=0.97,
        #       staircase=True)

        self.g_optimizer = tf.keras.optimizers.Adam(self.config.learning_rate)
        self.d_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.classifier_optimizer = tf.keras.optimizers.Adam(1e-3)

        self.discriminator_loss = tf.keras.metrics.Mean(name="discriminator_loss")
        self.classifier_loss = tf.keras.metrics.Mean(name="classifier_loss")
        self.generator_loss = tf.keras.metrics.Mean(name="generator_loss")
        self.total_generator_loss = tf.keras.metrics.Mean(name="total_generator_loss")
        self.imputation_loss = tf.keras.metrics.Mean(name="imputation_loss")
        self.regularization_loss = tf.keras.metrics.Mean(name="regularization_loss")

        self.accuracy = tf.keras.metrics.Accuracy(name="accuracy")
    
    @property
    def metrics(self):
        return [self.discriminator_loss, self.generator_loss, self.classifier_loss, self.total_generator_loss, self.imputation_loss, self.regularization_loss, self.accuracy]
    

    def discriminator_step(self, inputs,mask, training=True):
        for _ in range(self.config.D_epoch):

            dim_size = self.config.input_dimension_size
            num_steps = self.config.num_steps

            prediction, _ = self.generator(inputs)
        
            prediction = tf.reshape(prediction, [-1, (num_steps - 1) * dim_size])
            M = tf.reshape(mask, [-1, (num_steps - 1) * dim_size])


            with tf.GradientTape() as tape:
                predict_M = self.discriminator(prediction)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_M, labels=M))
            
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

                loss_imputation = tf.reduce_mean(tf.square( (prediction_targets - prediction) * masks )) / (self.config.batch_size)
                regularization_loss = 1e-4 * sum(tf.nn.l2_loss(i) for i in self.generator.trainable_weights)

            
                prediction = tf.reshape(prediction, [-1, (num_steps - 1) * dim_size])
                M = tf.reshape(mask, [-1, (num_steps - 1) * dim_size])
                prediction_target = tf.reshape(prediction_target, [-1, (num_steps - 1) * dim_size])
                
                real_pre = prediction * (1 - M) + prediction_target * M
                real_pre = tf.reshape(real_pre, [batch_size, (num_steps-1)*dim_size])

                predict_M = self.discriminator(real_pre)
                predict_M = tf.reshape(predict_M, [-1, (num_steps-1)*dim_size])
                

                G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_M, labels=1 - M) * (1-M))

                #loss_imputation = tf.reshape(loss_imputation, [-1, (num_steps-1) * dim_size])

                total_G_loss = loss_imputation + G_loss + regularization_loss

            if training:
                # update generator
                grads = tape.gradient(total_G_loss, self.generator.trainable_variables)
                self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        with tf.GradientTape() as tape:
            label_logits = self.classifier(last_cell)

            # NOTE: softmax_cross_entropy_with_logits
            # Measures the probability error in discrete classification tasks in which the classes are mutually exclusive (each entry is in exactly one class)
            loss_classification = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(label_target), logits=label_logits, name='loss_classification')

        if training:
            # update classifier
            grads = tape.gradient(loss_classification, self.classifier.trainable_variables)
            self.classifier_optimizer.apply_gradients(zip(grads, self.classifier.trainable_variables))

        self.regularization_loss.update_state(regularization_loss)
        self.imputation_loss.update_state(loss_imputation)   
        self.generator_loss.update_state(G_loss)
        self.classifier_loss.update_state(loss_classification)
        self.total_generator_loss.update_state(total_G_loss)
        
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
            "reg_loss": self.regularization_loss.result(),
            "total_g_loss": self.total_generator_loss.result()
        }

    def test_step(self, data):
        inputs, prediction_target, mask, label_target = data

        self.generator_step(inputs, prediction_target, mask, label_target, training=False)

        return {"accuracy": self.accuracy.result()}