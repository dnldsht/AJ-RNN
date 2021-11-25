import tensorflow as tf
from tensorflow.keras import layers
import tf_slim as slim

Missing_value = 128.0

def RNN_cell(type, hidden_size):
	if type == 'LSTM':
		cell = tf.keras.layers.LSTMCell(hidden_size)
	elif type == 'GRU':
		cell = tf.keras.layers.GRUCell(hidden_size)
	return cell

class Discriminator(layers.Layer):
    def call(self, x):
        # - with tf.compat.v1.variable_scope(self.name) as vs:
        x1 = slim.legacy_fully_connected(x = x, num_output_units = x.shape[1], activation_fn= tf.nn.tanh)
        x2 = slim.legacy_fully_connected(x = x1, num_output_units = int(x.shape[1])//2, activation_fn = tf.nn.tanh)
        predict_mask = slim.legacy_fully_connected(x = x2, num_output_units = x.shape[1], activation_fn = tf.nn.sigmoid)
        return predict_mask

class Generator(layers.Layer):

    def __init__(self, config):
        super(Generator, self).__init__()
        self.batch_size = config.batch_size  # configurable
        self.hidden_size = config.hidden_size  # congigurable for GRU/LSTM
        self.num_steps = config.num_steps  # length of input array
        # dimension of input eleemnt array univariate 1
        self.input_dimension_size = config.input_dimension_size
        self.cell_type = config.cell_type  # RNN Cell type
        self.lamda = config.lamda  # coefficient that balances the prediction loss
        self.class_num = config.class_num  # number of targes
        self.layer_num = config.layer_num  # number of layers of AJRNN
        #self.name = 'Generator_LSTM'

        # project layer weight W and bias
        self.W = tf.Variable(tf.random.truncated_normal( [self.hidden_size, self.input_dimension_size], stddev = 0.1 ), dtype = tf.float32, name= 'Project_W')
        self.bias = tf.Variable(tf.constant(0.1,shape = [self.input_dimension_size]), dtype = tf.float32, name= 'Project_bias')

    def call(self, inputs, prediction_target, mask, label_target):
        mulrnn_cell = tf.keras.layers.StackedRNNCells([RNN_cell(type = self.cell_type, hidden_size = self.hidden_size, keep_prob = lstm_keep_prob)	for _ in range(self.layer_num)])

        # initialize state to zero
        init_state = mulrnn_cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        state = init_state

        outputs = list()

        # makes cell run
			  # outputs has list of 'num_steps' with each element's shape (batch_size, hidden_size)
        # - with tf.compat.v1.variable_scope("RNN"):
        for time_step in range(self.num_steps):
            if time_step > 0 : tf.compat.v1.get_variable_scope().reuse_variables()
            if time_step == 0 :
              (cell_output, state) = mulrnn_cell(inputs[:, time_step, :],state)
              outputs.append(cell_output)
            else:
              # comparison has shape (batch_size, self.input_dimension_size) with elements 1 (means missing) when equal or 0 (not missing) otherwise
              comparison = tf.equal( inputs[:, time_step, :], tf.constant( Missing_value ) )
              current_prediction_output = tf.matmul(outputs[time_step - 1], self.W) + self.bias
              #change the current_input, select current_prediction_output when 1 (missing) or use input when 0 (not missing)
              current_input = tf.compat.v1.where(comparison, current_prediction_output, inputs[:,time_step,:])
              (cell_output, state) = mulrnn_cell(current_input, state)
              outputs.append(cell_output)
        # label_target_hidden_output has the last_time_step of shape (batch_size, hidden_size)
        label_target_hidden_output = outputs[-1]

        # prediction_target_hidden_output has list of 'num_steps - 1' with each element's shape (batch_size, hidden_size)
        prediction_target_hidden_output = outputs[:-1]

        #unfolded outputs into the [batch, hidden_size * (numsteps-1)], and then reshape it into [batch * (numsteps-1), hidden_size]
        prediction_hidden_output = tf.reshape( tensor = tf.concat(values = prediction_target_hidden_output, axis = 1), shape = [-1, self.hidden_size] )

        # prediction has shape (batch * (numsteps - 1), self.input_dimension_size)
        prediction = tf.add(tf.matmul(prediction_hidden_output, self.W),self.bias,name='prediction')

        # reshape prediction_target and corresponding mask  into [batch * (numsteps-1), hidden_size]
        prediction_targets = tf.reshape(prediction_target,[-1, self.input_dimension_size])
        masks = tf.reshape( mask,[-1, self.input_dimension_size] )

        # softmax for the label_prediction, label_logits has shape (batch_size, self.class_num)
			  # -- with tf.compat.v1.variable_scope('Softmax_layer'):
        label_logits = slim.legacy_fully_connected(x = label_target_hidden_output, num_output_units = self.class_num)
        loss_classficiation = tf.nn.softmax_cross_entropy_with_logits(labels = tf.stop_gradient( label_target), logits = label_logits, name = 'loss_classficiation')
				

        # use mask to use the observer values for the loss_prediction
        # -with tf.compat.v1.name_scope("loss_prediction"):
        loss_prediction = tf.reduce_mean(input_tensor=tf.square( (prediction_targets - prediction) * masks )) / (self.batch_size)

        # TODO: piangoooooooo 
        #regularization_loss = 0.0

        #for i in self.vars:
        #  regularization_loss += tf.nn.l2_loss(i)

        # -with tf.compat.v1.name_scope("loss_total"):
        loss =  loss_classficiation + self.lamda * loss_prediction # + 1e-4 * regularization_loss

        # for get the classfication accuracy, label_predict has shape (batch_size, self.class_num)
        label_predict = tf.nn.softmax(label_logits, name='test_probab')
        correct_predictions = tf.equal(tf.argmax(input=label_predict,axis=1), tf.argmax(input=label_target,axis=1))
        accuracy = tf.cast(correct_predictions, tf.float32,name='accuracy')

        prediction_res = tf.reshape( prediction, [-1, (self.num_steps - 1)*self.input_dimension_size] ) 
        M = tf.reshape( mask, [-1, (self.num_steps - 1)*self.input_dimension_size])

        return loss, prediction_res, M, accuracy