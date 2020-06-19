import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import backend as K


class NBRCell(keras.layers.GRUCell):
    def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):

        super(NBRCell, self).__init__(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=1,
            reset_after=False,
            **kwargs
        )

    def call(self, inputs, states, training=None):
        ## GRU
        ## r_t = self.recurrent_activation(tf.matmul(U_r, x_t) + tf.matmul(W_r, h_tm1))
        ## z_t = self.recurrent_activation(tf.matmul(U_z, x_t) + tf.matmul(W_z, h_tm1))
        ## h_t = z_t * h_tm1 + (1 - z_t) * self.activation( tf.matmul(U_h, x_t) + r_t * tf.matmul(W_h, h_tm1) )

        ## Neuromodulated Bistable RNN
        ## r_t = 1 + self.activation(tf.matmul(U_r, x_t) + tf.matmul(W_r, h_tm1))
        ## z_t = self.recurrent_activation(tf.matmul(U_z, x_t) + tf.matmul(W_z, h_tm1))
        ## h_t = (z_t * h_tm1) + (1 - z_t)*( self.activation( tf.matmul(U_h, x_t) + r_t * h_tm1 ))

        h_tm1 = states[0] if tf.nest.is_nested(states) else states  # previous memory

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=3)

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = tf.unstack(self.bias)

        if 0. < self.dropout < 1.:
            inputs_z = inputs * dp_mask[0]
            inputs_r = inputs * dp_mask[1]
            inputs_h = inputs * dp_mask[2]
        else:
            inputs_z = inputs
            inputs_r = inputs
            inputs_h = inputs

        x_z = tf.matmul(inputs_z, self.kernel[:, :self.units])
        x_r = tf.matmul(inputs_r, self.kernel[:, self.units:self.units * 2])
        x_h = tf.matmul(inputs_h, self.kernel[:, self.units * 2:])

        if self.use_bias:
            x_z = K.bias_add(x_z, input_bias[:self.units])
            x_r = K.bias_add(x_r, input_bias[self.units: self.units * 2])
            x_h = K.bias_add(x_h, input_bias[self.units * 2:])

        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z = h_tm1 * rec_dp_mask[0]
            h_tm1_r = h_tm1 * rec_dp_mask[1]
            h_tm1_h = h_tm1 * rec_dp_mask[2]
        else:
            h_tm1_z = h_tm1
            h_tm1_r = h_tm1
            h_tm1_h = h_tm1

        recurrent_z = tf.matmul(h_tm1_z, self.recurrent_kernel[:, :self.units])
        recurrent_r = tf.matmul(h_tm1_r,
                            self.recurrent_kernel[:, self.units:self.units * 2])
        if self.reset_after and self.use_bias:
            recurrent_z = K.bias_add(recurrent_z, recurrent_bias[:self.units])
            recurrent_r = K.bias_add(recurrent_r,
                                        recurrent_bias[self.units:self.units * 2])

        ## r_t = 1 + self.activation(tf.matmul(U_r, x_t) + tf.matmul(W_r, h_tm1))
        ## z_t = self.recurrent_activation(tf.matmul(U_z, x_t) + tf.matmul(W_z, h_tm1))

        z = self.recurrent_activation(x_z + recurrent_z)
        r = 1 + self.activation(x_r + recurrent_r)

        ## h_t = (z_t * h_tm1) + (1 - z_t)*( self.activation( tf.matmul(U_h, x_t) + r_t * h_tm1 )) 
        recurrent_h = r *  h_tm1_h

        hh = self.activation( x_h + recurrent_h )

        h = z * h_tm1 + (1 - z) * hh
        return h, [h]
