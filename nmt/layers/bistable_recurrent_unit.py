from tensorflow import keras 
from tensorflow.keras import backend as K
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops


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
               implementation=1,
               reset_after=False,
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
            implementation=implementation,
            reset_after=reset_after,
            **kwargs
        )

    def call(self, inputs, states, training=None):
        h_tm1 = states[0] if nest.is_sequence(states) else states  # previous memory

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=3)

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = array_ops.unstack(self.bias)

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            x_z = K.dot(inputs_z, self.kernel[:, :self.units])
            x_r = K.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
            x_h = K.dot(inputs_h, self.kernel[:, self.units * 2:])

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

            recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel[:, :self.units])
            recurrent_r = K.dot(h_tm1_r,
                                self.recurrent_kernel[:, self.units:self.units * 2])
            if self.reset_after and self.use_bias:
                recurrent_z = K.bias_add(recurrent_z, recurrent_bias[:self.units])
                recurrent_r = K.bias_add(recurrent_r,
                                            recurrent_bias[self.units:self.units * 2])

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            # reset gate applied after/before matrix multiplication
            if self.reset_after:
                recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
                if self.use_bias:
                    recurrent_h = K.bias_add(recurrent_h, recurrent_bias[self.units * 2:])
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = K.dot(r * h_tm1_h,
                                    self.recurrent_kernel[:, self.units * 2:])

            hh = self.activation(x_h + recurrent_h)
        else:
            if 0. < self.dropout < 1.:
                inputs = inputs * dp_mask[0]

            # inputs projected by all gate matrices at once
            matrix_x = K.dot(inputs, self.kernel)
            if self.use_bias:
            # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = K.bias_add(matrix_x, input_bias)

            x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=-1)

            if self.reset_after:
            # hidden state projected by all gate matrices at once
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
            else:
            # hidden state projected separately for update/reset and new
                matrix_inner = K.dot(h_tm1, self.recurrent_kernel[:, :2 * self.units])

            recurrent_z, recurrent_r, recurrent_h = array_ops.split(
                matrix_inner, [self.units, self.units, -1], axis=-1)

            z = self.recurrent_activation(x_z + recurrent_z)
            r = self.recurrent_activation(x_r + recurrent_r)

            if self.reset_after:
                recurrent_h = r * recurrent_h
            else:
                recurrent_h = K.dot(r * h_tm1,
                                self.recurrent_kernel[:, 2 * self.units:])

            hh = self.activation(x_h + recurrent_h)
        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]

    
