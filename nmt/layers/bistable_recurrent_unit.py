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

        ## Neuromodulated Bistable Recurrent Cell
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


class NBR(keras.layers.RNN):
    """
    Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
        for the recurrent step.
        Default: hard sigmoid (`hard_sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
        the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
    implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of
        smaller dot products and additions, whereas mode 2 will
        batch them into fewer, larger operations. These modes will
        have different performance profiles on different hardware and
        for different applications.
    return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
        in addition to the output.
    go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
    stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
    unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
        If True, the inputs and outputs will be in shape
        `(timesteps, batch, ...)`, whereas in the False case, it will be
        `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
        efficient because it avoids transposes at the beginning and end of the
        RNN calculation. However, most TensorFlow data is batch-major, so by
        default this function accepts input and emits output in batch-major
        form.
    reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before" (default),
        True = "after" (CuDNN compatible).
    Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
    training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
        call of the cell.
    """

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
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.,
                recurrent_dropout=0.,
                return_sequences=False,
                return_state=False,
                go_backwards=False,
                stateful=False,
                unroll=False,
                **kwargs):

        if 'enable_caching_device' in kwargs:
            cell_kwargs = {'enable_caching_device':
                            kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = NBRCell(
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
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
            **cell_kwargs)
        super(NBR, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.input_spec = [keras.layers.InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        return super(NBR, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                keras.activations.serialize(self.activation),
            'recurrent_activation':
                keras.activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                keras.initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                keras.initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                keras.regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                keras.regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                keras.constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                keras.constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                keras.constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout,
        }
        config.update(_config_for_enable_caching_device(self.cell))
        base_config = super(NBR, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config['implementation'] = 1
        return cls(**config)


def _config_for_enable_caching_device(rnn_cell):
    """Return the dict config for RNN cell wrt to enable_caching_device field.
    Since enable_caching_device is a internal implementation detail for speed up
    the RNN variable read when running on the multi remote worker setting, we
    don't want this config to be serialized constantly in the JSON. We will only
    serialize this field when a none default value is used to create the cell.
    Args:
    rnn_cell: the RNN cell for serialize.
    Returns:
    A dict which contains the JSON config for enable_caching_device value or
    empty dict if the enable_caching_device value is same as the default value.
    """
    default_enable_caching_device = tf.executing_eagerly()
    if rnn_cell._enable_caching_device != default_enable_caching_device:
        return {'enable_caching_device': rnn_cell._enable_caching_device}
    return {}

