import tensorflow as tf 
from tensorflow import keras 
from tensorflow.errors import InvalidArgumentError
import warnings

class ResidualGRU(keras.layers.GRU):
    def call(self, inputs, mask=None, training=None, initial_state=None):
        outputs = super().call(inputs=inputs, mask=mask, training=training, initial_state=initial_state)
        try:
            if isinstance(outputs, list):
                outputs[0] += inputs
            else: 
                outputs += inputs
        except InvalidArgumentError as e:
            warnings.warn(f"Residual GRU produced a InvalidArgumentError `{str(e)}` - Input and \
            Output shapes don't match. Proceeding with normal GRU output", category=UserWarning, stacklevel=2)
        return outputs

class ResidualLSTM(keras.layers.LSTM):
    def call(self, inputs, mask=None, training=None, initial_state=None):
        outputs = super().call(inputs=inputs, mask=mask, training=training, initial_state=initial_state)
        try:
            if isinstance(outputs, list):
                outputs[0] += inputs
            else: 
                outputs += inputs
        except InvalidArgumentError as e:
            warnings.warn(f"Residual LSTM produced a InvalidArgumentError `{str(e)}` - Input and \
            Output shapes don't match. Proceeding with normal LSTM output", category=UserWarning, stacklevel=2)
        return outputs