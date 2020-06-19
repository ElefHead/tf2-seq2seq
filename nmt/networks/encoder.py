import tensorflow as tf
from tensorflow import keras

from nmt.layers import ResidualGRU, ResidualLSTM, NBR 


class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru1 = NBR(self.enc_units, return_sequences=True, 
                return_state=False, recurrent_initializer="glorot_uniform")
        self.residual_layer = NBR(self.enc_units, return_sequences=True, return_state=False, recurrent_initializer="glorot_uniform")
        self.gru2 = NBR(self.enc_units, return_sequences=True, 
            return_state=True, recurrent_initializer="glorot_uniform")
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output = self.gru1(x, initial_state=hidden)
        output = self.residual_layer(output)
        output, state = self.gru2(output)
        return output, state
    
    def initialize_hidden_state(self, batch_sz):
        return tf.zeros((batch_sz, self.enc_units))
