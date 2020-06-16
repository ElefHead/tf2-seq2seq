import tensorflow as tf
from tensorflow import keras 

from nmt.layers import BahdanauAttention

class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = keras.layers.GRU(self.dec_units, return_sequences=True, 
                return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = keras.layers.Dense(vocab_size)
        
        # used for Attention
        self.attention = BahdanauAttention(self.dec_units)
        
    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        
        x = self.embedding(x)
        x = tf.concat(
            [tf.expand_dims(context_vector, 1), x], 
            axis=-1
            )
        
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        x = self.fc(output)
        
        return x, state, attention_weights
