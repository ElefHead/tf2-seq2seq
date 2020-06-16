import tensorflow as tf
from tensorflow import keras 

from nmt.networks import Encoder, Decoder

class NMTModel(keras.Model):
    def __init__(self, encoder: Encoder, decoder: Decoder, batch_sz: int,
                 target_lang, start_token: str = '<start>'):
        super(NMTModel, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder
        self.target_lang = target_lang
        self.start_token = start_token
        self.batch_sz = batch_sz

    def compile(self, optimizer, loss_fn):
        super(NMTModel, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        inp, targ = data
        step_loss = 0

        enc_hidden = self.encoder.initialize_hidden_state(self.batch_sz)
        dec_input = tf.expand_dims([self.target_lang.word_index[self.start_token]] * self.batch_sz, 1)

        with tf.GradientTape() as tape: 
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            
            ## Teacher forcing
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                step_loss += self.loss_fn(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (step_loss / int(targ.shape[1]))

            trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            grads = tape.gradient(step_loss, trainable_variables)
            self.optimizer.apply_gradients(zip(grads, trainable_variables))

        return {"loss": step_loss, "batch_loss": batch_loss}


