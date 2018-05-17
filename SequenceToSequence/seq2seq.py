# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np

PAD = 0 
EOS = 1

class Seq2Seq:
  def __init__(self,
              input_vocab_size,
              output_vocab_size,
              encoder_h_units=32,
              decoder_h_units=32):
    
    self.input_vocab_size = input_vocab_size
    self.output_vocab_size = output_vocab_size
    self.encoder_h_units = encoder_h_units
    self.decoder_h_units = decoder_h_units

    self.encoder_inputs = None
    self.encoder_inputs_emb = None
    self.encoder_states = None

    self.decoder_inputs = None
    self.decoder_inputs_emb = None
    self.decoder_outputs = None
    self.decoder_lstm = None
    self.decoder_dense = None

    self.model = None

    self.encoder_predict = None
    self.decoder_predict = None

  def _encoder_inputs(self):
    # Definir os inputs do encoder e aplicação do embedding
    self.encoder_inputs = Input(shape=(None,))
    self.encoder_inputs_emb = Embedding(self.input_vocab_size,
                                        self.encoder_h_units)(self.encoder_inputs)
  
  def _build_encoder(self):
    # Construção do encoder obetendo os estados e ignorando o output
    self._encoder_inputs()
    _, state_h, state_c = LSTM(self.encoder_h_units,
                                return_state=True)(self.encoder_inputs_emb)
    
    self.encoder_states = [state_h, state_c]
  
  def _decoder_inputs(self):
    self.decoder_inputs = Input(shape=(None,))
    self.decoder_inputs_emb = Embedding(self.output_vocab_size,
                                        self.decoder_h_units)(self.decoder_inputs)
  
  def _build_decoder(self):
    # Construção do decoder
    self._decoder_inputs()

    self.decoder_lstm = LSTM(self.decoder_h_units, return_sequences=True)   
    self.decoder_outputs = self.decoder_lstm(self.decoder_inputs_emb,
                                        initial_state=self.encoder_states)

    self.decoder_dense = Dense(self.output_vocab_size, activation='softmax')
    self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
  
  def _build_model(self):
    # Contrução do modelo
    self._build_encoder()
    self._build_decoder()

    self.model = Model([self.encoder_inputs, self.decoder_inputs],
                        self.decoder_outputs)
  
  def _compile_model(self):
    # Compilação do modelo
    self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
  
  def train(self, input_sequences, output_sequences, epochs):
    self._build_model()
    self._compile_model()

    def norm(word_sentences):
      sequences = np.zeros((len(word_sentences), 9, 10))
      for i, sentence in enumerate(word_sentences):
          for j, word in enumerate(sentence):
              sequences[i, j, word] = 1.
      return sequences

    for epoch in range(epochs):
      encoder_inputs = next(input_sequences)
      decoder_inputs = [[EOS] + (sequence) for sequence in encoder_inputs]
      decoder_outputs = [(sequence) + [EOS] for sequence in next(output_sequences)]

      self.model.train_on_batch([helpers.batch(encoder_inputs),
                                  helpers.batch(decoder_inputs)],
                                  norm(helpers.batch(decoder_outputs)))

      if not epoch%10:
        print("Época: ", epoch)
    
    print(self.inference(helpers.batch(next(input_sequences))))
  
  def _build_encoder_predict_model(self):
    self.encoder_predict = Model(self.encoder_inputs, self.encoder_states)

  def _build_decoder_predict_model(self):
    decoder_state_input_h = Input(shape=(self.decoder_h_units,))
    decoder_state_input_c = Input(shape=(self.decoder_h_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = self.decoder_lstm(
      self.decoder_inputs, initial_state=decoder_states_inputs)

    decoder_states = [state_h, state_c]
    decoder_outputs = self.decoder_dense(decoder_outputs)
    self.decoder_predict = Model([self.decoder_inputs_emb] + decoder_states_inputs,
                                 [decoder_outputs] + decoder_states)

  def _inference_setup(self):
    self._build_encoder_predict_model()
    self._build_decoder_predict_model()
  
  def inference(self, sequences):
    self._inference_setup() 
    # Encoda os inputs para vertor de estados
    state_value = self.encoder_predict.predict(sequences)

    # Sequência de saída vazia
    target_seq = np.zeros((1,1, self.output_vocab_size))

    target_seq[0,0,0] = 1

    stop = False
    decoded = ''

    while not stop:
      output, h, c  = self.decoder_predict.predict([target_seq] + state_value)
      
      tgt = np.argmax(output[0, -1, :])
      decoded += str(tgt)

      state_value = [h, c]

      stop = True

    return decoded

if __name__ == "__main__":
  import helpers

  ss = Seq2Seq(10,10)
  batches = helpers.random_sequences(4,8,2,9,20)
  ss.train(batches, batches, 20)