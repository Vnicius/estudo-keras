import re
from math import log

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, dot, Activation, concatenate, Bidirectional, Concatenate, Add


class Seq2Seq:

    def __init__(self,
            src_vocab_size,
            tgt_vocab_size,
            encoder_size,
            decoder_size
            ):
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size

        self.econder_input = None
        self.encoder_emb = None
        self.encoder_states = None
        self.encoder_model = None

        self.decoder = None
        self.decoder_input = None
        self.decoder_emb = None
        self.decoder_dense = None
        self.decoder_model = None

    def model(self):
        # Camada de Input do encoder
        self.econder_input = Input(shape=(None,))
        self.encoder_emb = Embedding(self.src_vocab_size, self.encoder_size)(self.econder_input)

        # Encoder
        encoder = LSTM(self.encoder_size, return_state=True)
        _, encoder_state_h, encoder_state_c = encoder(self.encoder_emb)
        self.encoder_states = [encoder_state_h, encoder_state_c]

        # Decoder Input
        self.decoder_input = Input(shape=(None,))
        self.decoder_emb = Embedding(self.tgt_vocab_size, self.decoder_size)(self.decoder_input)
        # Decoder
        self.decoder = LSTM(self.decoder_size, return_sequences=True, return_state=True)
        decoder_output, _, _ = self.decoder(self.decoder_emb, initial_state=self.encoder_states)

        # Activation
        self.decoder_dense = TimeDistributed(Dense(self.tgt_vocab_size, activation='softmax'))
        decoder_output = self.decoder_dense(decoder_output)

        return Model([self.econder_input, self.decoder_input], decoder_output)

    def model_bidirectional(self):
        # Camada de Input do encoder
        self.econder_input = Input(shape=(None,))
        self.encoder_emb = Embedding(self.src_vocab_size, self.encoder_size)(self.econder_input)

        # Encoder
        encoder = Bidirectional(LSTM(self.encoder_size, return_state=True), merge_mode='sum')
        _, forward_h, forward_c, backward_h, backward_c = encoder(self.encoder_emb)
        encoder_state_h = Concatenate()([forward_h, backward_h])
        encoder_state_c = Concatenate()([forward_c, backward_c])
        self.encoder_states = [encoder_state_h, encoder_state_c]

        # Decoder Input
        self.decoder_input = Input(shape=(None,))
        self.decoder_emb = Embedding(self.tgt_vocab_size, self.decoder_size)(self.decoder_input)
        # Decoder
        self.decoder = LSTM(self.decoder_size * 2, return_sequences=True, return_state=True)
        decoder_output, _, _ = self.decoder(self.decoder_emb, initial_state=self.encoder_states)

        # Activation
        self.decoder_dense = TimeDistributed(Dense(self.tgt_vocab_size, activation='softmax'))
        decoder_output = self.decoder_dense(decoder_output)

        return Model([self.econder_input, self.decoder_input], decoder_output)

    def model_deep(self, deep_size=3):
        # Camada de Input do encoder
        self.econder_input = Input(shape=(None,))
        self.encoder_emb = Embedding(self.src_vocab_size, self.encoder_size)(self.econder_input)

        # Encoder
        seq = self.encoder_emb
        i = 1

        while i < deep_size:
            seq = LSTM(self.encoder_size, return_sequences=True, name='Encoder-'+str(i))(seq)
            i += 1

        encoder = LSTM(self.encoder_size, return_state=True, name='Encoder-'+str(deep_size))
        _, encoder_state_h, encoder_state_c = encoder(seq)

        self.encoder_states = [encoder_state_h, encoder_state_c]

        # Decoder Input
        self.decoder_input = Input(shape=(None,))
        self.decoder_emb = Embedding(self.tgt_vocab_size, self.decoder_size)(self.decoder_input)
        
        # Decoder
        seq = self.decoder_emb
        i = 1

        while i < deep_size:
            seq = LSTM(self.decoder_size, return_sequences=True, name='Decoder-'+str(i))(seq)
            i += 1

        self.decoder = LSTM(self.decoder_size, return_sequences=True, return_state=True, name='Decoder-'+str(deep_size))
        decoder_output, _, _ = self.decoder(seq, initial_state=self.encoder_states)

        # Activation
        self.decoder_dense = TimeDistributed(Dense(self.tgt_vocab_size, activation='softmax'))
        decoder_output = self.decoder_dense(decoder_output)

        return Model([self.econder_input, self.decoder_input], decoder_output)

    def model_bidirectional_deep(self, deep_size=3):
        # Camada de Input do encoder
        self.econder_input = Input(shape=(None,))
        self.encoder_emb = Embedding(self.src_vocab_size, self.encoder_size)(self.econder_input)

       # Encoder
        seq = self.encoder_emb
        i = 1

        while i < deep_size:
            seq = Bidirectional(LSTM(self.encoder_size, return_sequences=True), merge_mode='sum', name='Encoder-'+str(i))(seq)
            i += 1

        encoder = Bidirectional(LSTM(self.encoder_size, return_state=True), merge_mode='sum', name='Encoder-'+str(deep_size))
        _, forward_h, forward_c, backward_h, backward_c = encoder(self.encoder_emb)
        encoder_state_h = Concatenate()([forward_h, backward_h])
        encoder_state_c = Concatenate()([forward_c, backward_c])
        self.encoder_states = [encoder_state_h, encoder_state_c]

        # Decoder Input
        self.decoder_input = Input(shape=(None,))
        self.decoder_emb = Embedding(self.tgt_vocab_size, self.decoder_size)(self.decoder_input)
        seq = self.decoder_emb
        i = 1

        while i < deep_size:
            seq = LSTM(self.encoder_size * 2, return_sequences=True, name='Decoder-'+str(i))(seq)
            i += 1

        self.decoder = LSTM(self.decoder_size * 2, return_sequences=True, return_state=True, name='Decoder-'+str(deep_size))
        decoder_output, _, _ = self.decoder(seq, initial_state=self.encoder_states)

        # Activation
        self.decoder_dense = TimeDistributed(Dense(self.tgt_vocab_size, activation='softmax'))
        decoder_output = self.decoder_dense(decoder_output)

        return Model([self.econder_input, self.decoder_input], decoder_output)

    def model_attention(self):
        # Camada de Input do encoder
        self.econder_input = Input(shape=(None,))
        self.encoder_emb = Embedding(self.src_vocab_size, self.encoder_size)(self.econder_input)

        # Encoder
        encoder = LSTM(self.encoder_size, return_sequences=True, return_state=True)
        encoder_seq, encoder_state_h, encoder_state_c = encoder(self.encoder_emb)
        self.encoder_states = [encoder_state_h, encoder_state_c]

        # Decoder Input
        self.decoder_input = Input(shape=(None,))
        self.decoder_emb = Embedding(self.tgt_vocab_size, self.decoder_size)(self.decoder_input)
        # Decoder
        self.decoder = LSTM(self.decoder_size, return_sequences=True, return_state=True)
        dec, _, _ = self.decoder(self.decoder_emb, initial_state=self.encoder_states)

        attention = dot([dec, encoder_seq], axes=[2, 2])
        attention = Activation('softmax')(attention)

        context = dot([attention, encoder_seq], axes=[2,1])
        decoder_combined_context = concatenate([context, dec])

        # Has another weight + tanh layer as described in equation (5) of the paper
        attention_dense = TimeDistributed(Dense(64, activation="tanh"))
        output = attention_dense(decoder_combined_context) # equation (5) of the paper

        # Activation
        self.decoder_dense = TimeDistributed(Dense(self.tgt_vocab_size, activation='softmax'))
        decoder_output = self.decoder_dense(output)

        return Model([self.econder_input, self.decoder_input], decoder_output)

    def model_bidirectional_att(self):
        # Camada de Input do encoder
        self.econder_input = Input(shape=(None,))
        self.encoder_emb = Embedding(self.src_vocab_size, self.encoder_size)(self.econder_input)

        # Encoder
        encoder = Bidirectional(LSTM(self.encoder_size, return_sequences=True, return_state=True), merge_mode='sum')
        encoder_seq, forward_h, forward_c, backward_h, backward_c = encoder(self.encoder_emb)
        encoder_state_h = Add()([forward_h, backward_h])
        encoder_state_c = Add()([forward_c, backward_c])
        self.encoder_states = [encoder_state_h, encoder_state_c]
        
        # Decoder Input
        self.decoder_input = Input(shape=(None,))
        self.decoder_emb = Embedding(self.tgt_vocab_size, self.decoder_size)(self.decoder_input)
        # Decoder
        self.decoder = LSTM(self.decoder_size, return_sequences=True, return_state=True)
        decoder_output, _, _ = self.decoder(self.decoder_emb, initial_state=self.encoder_states)

        attention = dot([decoder_output, encoder_seq], axes=[2, 2])
        attention = Activation('softmax')(attention)

        context = dot([attention, encoder_seq], axes=[2,1])
        decoder_combined_context = concatenate([context, decoder_output])

        # Has another weight + tanh layer as described in equation (5) of the paper
        attention_dense = TimeDistributed(Dense(64, activation="tanh"))
        output = attention_dense(decoder_combined_context) # equation (5) of the paper

        # Activation
        self.decoder_dense = TimeDistributed(Dense(self.tgt_vocab_size, activation='softmax'))
        decoder_output = self.decoder_dense(output)

        return Model([self.econder_input, self.decoder_input], decoder_output)
    
    def model_deep_att(self, deep_size=3):
        # Camada de Input do encoder
        self.econder_input = Input(shape=(None,))
        self.encoder_emb = Embedding(self.src_vocab_size, self.encoder_size)(self.econder_input)

        # Encoder
        seq = self.encoder_emb
        i = 1

        while i < deep_size:
            seq = LSTM(self.encoder_size, return_sequences=True, name='Encoder-'+str(i))(seq)
            i += 1

        encoder = LSTM(self.encoder_size, return_sequences=True ,return_state=True, name='Encoder-'+str(deep_size))
        encoder_seq, encoder_state_h, encoder_state_c = encoder(seq)

        self.encoder_states = [encoder_state_h, encoder_state_c]

        # Decoder Input
        self.decoder_input = Input(shape=(None,))
        self.decoder_emb = Embedding(self.tgt_vocab_size, self.decoder_size)(self.decoder_input)
        
        # Decoder
        seq = self.decoder_emb
        i = 1

        while i < deep_size:
            seq = LSTM(self.decoder_size, return_sequences=True, name='Decoder-'+str(i))(seq)
            i += 1

        self.decoder = LSTM(self.decoder_size, return_sequences=True, return_state=True, name='Decoder-'+str(deep_size))
        decoder_output, _, _ = self.decoder(seq, initial_state=self.encoder_states)

        attention = dot([decoder_output, encoder_seq], axes=[2, 2])
        attention = Activation('softmax')(attention)

        context = dot([attention, encoder_seq], axes=[2,1])
        decoder_combined_context = concatenate([context, decoder_output])

        # Has another weight + tanh layer as described in equation (5) of the paper
        attention_dense = TimeDistributed(Dense(64, activation="tanh"))
        output = attention_dense(decoder_combined_context) # equation (5) of the paper

        # Activation
        self.decoder_dense = TimeDistributed(Dense(self.tgt_vocab_size, activation='softmax'))
        decoder_output = self.decoder_dense(output)

        return Model([self.econder_input, self.decoder_input], decoder_output)
    
    def model_bidirectional_deep_att(self, deep_size=3):
        # Camada de Input do encoder
        self.econder_input = Input(shape=(None,))
        self.encoder_emb = Embedding(self.src_vocab_size, self.encoder_size)(self.econder_input)

       # Encoder
        seq = self.encoder_emb
        i = 1

        while i < deep_size:
            seq = Bidirectional(LSTM(self.encoder_size, return_sequences=True), merge_mode='sum', name='Encoder-'+str(i))(seq)
            i += 1

        encoder = Bidirectional(LSTM(self.encoder_size, return_sequences=True, return_state=True), merge_mode='sum', name='Encoder-'+str(deep_size))
        encoder_seq, forward_h, forward_c, backward_h, backward_c = encoder(self.encoder_emb)
        encoder_state_h = Add()([forward_h, backward_h])
        encoder_state_c = Add()([forward_c, backward_c])
        self.encoder_states = [encoder_state_h, encoder_state_c]

        # Decoder Input
        self.decoder_input = Input(shape=(None,))
        self.decoder_emb = Embedding(self.tgt_vocab_size, self.decoder_size)(self.decoder_input)
        seq = self.decoder_emb
        i = 1

        while i < deep_size:
            seq = LSTM(self.encoder_size * 2, return_sequences=True, name='Decoder-'+str(i))(seq)
            i += 1

        self.decoder = LSTM(self.decoder_size, return_sequences=True, return_state=True, name='Decoder-'+str(deep_size))
        decoder_output, _, _ = self.decoder(seq, initial_state=self.encoder_states)

        attention = dot([decoder_output, encoder_seq], axes=[2, 2])
        attention = Activation('softmax')(attention)

        context = dot([attention, encoder_seq], axes=[2,1])
        decoder_combined_context = concatenate([context, decoder_output])

        # Has another weight + tanh layer as described in equation (5) of the paper
        attention_dense = TimeDistributed(Dense(64, activation="tanh"))
        output = attention_dense(decoder_combined_context) # equation (5) of the paper

        # Activation
        self.decoder_dense = TimeDistributed(Dense(self.tgt_vocab_size, activation='softmax'))
        decoder_output = self.decoder_dense(output)

        return Model([self.econder_input, self.decoder_input], decoder_output)

    def train(self,
            model,
            encoder_input_data,
            decoder_input_data,
            decoder_target_data,
            batch_size,
            epochs
            ):

        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    shuffle=True,
                    verbose=2
                    )
        # model.save('s2s.h5')
    
    def inference_model(self):
        self.encoder_model = Model(self.econder_input, self.encoder_states)
        
        # Executar primeiro timesep
        decoder_state_input_h = Input(shape=(None,))
        decoder_state_input_c = Input(shape=(None,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_outputs, state_h, state_c = self.decoder(self.decoder_emb, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.decoder_model = Model([self.decoder_input] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)
        #self.decoder_model.summary()
    
    def inference_model_attention(self):
        self.encoder_model = Model(self.econder_input, self.encoder_states)
        
        # Executar primeiro timesep
        decoder_state_input_h = Input(shape=(None,))
        decoder_state_input_c = Input(shape=(None,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_outputs, state_h, state_c = self.decoder(self.decoder_emb, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]

        attention_dot = dot([decoder_outputs, self.encoder_seq], axes=[2, 2])
        attention = Activation('softmax')(attention_dot)

        context = dot([attention_out, self.encoder_seq], axes=[2,1])
        decoder_outputs = concatenate([context, decoder_outputs])

        decoder_outputs = self.attention_dense(decoder_outputs)
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.decoder_model = Model([self.decoder_input] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)
        self.decoder_model.summary()
    
    def predict(self, input_seq, dict_target, vocab_target, max_output_len):
        self.inference_model()
        # Encode da entrada
        states_value = self.encoder_model.predict(input_seq)
        # sequência de saída
        tgt_seq = np.zeros((1,1))
        # adicionar o caracter de start
        tgt_seq[0,0] = dict_target['<GO>']

        decoded_sequence = ''
        stop = False

        while not stop:
            output_tokes, state_h, state_c = self.decoder_model.predict([tgt_seq] + states_value)

            # converter token para caracter
            sampled_token_index = np.argmax(output_tokes[0, 0, :])
            sampled_char = vocab_target[sampled_token_index]
            decoded_sequence += ' ' + sampled_char

            # verificar se chegou no fim da sentença ou no limite de tamanho
            if (sampled_char == '<EOS>' or len(decoded_sequence) > max_output_len):
                stop = True

            # atualizar a entrada do decoder
            tgt_seq = np.zeros((1,1))
            tgt_seq[0,0] = sampled_token_index

            states_value = [state_h, state_c]
        
        return decoded_sequence

    def predict_attention(self, input_seq, dict_target, vocab_target, max_output_len):
        self.inference_model_attention()
        # Encode da entrada
        states_value = self.encoder_model.predict(input_seq)
        # sequência de saída
        tgt_seq = np.zeros((1,1))
        # adicionar o caracter de start
        tgt_seq[0,0] = dict_target['<GO>']

        decoded_sequence = ''
        stop = False

        while not stop:
            output_tokes, state_h, state_c = self.decoder_model.predict([tgt_seq] + states_value)

            # converter token para caracter
            sampled_token_index = np.argmax(output_tokes[0, 0, :])
            sampled_char = vocab_target[sampled_token_index]
            decoded_sequence += ' ' + sampled_char

            # verificar se chegou no fim da sentença ou no limite de tamanho
            if (sampled_char == '<EOS>' or len(decoded_sequence) > max_output_len):
                stop = True

            # atualizar a entrada do decoder
            tgt_seq = np.zeros((1,1))
            tgt_seq[0,0] = sampled_token_index

            states_value = [state_h, state_c]
        
        return decoded_sequence
    
    def predict2(self, model, input_seq, dict_target, vocab_target, max_output_len):
        # sequência de saída
        tgt_seq = np.zeros((1,max_output_len))
        # adicionar o caracter de start
        tgt_seq[0,0] = dict_target['<GO>']

        
        decoded_sequences= []
        
        for i, seq in enumerate(input_seq):
            decoded_sequence = ''
            output_token = model.predict([np.array([seq]), tgt_seq]).argmax(axis=2)
            #input(output_token)
            #input(output_token)
            # converter token para caracter
            
            for cod in output_token[0]:
                sampled_char = vocab_target[cod]
                decoded_sequence += ' ' + sampled_char
            
            decoded_sequences.append(decoded_sequence) 
        return decoded_sequences
    
    def predict_beam(self, model, input_seq, dict_target, vocab_target, max_output_len, k_beam=3):
        self.inference_model()
        # Encode da entrada
        states_value = self.encoder_model.predict(input_seq)
        # sequência de saída
        tgt_seq = np.zeros((1,1))
        # adicionar o caracter de start
        tgt_seq[0,0] = dict_target['<GO>']

        decoded_sequence = ''
        stop = False

        beam_seq = [[list(), 1.0]]

        while not stop:
            output_probs, state_h, state_c = self.decoder_model.predict([tgt_seq] + states_value)
            #output_probs = model.predict([input_seq, tgt_seq])[0, 0, :]
            output_probs = output_probs[0,0,:]
            all_candidates = list()
            
            for i in range(len(beam_seq)):
                prev_seq, score = beam_seq[i]
                for j in range(len(output_probs)):
                    candidate = [prev_seq + [j], score * -log(output_probs[j])]
                    all_candidates.append(candidate)
                
                ordered = sorted(all_candidates, key= lambda tup: tup[1])
                beam_seq = ordered[:k_beam]
            # converter token para caracter
            sampled_char = vocab_target[beam_seq[0][0][-1]]
            decoded_sequence += ' ' + sampled_char

            # verificar se chegou no fim da sentença ou no limite de tamanho
            if (sampled_char == '<EOS>' or len(decoded_sequence) > max_output_len):
                stop = True
        
        return decoded_sequence
    
    def predict_beam_attention(self, model, input_seq, dict_target, vocab_target, max_output_len, k_beam=3):
        # sequência de saída
        tgt_seq = np.zeros((1,max_output_len))
        # adicionar o caracter de start
        tgt_seq[0,0] = dict_target['<GO>']

        decoded_sequences = []

        for s, seq in enumerate(input_seq):
            output_probs = model.predict([np.array([seq]), tgt_seq])
            
            beam_seq = [[list(), 1.0]]
            
            for row in output_probs[0]:
                all_candidates = list()

                for i in range(len(beam_seq)):
                    prev_seq, score = beam_seq[i]
                    #input(beam_seq)
                    for j in range(len(row)):
                        candidate = [prev_seq + [j], score * -log(row[j])]
                        all_candidates.append(candidate)
                    #input(all_candidates)
                    ordered = sorted(all_candidates, key= lambda tup: tup[1])
                    #input(ordered)
                    beam_seq = ordered[:k_beam]
            # converter token para caracter
            
            decoded_sequence = ''

            for cod in beam_seq[0][0]:
                sampled_char = vocab_target[cod]
                decoded_sequence += ' ' + sampled_char
            
            decoded_sequences.append(decoded_sequence) 
        
        return decoded_sequences

data_path = 'por.txt'

pontuacao = re.compile('([\.,\?!])')

src_data = []
target_data = []
vocab_src = ['<PAD>', '<EOS>', '<GO>', '<UNK>', '.', ',', '!', '?', ' ']
vocab_target = ['<PAD>', '<EOS>', '<GO>', '<UNK>', '.', ',', '!', '?', ' ']
vocab_src_size = 0
vocab_target_size = 0

#num_samples = 10100
test_samples = 100
num_samples = 660

# criando dicionários
with open(data_path, 'r', encoding='utf-8') as data_file:
    lines = data_file.readlines()
    for line in lines[: min(num_samples, len(lines)-1)]:
        src, tgt = line.replace('\n', '').split('\t')

        src = pontuacao.sub(r' \1', src)
        tgt = pontuacao.sub(r' \1', tgt)

        src += ' <EOS>' 
        tgt = '<GO> ' + tgt + ' <EOS>'

        src_data.append(src)
        target_data.append(tgt)

        for c in src.split(' '):
            if c not in vocab_src:
                vocab_src.append(c)
        
        for c in tgt.split(' '):
            if c not in vocab_target:
                vocab_target.append(c)

vocab_src_size = len(vocab_src)
vocab_target_size = len(vocab_target)
max_input_len = max([len(text) for text in src_data])
max_output_len = max([len(text) for text in target_data])

dict_src = dict([(c, i) for i, c in enumerate(vocab_src)])
dict_target = dict([(c, i) for i, c in enumerate(vocab_target)])  

encoder_input_data = np.zeros((len(src_data), max_input_len), dtype='float32')
decoder_input_data = np.zeros((len(target_data), max_output_len), dtype='float32')
decoder_target_data = np.zeros((len(target_data), max_output_len, 1), dtype='float32')

# sequências em one-hot
for i, (data_in, data_out) in enumerate(zip(src_data, target_data)):
    for t, c in enumerate(data_in.split(' ')):
        encoder_input_data[i, t] = dict_src[c]

    for t, c in enumerate(data_out.split(' ')):
        decoder_input_data[i, t] = dict_target[c]

        if t != 0:
            decoder_target_data[i, t-1, 0] = dict_target[c]

seq2seq = Seq2Seq(vocab_src_size, vocab_target_size, 256, 256)
#modelo = seq2seq.model()
#modelo = seq2seq.model_bidirectional()
#modelo = seq2seq.model_deep()
#modelo = seq2seq.model_bidirectional_deep()
#modelo = seq2seq.model_attention()
#modelo = seq2seq.model_bidirectional_att()
#modelo = seq2seq.model_deep_att()
modelo = seq2seq.model_bidirectional_deep_att()

modelo.summary()

seq2seq.train(modelo, encoder_input_data[:num_samples-test_samples], decoder_input_data[:num_samples-test_samples], decoder_target_data[:num_samples-test_samples], 64, 100)

for i in range(num_samples-test_samples, num_samples):

    print("In: ", src_data[i])
    print("Target: ", target_data[i])
    #print("Predict_attention: ", seq2seq.predict_attention(encoder_input_data[i:i+1], dict_target, vocab_target, max_output_len))        
    #print("Predict_beam: ", seq2seq.predict_beam(modelo, encoder_input_data[i:i+1], dict_target, vocab_target, max_output_len, 5))    
    print("Predict_beam_attention: ", seq2seq.predict_beam_attention(modelo, encoder_input_data[i:i+1], dict_target, vocab_target, max_output_len, 5))        
    print("Predict2_attention: ", seq2seq.predict2(modelo, encoder_input_data[i:i+1], dict_target, vocab_target, max_output_len))
    #print("Predict: ", seq2seq.predict(encoder_input_data[i:i+1], dict_target, vocab_target, max_output_len))
    print('')
    #input('')