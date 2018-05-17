import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense

def model(src_vocab_size,
          tgt_vocab_size,
          encoder_size,
          decoder_size
        ):
    # Camada de Input do encoder
    encoder_input = Input(shape=(None, src_vocab_size))

    # Encoder
    encoder = LSTM(encoder_size, return_state=True)
    _, encoder_state_h, encoder_state_c = encoder(encoder_input)
    encoder_state = [encoder_state_h, encoder_state_c]

    # Decoder Input
    decoder_input = Input(shape=(None, tgt_vocab_size))

    # Decoder
    decoder = LSTM(decoder_size, return_state=True, return_sequences=True)
    decoder_output, _, _ = decoder(decoder_input, initial_state=encoder_state)

    # Activation
    decoder_dense = Dense(tgt_vocab_size, activation='softmax')
    decoder_output = decoder_dense(decoder_output)

    return Model([encoder_input, decoder_input], decoder_output)

def train(model,
        encoder_input_data,
        decoder_input_data,
        decoder_target_data,
        batch_size,
        epochs
        ):

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2
                )

data_path = 'por.txt'

src_data = []
target_data = []
vocab_src = []
vocab_target = []
vocab_src_size = 0
vocab_target_size = 0

num_samples = 10000

with open(data_path, 'r', encoding='utf-8') as data_file:
    lines = data_file.readlines()
    for line in lines[: min(num_samples, len(lines)-1)]:
        src, tgt = line.replace('\n', '').split('\t')

        src_data.append(src)
        target_data.append(tgt)

        for c in src:
            if c not in vocab_src:
                vocab_src.append(c)
        
        for c in tgt:
            if c not in vocab_target:
                vocab_target.append(c)
        

vocab_src = sorted(vocab_src)
vocab_src_size = len(vocab_src)
vocab_target = sorted(vocab_target)
vocab_target_size = len(vocab_target)
max_input_len = max([len(text) for text in src_data])
max_output_len = max([len(text) for text in target_data])

dict_src = dict([(c, i) for i, c in enumerate(vocab_src)])
dict_target = dict([(c, i) for i, c in enumerate(vocab_target)])  

encoder_input_data = np.zeros((len(src_data), max_input_len, vocab_src_size), dtype='float32')
decoder_input_data = np.zeros((len(target_data), max_output_len, vocab_target_size), dtype='float32')
decoder_target_data = np.zeros((len(target_data), max_output_len, vocab_target_size), dtype='float32')

for i, (data_in, data_out) in enumerate(zip(src_data, target_data)):
    for t, c in enumerate(data_in):
        encoder_input_data[i, t, dict_src[c]] = 1

    for t, c in enumerate(data_out):
        decoder_input_data[i, t, dict_target[c]] = 1

        if t != 0:
            decoder_target_data[i, t-1, dict_target[c]] = 1

modelo = model(vocab_src_size, vocab_target_size, 256, 256)

print(modelo.summary())

train(modelo, encoder_input_data, decoder_input_data, decoder_target_data, 64, 100)