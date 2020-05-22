# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import tensorflow as tf

# tf.compat.v1.enable_eager_execution()
import numpy as np
import time
import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

print("Loading the data from the files...\n")
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
print("Done!!!\n")

MAX_LEN = 100

# we will need to tokenize the text as the model trains better on numbers that on strings


train_data.dropna(axis=0)
print("Padding the input sentences with the <start> and the <end> tokens...\n")
for index, row in train_data.iterrows():
    row['text'] = '<start> ' + str(row['text']) + ' /s /s ' + str(row['sentiment']) + ' <end>'
    row['selected_text'] = '<start> ' + str(row['selected_text']) + ' <end>'


for index, row in test_data.iterrows():
    row['text'] = str(row['text'])

print("Done!!!\n")


frames = [train_data, test_data]
combined_data = pd.concat(frames)

print("Tokenizing the input and the selected sentences...\n")

# Now we need to tokenize the texts

tokenizer = Tokenizer(filters='', oov_token=100, lower=True)
tokenizer.fit_on_texts(combined_data['text'])

# print(tokenizer.word_index)
tensor = tokenizer.texts_to_sequences(train_data['text'])

input_tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=MAX_LEN)

# Tokenizing the targets sentences..
target_tensor = tokenizer.texts_to_sequences(train_data['selected_text'])
target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, padding='post', maxlen=MAX_LEN)
print("Done!!!\n")

# Splitting the data in the train and the test set
print("Splitting the data in the train and the test set...\n")
x_train, x_test, y_train, y_test = train_test_split(input_tensor, target_tensor, test_size=0.2)


# Function to visualize the word to index mapping
def visualize(tokenizer, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, tokenizer.index_word[t]))


# visualize(tokenizer, x_train[0])


# Creating the dataset
BUFFER_SIZE = len(x_train)
BATCH_SIZE = 64
steps_per_epoch = len(x_train)
embedding_dims = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1

print("Making the tensorflow type datasets from the data so splitted...\n")
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
print("Done!!!\n")

print("Iterating over the batches...\n")
example_input_batch, example_target_batch = next(iter(dataset))
print("Done!!!\n")


# Writing the encoder decoder model for the dataset
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


encoder = Encoder(vocab_size, embedding_dims, units, BATCH_SIZE)


# sample_hidden = encoder.initialize_hidden_state()
# sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# attention_layer = BahdanauAttention(10)
# attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

# Making the decoder function
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


decoder = Decoder(vocab_size, embedding_dims, units, BATCH_SIZE)

#
# sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
#                                        sample_hidden, sample_output)

print("Defining the optimizer and the loss function...\n")
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

print("Done!!!\n")

# @tf.function
# def train_step(inp, targ, enc_hidden):
#     loss = 0
#     with tf.GradientTape() as tape:
#         enc_output, enc_hidden = encoder(inp, enc_hidden)
#         dec_hidden = enc_hidden
# #         print(tokenizer.word_index['<start>'])
#         dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
#         for t in range(1, targ.shape[1]):
#             predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
#             loss += loss_function(targ[:, t], predictions)
#             dec_input = tf.expand_dims(targ[:, t], 1)

#     batch_loss = (loss / int(targ.shape[1]))
#     variables = encoder.trainable_variables + decoder.trainable_variables
#     gradients = tape.gradient(loss, variables)
#     optimizer.apply_gradients(zip(gradients, variables))

#     return batch_loss

# EPOCHS = 30

# for epoch in range(EPOCHS):
#     print("EPOCH STARTED\n")
#     start = time.time()
#     enc_hidden = encoder.initialize_hidden_state()
#     total_loss = 0

#     for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
#         batch_loss = train_step(inp, targ, enc_hidden)
#         total_loss += batch_loss

#         if batch % 100 == 0:
#             print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
#                                                         batch,
#                                                         batch_loss.numpy()))
#     # saving (checkpoint) the model every 2 epochs
#     if (epoch + 1) % 2 == 0:
#         checkpoint.save(file_prefix = checkpoint_prefix)

#     print('Epoch {} Loss {:.4f}'.format(epoch + 1,
#                                         total_loss / steps_per_epoch))
#     print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def preprocess_sentence(sentence, sentiment):
    return '<start> ' + str(sentence).lower() + ' /s /s ' + str(sentiment).lower() + ' <end>'


def evaluate(sentence, sentiment):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    sentence = preprocess_sentence(sentence, sentiment)
    inputs = []
    for i in sentence.split(' '):
        if i is not '':
            inputs.append(tokenizer.word_index[i])
    # inputs = [tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=MAX_LEN,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    for t in range(MAX_LEN):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += tokenizer.index_word[predicted_id] + ' '

        # print(result)
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


print("Predicting the final result...\n")
results = []
result, sentence = evaluate("I am unable to finish this task", "negative")
print(sentence)
print(result)
# for index, row in test_data.iterrows():
#     result, sentence = evaluate(row['text'], row['sentiment'])
#     results.append(str(result))


# test_data['selected_text'] = results
# test_data[['textID', 'selected_text']].to_csv('submission.csv', index=False)
# print("Done!!!\n")