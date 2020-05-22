import pandas as pd
import numpy as np
import tensorflow as tf
import tokenizers
from transformers import *

# Initialising the path variables
training_path = "./train.csv"
test_path = "./test.csv"

# Reading the data from the csv files
train_data = pd.read_csv(training_path)
test_data = pd.read_csv(test_path)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

# Preprocessing the data
import os

TOKENIZE_PATH = './RoBERTA Files/'
MAX_LEN = 100

# Initializing thr token variable for tokenization
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=TOKENIZE_PATH + 'vocab.json',
    merges_file=TOKENIZE_PATH + 'merges.txt',
    lowercase=True,
    add_prefix_space=True
)

train_data.dropna(axis=0)
# For bert model we need to tokenize the data as per our needs
# For tokenizing the data we are using a pretrained tokenizer from the Roberta Hugging Face


# Assuming to be the mazimum length of the tweet be 100 words
MAX_LEN = 100
instances = train_data.shape[0]

# Inititalizing the tokenization arrays
input_ids = np.ones((instances, MAX_LEN), dtype='int32')
attention_mask = np.zeros((instances, MAX_LEN), dtype='int32')
token_type_ids = np.zeros((instances, MAX_LEN), dtype='int32')
start_tokens = np.zeros((instances, MAX_LEN), dtype='int32')
end_tokens = np.zeros((instances, MAX_LEN), dtype='int32')

# Now we will be assigning the values to these empty arrays
for i in range(train_data.shape[0]):
    text1 = " " + " ".join(str(train_data.loc[i, 'text']).split())
    text2 = " ".join(str(train_data.loc[i, 'selected_text']).split())
    idx = text1.find(text2)
    chars = np.zeros((len(text1)))
    chars[idx: idx + len(text2)] = 1
    if text1[idx - 1] == ' ':
        chars[idx - 1] = 1

    enc = tokenizer.encode(text1)
    offsets = []
    idx = 0

    for t in enc.ids:
        w = tokenizer.decode([t])
        offsets.append((idx, idx + len(w)))
        idx += len(w)

    # Start and end tokens
    toks = []
    for i, (a, b) in enumerate(offsets):
        sm = np.sum(chars[a: b])
        if sm > 0:
            toks.append(i)

    s_tok = sentiment_id[train_data.loc[i, 'sentiment']]
    input_ids[i, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
    attention_mask[i, :len(enc.ids) + 5] = 1
    if len(toks) > 0:
        start_tokens[i, toks[0] + 1] = 1
        end_tokens[i, toks[-1] + 1] = 1

# Loading the test data
test_data = pd.read_csv(test_path)

instances2 = test_data.shape[0]
input_ids_test = np.ones((instances2, MAX_LEN), dtype='int32')
attention_mask_test = np.zeros((instances2, MAX_LEN), dtype='int32')
token_type_ids_test = np.zeros((instances2, MAX_LEN), dtype='int32')

for k in range(test_data.shape[0]):
    text1 = " " + " ".join(test_data.loc[k, 'text'].split())
    enc = tokenizer.encode(text1)
    s_tok = sentiment_id[test_data.loc[k, 'sentiment']]
    input_ids_test[k, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
    attention_mask_test[k, :len(enc.ids) + 5] = 1

# Building the model for training
# Here we will be using the pretrained roberta model but we will add a custom question and answer head

from tensorflow.keras.layers import Input


def create_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained('./RoBERTA Files/config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained('./RoBERTA Files/pretrained-roberta-base.h5', config=config)
    x = bert_model(ids, attention_mask=att, token_type_ids=tok)

    x1 = tf.keras.layers.Dropout(0.1)(x[0])
    x1 = tf.keras.layers.Conv1D(1, 1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)

    x2 = tf.keras.layers.Dropout(0.1)(x[0])
    x2 = tf.keras.layers.Conv1D(1, 1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1, x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if (len(a) == 0) & (len(b) == 0): return 0.5
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


from sklearn.model_selection import StratifiedKFold
import tensorflow.keras.backend as K

jac = [];
VER = 'v0';
DISPLAY = 1  # USE display=1 FOR INTERACTIVE
oof_start = np.zeros((input_ids.shape[0], MAX_LEN))
oof_end = np.zeros((input_ids.shape[0], MAX_LEN))
preds_start = np.zeros((input_ids_test.shape[0], MAX_LEN))
preds_end = np.zeros((input_ids_test.shape[0], MAX_LEN))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
for fold, (idxT, idxV) in enumerate(skf.split(input_ids, train_data.sentiment.values)):

    K.clear_session()
    model = create_model()

    sv = tf.keras.callbacks.ModelCheckpoint(
        '%s-roberta-%i.h5' % (VER, fold), monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch')

    model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]],
              [start_tokens[idxT,], end_tokens[idxT,]],
              epochs=3, batch_size=32, verbose=DISPLAY, callbacks=[sv],
              validation_data=([input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]],
                               [start_tokens[idxV,], end_tokens[idxV,]]))

    print('Loading model...')
    model.load_weights('%s-roberta-%i.h5' % (VER, fold))

    print('Predicting OOF...')
    oof_start[idxV,], oof_end[idxV,] = model.predict([input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]],
                                                     verbose=DISPLAY)

    print('Predicting Test...')
    preds = model.predict([input_ids_test, attention_mask_test, token_type_ids_test], verbose=DISPLAY)
    preds_start += preds[0] / skf.n_splits
    preds_end += preds[1] / skf.n_splits

    # DISPLAY FOLD JACCARD
    all = []
    for k in idxV:
        a = np.argmax(oof_start[k,])
        b = np.argmax(oof_end[k,])
        if a > b:
            st = train_data.loc[k, 'text']  # IMPROVE CV/LB with better choice here
        else:
            text1 = " " + " ".join(train_data.loc[k, 'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a - 1:b])
        all.append(jaccard(st, train_data.loc[k, 'selected_text']))
    jac.append(np.mean(all))
    print('>>>> FOLD %i Jaccard =' % (fold + 1), np.mean(all))
    print()

all = []
for k in range(input_ids_test.shape[0]):
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])
    if a > b:
        st = test_data.loc[k, 'text']
    else:
        text1 = " " + " ".join(test_data.loc[k, 'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc.ids[a - 1:b])
    all.append(st)
test_data['selected_text'] = all
test_data[['textID', 'selected_text']].to_csv('submission.csv', index=False)