# From the top
# In[ ]: Import

import pandas as pd
import numpy as np
import nltk
import pickle
import tensorflow as tf
import string
import re
import matplotlib.pyplot as plt
import math
import RNN
from nltk.corpus import stopwords

# In[ ]: Preprocessing

# Execute this in command line to use tokenize nltk.download()

dataset = pd.read_csv("data/reviews.csv")

# Rename names
dataset = dataset.rename(index=str, columns={"Summary": "summary", "Text": "text"})

# Remove rows containing missing value in title or text column
dataset = dataset[["summary", "text"]]
dataset = dataset[pd.notnull(dataset["summary"])]
dataset = dataset[pd.notnull(dataset["text"])]


# Filter acc. to 1.5*IQR
def iqr_filter(ds, series):
    initial_length = ds.shape[0]
    q1, q3 = ds[series].str.len().quantile([.25, .75])
    iqr = 1.5 * (q3 - q1)
    min_limit = 1 if (q1 - iqr) < 1 else (q1 - iqr)
    max_limit = (q3 + iqr)
    print("Min limit: ", min_limit)
    print("Max limit: ", max_limit)
    ds = ds[ds[series].map(len) > min_limit]
    ds = ds[ds[series].map(len) < max_limit]
    print("Initial Length: ", initial_length)
    return ds


# Reduce data for now for testing
MAX_NUMBER_OF_DATA = 1000
dataset = dataset.iloc[:MAX_NUMBER_OF_DATA, :]
print("Dataset limited.")

# Remove punctuation
dataset = dataset.applymap(lambda x: re.sub(r'[^\w\s]', '', x))
print("Punctuation removed.")

# Lowercase all text
dataset = dataset.applymap(lambda x: x.lower())
print("Lower case applied.")

# Remove outliers by 1.5 IQR Rule
dataset = iqr_filter(dataset, "summary")
print("After IQR: ", dataset.shape[0])
dataset = iqr_filter(dataset, "text")
print("After IQR: ", dataset.shape[0])
print("Removed outliers.")

# Tokenize texts
tokenized_dataset = dataset.applymap(lambda x: nltk.word_tokenize(x))
print("Dataset tokenized.")

""" Optional
# Remove stopwords
def filter_stopwords(word_list):
    return [word for word in word_list if word not in stopwords.words('english')]


filtered_dataset = tokenized_dataset.applymap(lambda x: filter_stopwords(x))
print("Dataset filtered.")
"""

# Load Glove for pre-defined word embeddings

filename = 'data/glove.6B.50d.txt'


def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename, 'r', encoding="utf-8")
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab, embd


vocab, embd = loadGloVe(filename)

embedding = np.asarray(embd)
embedding = embedding.astype(np.float32)

word_vec_dim = len(embedding[0])  # word_vec_dim = dimension of each word vectors


# returns the most similar "word"(represented as vector) in embedding array
# returns array in embedding that's most similar (in terms of cosine similarity) to x
def np_nearest_neighbour(x):
    xdoty = np.multiply(embedding, x)
    xdoty = np.sum(xdoty, 1)
    xlen = np.square(x)
    xlen = np.sum(xlen, 0)
    xlen = np.sqrt(xlen)
    ylen = np.square(embedding)
    ylen = np.sum(ylen, 1)
    ylen = np.sqrt(ylen)
    xlenylen = np.multiply(xlen, ylen)
    cosine_similarities = np.divide(xdoty, xlenylen)

    return embedding[np.argmax(cosine_similarities)]


# converts a given word into its vector representation
def word2vec(word):
    if word in vocab:
        return embedding[vocab.index(word)]
    else:
        return embedding[vocab.index('unk')]


# converts a given vector representation into the represented word
def vec2word(vec):
    for x in range(0, len(embedding)):
        if np.array_equal(embedding[x], np.asarray(vec)):
            return vocab[x]
    return vec2word(np_nearest_neighbour(np.asarray(vec)))


# Get only unique words
vocab_limit = []
embd_limit = []


def special_token_recorder():
    special_tokens = ['unk', '<SOS>', 'eos']
    for token in special_tokens:
        vocab_limit.append(token)
        embd_limit.append(word2vec(token))
    vocab_limit.append('<PAD>')
    null_vector = np.zeros([word_vec_dim])
    embd_limit.append(null_vector)


# Improved version
def unique_token_recorder(text):
    difference = set(text) - set(vocab_limit)
    for token in difference:
        if token in vocab:
            vocab_limit.append(token)
            embd_limit.append(word2vec(token))


special_token_recorder()
print("Special Tokens added.")

for i, row in enumerate(tokenized_dataset.values):
    print("Row: ", i, end="..")
    for col in row:
        unique_token_recorder(col)
    print("")
print("All Unique Tokens added.")


# Vectorize all texts
def vectorize_text(text):
    vec_text = []

    for word in text:
        vec_text.append(word2vec(word))

    vec_text.append(word2vec('eos'))
    vec_text = np.asarray(vec_text)
    return vec_text.astype(np.float32)


vectorized_titles = []
vectorized_texts = []

for i, row in enumerate(tokenized_dataset.values):
    print("Vectorizing # ", i, end="..")
    vectorized_titles.append(vectorize_text(row[0]))
    vectorized_texts.append(vectorize_text(row[1]))
    print("")
print("All vectorized added.")

# Save our processed data
with open('vocab_limit', 'wb') as fp:
    pickle.dump(vocab_limit, fp)
with open('embd_limit', 'wb') as fp:
    pickle.dump(embd_limit, fp)
with open('vec_texts', 'wb') as fp:
    pickle.dump(vectorized_texts, fp)
with open('vec_titles', 'wb') as fp:
    pickle.dump(vectorized_titles, fp)
print("All data saved.")

## Data Preprocessing is done.

# In[ ]: Training

# Training Section Starts here


# Load pre-processed data
with open('vec_texts', 'rb') as fp:
    vectorized_texts = pickle.load(fp)

with open('vec_titles', 'rb') as fp:
    vectorized_titles = pickle.load(fp)

with open('vocab_limit', 'rb') as fp:
    vocab_limit = pickle.load(fp)

with open('embd_limit', 'rb') as fp:
    embd_limit = pickle.load(fp)
print("All data loaded.")

# We are implementing a local attention model here, so we set the D (window size)
D = 10  # offset from the current position
window_size = 2 * D + 1  # window size, our model will be looking 10 hidden states at a time

print("Remove texts smaller than window size")
print("Before: ", len(vectorized_texts))
vectorized_texts = [vectorized_text for vectorized_text in vectorized_texts if vectorized_text.shape[0] > window_size]
print("After: ", len(vectorized_texts))

# Split dataset
train_set_split_ratio = 0.8
validation_set_split_ratio = 0.2

total_length = len(vectorized_texts)
train_length = int(total_length * train_set_split_ratio)
validation_length = int(total_length * validation_set_split_ratio)
test_length = int(total_length - (train_length + validation_length))
print("Total Data:", total_length)

train_titles = vectorized_titles[:train_length]
train_texts = vectorized_texts[:train_length]
print("Training Data: ", len(train_titles), " - ", len(train_texts))

valid_texts = vectorized_texts[train_length:total_length]
valid_titles = vectorized_titles[train_length:total_length]
print("Validation Data: ", len(valid_texts), " - ", len(valid_titles))

test_texts = vectorized_texts[train_length:total_length]
test_titles = vectorized_titles[train_length:total_length]
print("Test Data: ", len(test_texts), " - ", len(test_titles))


# Convert all vocab limit words into integer classes so that tensorflow's loss function can be used on this
def transform_out(output_text):
    output_len = len(output_text)
    transformed_output = np.zeros([output_len], dtype=np.int32)
    for i in range(0, output_len):
        transformed_output[i] = vocab_limit.index(vec2word(output_text[i]))
    return transformed_output


# In[ ]:
batch_size = 8
hidden_size = 175
learning_rate = 0.003
K = 8
vocab_len = len(vocab_limit)
epochs = 100
print("Hyper params set.")

# Tenserflow placeholders
tf_text = tf.placeholder(tf.float32, [None, word_vec_dim])
tf_seq_len = tf.placeholder(tf.int32)
tf_title = tf.placeholder(tf.int32, [None])
tf_output_len = tf.placeholder(tf.int32)

np_embd_limit = np.asarray(embd_limit, dtype=np.float32)
SOS = embd_limit[vocab_limit.index('<SOS>')]


## Encoder
# Forward and Backward LSTM
# The RNN used here, is a standard LSTM with RRA (Recurrent Residual Attention)
# Forward LSTM will be act like as encoder and starting from the first word encode each word with the context of the previous words.

def forward_encoder(inp, hidden, cell,
                    wf, uf, bf,
                    wi, ui, bi,
                    wo, uo, bo,
                    wc, uc, bc,
                    Wattention, seq_len, inp_dim):
    Wattention = tf.nn.softmax(Wattention, 0)
    hidden_forward = tf.TensorArray(size=seq_len, dtype=tf.float32)

    hidden_residuals = tf.TensorArray(size=K, dynamic_size=True, dtype=tf.float32, clear_after_read=False)
    hidden_residuals = hidden_residuals.unstack(tf.zeros([K, hidden_size], dtype=tf.float32))

    i = 0
    j = K

    def cond(i, j, hidden, cell, hidden_forward, hidden_residuals):
        return i < seq_len

    def body(i, j, hidden, cell, hidden_forward, hidden_residuals):
        x = tf.reshape(inp[i], [1, inp_dim])

        hidden_residuals_stack = hidden_residuals.stack()

        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j - K:j], Wattention), 0)
        RRA = tf.reshape(RRA, [1, hidden_size])

        # LSTM with RRA
        fg = tf.sigmoid(tf.matmul(x, wf) + tf.matmul(hidden, uf) + bf)
        ig = tf.sigmoid(tf.matmul(x, wi) + tf.matmul(hidden, ui) + bi)
        og = tf.sigmoid(tf.matmul(x, wo) + tf.matmul(hidden, uo) + bo)
        cell = tf.multiply(fg, cell) + tf.multiply(ig, tf.sigmoid(tf.matmul(x, wc) + tf.matmul(hidden, uc) + bc))
        hidden = tf.multiply(og, tf.tanh(cell + RRA))

        hidden_residuals = tf.cond(tf.equal(j, seq_len - 1 + K),
                                   lambda: hidden_residuals,
                                   lambda: hidden_residuals.write(j, tf.reshape(hidden, [hidden_size])))

        hidden_forward = hidden_forward.write(i, tf.reshape(hidden, [hidden_size]))

        return i + 1, j + 1, hidden, cell, hidden_forward, hidden_residuals

    _, _, _, _, hidden_forward, hidden_residuals = tf.while_loop(cond, body,
                                                                 [i, j, hidden, cell, hidden_forward, hidden_residuals])

    hidden_residuals.close().mark_used()

    return hidden_forward.stack()


# Backward LSTM will be act like as encoder and starting from the last word  encode each word with the context of the later words.
def backward_encoder(inp, hidden, cell,
                     wf, uf, bf,
                     wi, ui, bi,
                     wo, uo, bo,
                     wc, uc, bc,
                     Wattention, seq_len, inp_dim):
    Wattention = tf.nn.softmax(Wattention, 0)
    hidden_backward = tf.TensorArray(size=seq_len, dtype=tf.float32)

    hidden_residuals = tf.TensorArray(size=K, dynamic_size=True, dtype=tf.float32, clear_after_read=False)
    hidden_residuals = hidden_residuals.unstack(tf.zeros([K, hidden_size], dtype=tf.float32))

    i = seq_len - 1
    j = K

    def cond(i, j, hidden, cell, hidden_backward, hidden_residuals):
        return i > -1

    def body(i, j, hidden, cell, hidden_backward, hidden_residuals):
        x = tf.reshape(inp[i], [1, inp_dim])

        hidden_residuals_stack = hidden_residuals.stack()

        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j - K:j], Wattention), 0)
        RRA = tf.reshape(RRA, [1, hidden_size])

        # LSTM with RRA
        fg = tf.sigmoid(tf.matmul(x, wf) + tf.matmul(hidden, uf) + bf)
        ig = tf.sigmoid(tf.matmul(x, wi) + tf.matmul(hidden, ui) + bi)
        og = tf.sigmoid(tf.matmul(x, wo) + tf.matmul(hidden, uo) + bo)
        cell = tf.multiply(fg, cell) + tf.multiply(ig, tf.sigmoid(tf.matmul(x, wc) + tf.matmul(hidden, uc) + bc))
        hidden = tf.multiply(og, tf.tanh(cell + RRA))

        hidden_residuals = tf.cond(tf.equal(j, seq_len - 1 + K),
                                   lambda: hidden_residuals,
                                   lambda: hidden_residuals.write(j, tf.reshape(hidden, [hidden_size])))

        hidden_backward = hidden_backward.write(i, tf.reshape(hidden, [hidden_size]))

        return i - 1, j + 1, hidden, cell, hidden_backward, hidden_residuals

    _, _, _, _, hidden_backward, hidden_residuals = tf.while_loop(cond, body, [i, j, hidden, cell, hidden_backward,
                                                                               hidden_residuals])

    hidden_residuals.close().mark_used()

    return hidden_backward.stack()


# Decoder

def decoder(x, hidden, cell,
            wf, uf, bf,
            wi, ui, bi,
            wo, uo, bo,
            wc, uc, bc, RRA):
    # LSTM with RRA
    fg = tf.sigmoid(tf.matmul(x, wf) + tf.matmul(hidden, uf) + bf)
    ig = tf.sigmoid(tf.matmul(x, wi) + tf.matmul(hidden, ui) + bi)
    og = tf.sigmoid(tf.matmul(x, wo) + tf.matmul(hidden, uo) + bo)
    cell_next = tf.multiply(fg, cell) + tf.multiply(ig, tf.sigmoid(tf.matmul(x, wc) + tf.matmul(hidden, uc) + bc))
    hidden_next = tf.multiply(og, tf.tanh(cell + RRA))

    return hidden_next, cell_next


# Local Attention model
# The attention mechanism is usually implemented to compute an attention score for each of the encoded hidden state
# in the context of a particular decoder hidden state in each timestep - all to determine which encoded hidden states to
# attend to, given the context of a particular decoder hidden state.

def score(hs, ht, Wa, seq_len):
    return tf.reshape(tf.matmul(tf.matmul(hs, Wa), tf.transpose(ht)), [seq_len])


def align(hs, ht, Wp, Vp, Wa, tf_seq_len):
    pd = tf.TensorArray(size=(2 * D + 1), dtype=tf.float32)

    positions = tf.cast(tf_seq_len - 1 - 2 * D, dtype=tf.float32)

    sigmoid_multiplier = tf.nn.sigmoid(tf.matmul(tf.tanh(tf.matmul(ht, Wp)), Vp))
    sigmoid_multiplier = tf.reshape(sigmoid_multiplier, [])

    pt_float = positions * sigmoid_multiplier

    pt = tf.cast(pt_float, tf.int32)
    pt = pt + D  # center to window

    sigma = tf.constant(D / 2, dtype=tf.float32)

    i = 0
    pos = pt - D

    def cond(i, pos, pd):
        return i < (2 * D + 1)

    def body(i, pos, pd):
        comp_1 = tf.cast(tf.square(pos - pt), tf.float32)
        comp_2 = tf.cast(2 * tf.square(sigma), tf.float32)

        pd = pd.write(i, tf.exp(-(comp_1 / comp_2)))

        return i + 1, pos + 1, pd

    i, pos, pd = tf.while_loop(cond, body, [i, pos, pd])

    local_hs = hs[(pt - D):(pt + D + 1)]

    normalized_scores = tf.nn.softmax(score(local_hs, ht, Wa, 2 * D + 1))

    pd = pd.stack()

    G = tf.multiply(normalized_scores, pd)
    G = tf.reshape(G, [2 * D + 1, 1])

    return G, pt


# Model

def model(tf_text, tf_seq_len, tf_output_len):
    # PARAMETERS

    # 1.1 FORWARD ENCODER PARAMETERS

    initial_hidden_f = tf.zeros([1, hidden_size], dtype=tf.float32)
    cell_f = tf.zeros([1, hidden_size], dtype=tf.float32)
    wf_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim, hidden_size], stddev=0.01))
    uf_f = tf.Variable(np.eye(hidden_size), dtype=tf.float32)
    bf_f = tf.Variable(tf.zeros([1, hidden_size]), dtype=tf.float32)
    wi_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim, hidden_size], stddev=0.01))
    ui_f = tf.Variable(np.eye(hidden_size), dtype=tf.float32)
    bi_f = tf.Variable(tf.zeros([1, hidden_size]), dtype=tf.float32)
    wo_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim, hidden_size], stddev=0.01))
    uo_f = tf.Variable(np.eye(hidden_size), dtype=tf.float32)
    bo_f = tf.Variable(tf.zeros([1, hidden_size]), dtype=tf.float32)
    wc_f = tf.Variable(tf.truncated_normal(shape=[word_vec_dim, hidden_size], stddev=0.01))
    uc_f = tf.Variable(np.eye(hidden_size), dtype=tf.float32)
    bc_f = tf.Variable(tf.zeros([1, hidden_size]), dtype=tf.float32)
    Wattention_f = tf.Variable(tf.zeros([K, 1]), dtype=tf.float32)

    # 1.2 BACKWARD ENCODER PARAMETERS

    initial_hidden_b = tf.zeros([1, hidden_size], dtype=tf.float32)
    cell_b = tf.zeros([1, hidden_size], dtype=tf.float32)
    wf_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim, hidden_size], stddev=0.01))
    uf_b = tf.Variable(np.eye(hidden_size), dtype=tf.float32)
    bf_b = tf.Variable(tf.zeros([1, hidden_size]), dtype=tf.float32)
    wi_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim, hidden_size], stddev=0.01))
    ui_b = tf.Variable(np.eye(hidden_size), dtype=tf.float32)
    bi_b = tf.Variable(tf.zeros([1, hidden_size]), dtype=tf.float32)
    wo_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim, hidden_size], stddev=0.01))
    uo_b = tf.Variable(np.eye(hidden_size), dtype=tf.float32)
    bo_b = tf.Variable(tf.zeros([1, hidden_size]), dtype=tf.float32)
    wc_b = tf.Variable(tf.truncated_normal(shape=[word_vec_dim, hidden_size], stddev=0.01))
    uc_b = tf.Variable(np.eye(hidden_size), dtype=tf.float32)
    bc_b = tf.Variable(tf.zeros([1, hidden_size]), dtype=tf.float32)
    Wattention_b = tf.Variable(tf.zeros([K, 1]), dtype=tf.float32)

    # 2 ATTENTION PARAMETERS

    Wp = tf.Variable(tf.truncated_normal(shape=[2 * hidden_size, 50], stddev=0.01))
    Vp = tf.Variable(tf.truncated_normal(shape=[50, 1], stddev=0.01))
    Wa = tf.Variable(tf.truncated_normal(shape=[2 * hidden_size, 2 * hidden_size], stddev=0.01))
    Wc = tf.Variable(tf.truncated_normal(shape=[4 * hidden_size, 2 * hidden_size], stddev=0.01))

    # 3 DECODER PARAMETERS

    Ws = tf.Variable(tf.truncated_normal(shape=[2 * hidden_size, vocab_len], stddev=0.01))

    cell_d = tf.zeros([1, 2 * hidden_size], dtype=tf.float32)
    wf_d = tf.Variable(tf.truncated_normal(shape=[word_vec_dim, 2 * hidden_size], stddev=0.01))
    uf_d = tf.Variable(np.eye(2 * hidden_size), dtype=tf.float32)
    bf_d = tf.Variable(tf.zeros([1, 2 * hidden_size]), dtype=tf.float32)
    wi_d = tf.Variable(tf.truncated_normal(shape=[word_vec_dim, 2 * hidden_size], stddev=0.01))
    ui_d = tf.Variable(np.eye(2 * hidden_size), dtype=tf.float32)
    bi_d = tf.Variable(tf.zeros([1, 2 * hidden_size]), dtype=tf.float32)
    wo_d = tf.Variable(tf.truncated_normal(shape=[word_vec_dim, 2 * hidden_size], stddev=0.01))
    uo_d = tf.Variable(np.eye(2 * hidden_size), dtype=tf.float32)
    bo_d = tf.Variable(tf.zeros([1, 2 * hidden_size]), dtype=tf.float32)
    wc_d = tf.Variable(tf.truncated_normal(shape=[word_vec_dim, 2 * hidden_size], stddev=0.01))
    uc_d = tf.Variable(np.eye(2 * hidden_size), dtype=tf.float32)
    bc_d = tf.Variable(tf.zeros([1, 2 * hidden_size]), dtype=tf.float32)

    hidden_residuals_d = tf.TensorArray(size=K, dynamic_size=True, dtype=tf.float32, clear_after_read=False)
    hidden_residuals_d = hidden_residuals_d.unstack(tf.zeros([K, 2 * hidden_size], dtype=tf.float32))

    Wattention_d = tf.Variable(tf.zeros([K, 1]), dtype=tf.float32)

    output = tf.TensorArray(size=tf_output_len, dtype=tf.float32)

    # BI-DIRECTIONAL LSTM

    hidden_forward = forward_encoder(tf_text,
                                     initial_hidden_f, cell_f,
                                     wf_f, uf_f, bf_f,
                                     wi_f, ui_f, bi_f,
                                     wo_f, uo_f, bo_f,
                                     wc_f, uc_f, bc_f,
                                     Wattention_f,
                                     tf_seq_len,
                                     word_vec_dim)

    hidden_backward = backward_encoder(tf_text,
                                       initial_hidden_b, cell_b,
                                       wf_b, uf_b, bf_b,
                                       wi_b, ui_b, bi_b,
                                       wo_b, uo_b, bo_b,
                                       wc_b, uc_b, bc_b,
                                       Wattention_b,
                                       tf_seq_len,
                                       word_vec_dim)

    encoded_hidden = tf.concat([hidden_forward, hidden_backward], 1)

    # ATTENTION MECHANISM AND DECODER

    decoded_hidden = encoded_hidden[0]
    decoded_hidden = tf.reshape(decoded_hidden, [1, 2 * hidden_size])
    Wattention_d_normalized = tf.nn.softmax(Wattention_d)
    tf_embd_limit = tf.convert_to_tensor(np_embd_limit)

    y = tf.convert_to_tensor(SOS)  # inital decoder token <SOS> vector
    y = tf.reshape(y, [1, word_vec_dim])

    j = K

    hidden_residuals_stack = hidden_residuals_d.stack()

    RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j - K:j], Wattention_d_normalized), 0)
    RRA = tf.reshape(RRA, [1, 2 * hidden_size])

    decoded_hidden_next, cell_d = decoder(y, decoded_hidden, cell_d,
                                          wf_d, uf_d, bf_d,
                                          wi_d, ui_d, bf_d,
                                          wo_d, uo_d, bf_d,
                                          wc_d, uc_d, bc_d,
                                          RRA)
    decoded_hidden = decoded_hidden_next

    hidden_residuals_d = hidden_residuals_d.write(j, tf.reshape(decoded_hidden, [2 * hidden_size]))

    j = j + 1

    i = 0

    def attention_decoder_cond(i, j, decoded_hidden, cell_d, hidden_residuals_d, output):
        return i < tf_output_len

    def attention_decoder_body(i, j, decoded_hidden, cell_d, hidden_residuals_d, output):
        # LOCAL ATTENTION

        G, pt = align(encoded_hidden, decoded_hidden, Wp, Vp, Wa, tf_seq_len)
        local_encoded_hidden = encoded_hidden[pt - D:pt + D + 1]
        weighted_encoded_hidden = tf.multiply(local_encoded_hidden, G)
        context_vector = tf.reduce_sum(weighted_encoded_hidden, 0)
        context_vector = tf.reshape(context_vector, [1, 2 * hidden_size])

        attended_hidden = tf.tanh(tf.matmul(tf.concat([context_vector, decoded_hidden], 1), Wc))

        # DECODER

        y = tf.matmul(attended_hidden, Ws)

        output = output.write(i, tf.reshape(y, [vocab_len]))
        # Save probability distribution as output

        y = tf.nn.softmax(y)

        y_index = tf.cast(tf.argmax(tf.reshape(y, [vocab_len])), tf.int32)
        y = tf_embd_limit[y_index]
        y = tf.reshape(y, [1, word_vec_dim])

        # setting next decoder input token as the word_vector of maximum probability
        # as found from previous attention-decoder output.

        hidden_residuals_stack = hidden_residuals_d.stack()

        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j - K:j], Wattention_d_normalized), 0)
        RRA = tf.reshape(RRA, [1, 2 * hidden_size])

        decoded_hidden_next, cell_d = decoder(y, decoded_hidden, cell_d,
                                              wf_d, uf_d, bf_d,
                                              wi_d, ui_d, bf_d,
                                              wo_d, uo_d, bf_d,
                                              wc_d, uc_d, bc_d,
                                              RRA)

        decoded_hidden = decoded_hidden_next

        hidden_residuals_d = tf.cond(tf.equal(j, tf_output_len - 1 + K + 1),  # (+1 for <SOS>)
                                     lambda: hidden_residuals_d,
                                     lambda: hidden_residuals_d.write(j, tf.reshape(decoded_hidden, [2 * hidden_size])))

        return i + 1, j + 1, decoded_hidden, cell_d, hidden_residuals_d, output

    i, j, decoded_hidden, cell_d, hidden_residuals_d, output = tf.while_loop(attention_decoder_cond,
                                                                             attention_decoder_body,
                                                                             [i, j, decoded_hidden, cell_d,
                                                                              hidden_residuals_d, output])
    hidden_residuals_d.close().mark_used()

    output = output.stack()

    return output


output = model(tf_text, tf_seq_len, tf_output_len)

# OPTIMIZER

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf_title))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate / train_length).minimize(cost)

print("Optimization done.")

# PREDICTION
pred = tf.TensorArray(size=tf_output_len, dtype=tf.int32)

i = 0


def cond_pred(i, pred):
    return i < tf_output_len


def body_pred(i, pred):
    pred = pred.write(i, tf.cast(tf.argmax(output[i]), tf.int32))
    return i + 1, pred


i, pred = tf.while_loop(cond_pred, body_pred, [i, pred])

prediction = pred.stack()

init = tf.global_variables_initializer()

loss_history = []
mse_history = []

train_batch_amount = math.floor(len(train_texts) / batch_size)
train_texts = train_texts[: batch_size * train_batch_amount]
train_titles = train_titles[: batch_size * train_batch_amount]

valid_batch_size = math.ceil(batch_size / 8)
valid_batch_amount = math.floor(len(valid_texts) / valid_batch_size)
valid_texts = valid_texts[: valid_batch_size * valid_batch_amount]
valid_titles = valid_titles[: valid_batch_size * valid_batch_amount]
train_length = len(train_texts)
valid_length = len(valid_texts)

print("Starting Training - Epochs: ", epochs, "\nTraining set Size: ", train_length, ", Batch Size/Amount:", batch_size,
      "/", train_batch_amount, "\nValidation set Size: ", valid_length, "Validation Batch Size/Amount: ",
      valid_batch_size, "/", valid_batch_amount)
# In[]:
from random import shuffle

saver = tf.train.Saver()
with tf.Session() as sess:  # Start Tensorflow Session

    # Prepares variable for saving the model
    sess.run(init)  # initialize all variables
    step = 0
    loss_list = []
    acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_val_acc = 0
    display_step = 1

    epoch_loss_history = []
    valid_epoch_loss_history = []

    while step < epochs:
        print("\n")
        print("---------TRAINING---------")
        print("Step: ", step, "/", epochs)
        print("---------TRAINING---------")
        print("\n")
        total_loss = 0
        total_acc = 0
        total_val_loss = 0
        total_val_acc = 0

        batch_loss_history = []
        valid_batch_loss_history = []

        shuffle(train_texts)
        shuffle(train_titles)

        for u in range(0, train_length, batch_size):
            print("Training range: (", u, ",", u + batch_size, ")")
            X_train = train_texts[u:u + batch_size]
            y_train = train_titles[u:u + batch_size]

            loss_records = []

            for i in range(0, batch_size):
                train_out = transform_out(y_train[i][0:len(y_train[i]) - 1])
                if i % display_step == 0 and step + 1 == epochs:
                    print("Sample #" + str(i + step * train_length), "/", str(train_length * epochs))
                    print("\nTEXT: ", end="")
                    flag = 0
                    for vec in X_train[i]:
                        if vec2word(vec) in string.punctuation or flag == 0:
                            print(str(vec2word(vec)), end='')
                        else:
                            print((" " + str(vec2word(vec))), end='')
                        flag = 1

                # Run optimization operation (backpropagation)
                _, loss, pred = sess.run([optimizer, cost, prediction], feed_dict={tf_text: X_train[i],
                                                                                   tf_seq_len: len(X_train[i]),
                                                                                   tf_title: train_out,
                                                                                   tf_output_len: len(train_out)})
                if i % display_step == 0 and step + 1 == epochs:
                    print("\nPREDICTED SUMMARY: ", end="")
                    flag = 0
                    for index in pred:
                        # if int(index)!=vocab_limit.index('eos'):
                        if vocab_limit[int(index)] in string.punctuation or flag == 0:
                            print(str(vocab_limit[int(index)]), end='')
                        else:
                            print(" " + str(vocab_limit[int(index)]), end='')
                        flag = 1
                    print("\n")
                    print("ACTUAL SUMMARY   : ", end="")
                    flag = 0
                    for vec in train_titles[i]:
                        if vec2word(vec) != 'eos':
                            if vec2word(vec) in string.punctuation or flag == 0:
                                print(str(vec2word(vec)), end='')
                            else:
                                print((" " + str(vec2word(vec))), end='')
                        flag = 1

                    print("\n")
                    print("loss=" + str(loss))
                    print("\n")
                loss_records.append(loss)
            loss_median = np.median(loss_records)
            print("Loss median: ", loss_median)
            batch_loss_history.append(loss_median)
        batch_loss_mean = np.mean(batch_loss_history)
        print("Training Batch Loss mean: ", batch_loss_mean)
        epoch_loss_history.append(batch_loss_mean)

        for u in range(0, valid_length, valid_batch_size):
            print("Validation range: (", u, ",", u + batch_size, ")")
            X_valid = valid_texts[u:u + valid_batch_size]
            y_valid = valid_titles[u:u + valid_batch_size]

            valid_loss_records = []

            for i in range(0, valid_batch_size):
                valid_out = transform_out(y_valid[i][0:len(y_valid[i]) - 1])
                """
                if i % display_step == 0 and step + 1 == epochs:
                    print("Sample #" + str(i + step * train_length), "/", str(train_length * epochs))
                    print("\nTEXT: ", end="")
                    flag = 0
                    for vec in X_train[i]:
                        if vec2word(vec) in string.punctuation or flag == 0:
                            print(str(vec2word(vec)), end='')
                        else:
                            print((" " + str(vec2word(vec))), end='')
                        flag = 1
                """

                # Run optimization operation (backpropagation)
                _, loss, pred = sess.run([optimizer, cost, prediction], feed_dict={tf_text: X_valid[i],
                                                                                   tf_seq_len: len(X_valid[i]),
                                                                                   tf_title: valid_out,
                                                                                   tf_output_len: len(valid_out)})

                valid_loss_records.append(loss)
            valid_loss_median = np.median(valid_loss_records)
            print("Loss median: ", valid_loss_median)
            valid_batch_loss_history.append(valid_loss_median)
        valid_batch_loss_mean = np.mean(valid_batch_loss_history)
        print("Validation Batch Loss mean: ", valid_batch_loss_mean)
        valid_epoch_loss_history.append(valid_batch_loss_mean)
        step = step + 1
    # save_path = saver.save(sess, "models/trained_model.ckpt")
    # print("Model saved in path: %s" % save_path)
# In[]:
plt.plot(range(epochs), epoch_loss_history, color="blue", label="Training")
plt.plot(range(epochs), valid_epoch_loss_history, color="green", label="Validation")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Epoch Loss")
plt.show()

# In[]:
plt.plot(range(train_batch_amount), batch_loss_history)
plt.xlabel("Iterations")
plt.ylabel("Batch Loss")
plt.show()
# In[]:
test_loss_history = []
pred_titles = []

with tf.Session() as sess:
    saver.restore(sess, "models/trained_model.ckpt")
    print("Model restored.")

    for i in range(0, test_length):
        test_out = transform_out(test_titles[i][0:len(test_titles[i]) - 1])
        print("Test out generated.")

        print("\nText:")
        flag = 0
        print(i)
        for vec in test_texts[i]:
            if vec2word(vec) in string.punctuation or flag == 0:
                print(str(vec2word(vec)), end='')
            else:
                print((" " + str(vec2word(vec))), end='')
            flag = 1

        print("\n")

        # Run optimization operation (backpropagation)
        _, loss, pred_title = sess.run([optimizer, cost, prediction], feed_dict={tf_text: test_texts[i],
                                                                                 tf_seq_len: len(test_texts[i]),
                                                                                 tf_title: test_out,
                                                                                 tf_output_len: len(test_out)})

        predicted_title_in_words = []
        print("\nPREDICTED SUMMARY:\n")
        flag = 0
        for index in pred_title:
            # if int(index)!=vocab_limit.index('eos'):
            if vocab_limit[int(index)] in string.punctuation or flag == 0:
                predicted_title_in_words.append(str(vocab_limit[int(index)]))
                print(str(vocab_limit[int(index)]), end='')
            else:
                predicted_title_in_words.append(str(vocab_limit[int(index)]))
                print(" " + str(vocab_limit[int(index)]), end='')
            flag = 1
        print("\n")

        pred_titles.append(vectorize_text(predicted_title_in_words))

        print("ACTUAL SUMMARY:\n")
        flag = 0
        for vec in test_titles[i]:
            if vec2word(vec) != 'eos':
                if vec2word(vec) in string.punctuation or flag == 0:
                    print(str(vec2word(vec)), end='')
                else:
                    print((" " + str(vec2word(vec))), end='')
            flag = 1

        print("\n")
        print("loss=" + str(loss))
        test_loss_history.append(loss)
        print("Loss appended")

plt.plot(range(len(test_loss_history)), test_loss_history)
plt.title("Test Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

test_out = [transform_out(x) for x in test_titles]
print("Test out generated.")
pred_out = [transform_out(x) for x in pred_titles]
print("Pred out generated.")
mse = tf.reduce_mean(tf.squared_difference(pred_out, test_out))
print("MSE=" + str(mse))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


"""
great_in_vec = word2vec("great")
good_in_vec = word2vec("good")
bad_in_vec = word2vec("bad")
sucks_in_vec = word2vec("sucks")
angle_between(great_in_vec, great_in_vec)
angle_between(great_in_vec, bad_in_vec)
angle_between(word2vec("father"), word2vec("mother"))
angle_between(good_in_vec, bad_in_vec)

angle_between(pred_titles[0][1], test_titles[0][1])
"""

plt.plot(range(len(test_loss_history)), test_loss_history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
