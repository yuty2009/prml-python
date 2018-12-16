# -*- coding: utf-8 -*-

import random
import numpy as np
import tensorflow as tf
import lmbasic as model
import wordreader as reader
from matplotlib import pylab
from sklearn.manifold import TSNE

datapath = 'E:\\prmldata\\text8\\text8.zip'
modelpath = 'E:\\prmldata\\text8\\models\\'

words = reader.load_words(datapath)
print('Data size %d' % len(words))

vocab_size = 50000
words_indices, vocab_dict, vocab_rdict, vocab_counter = \
    reader.word2index(words, vocab_size)
print('Most common words (+UNKNOWN)', vocab_counter[:5])
print('Sample data', words_indices[:10])
del words  # Hint to reduce memory.

wordset = reader.Dataset(words_indices)

print('data:', [vocab_rdict[di] for di in words_indices[:16]])

for skip_window in [2, 1]:
    wordset.set_window(skip_window)
    batch, labels = wordset.next_batch_skipgram(batch_size=16)
    print('\nwith skip_window = %d:' % skip_window)
    print('    batch:', [vocab_rdict[bi] for bi in batch])
    print('    labels:', [vocab_rdict[li] for li in labels.reshape(16)])

valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))

loss, train_op, embeddings, X, ytrue, lr = model.lm(vocab_size, 128)

# Compute the similarity between minibatch examples and all embeddings.
# We use the cosine distance:
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_examples)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

maxstep = 100000
reportstep = 2000
savestep = 20000
lr_start = 1e-3
gfromscratch = True

global_step = 0
saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(modelpath)
    if not gfromscratch and ckpt and ckpt.model_checkpoint_path:
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Pre-trained model restored from %s' % (ckpt.model_checkpoint_path))
    else:
        sess.run(tf.global_variables_initializer())
    average_loss = 0
    for step in range(global_step, global_step+maxstep):
        batch_X, batch_y = wordset.next_batch_skipgram(batch_size=128)
        lr_now = lr_start * (1 + 1e-4 * step) ** (-0.75)
        _, train_loss = sess.run([train_op, loss], feed_dict={X: batch_X, ytrue: batch_y, lr: lr_now})
        average_loss += train_loss
        if step % reportstep == 0:
            if step > 0: average_loss = average_loss / reportstep
            print('Step=%d, lr=%.4f, loss=%.4f' % (step, lr_now, average_loss))
            average_loss = 0

            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = vocab_rdict[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = vocab_rdict[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)

        if (step + 1) % savestep == 0:
            saver.save(sess, modelpath+'word2vec-basic', global_step=step+1)

    final_embeddings = normalized_embeddings.eval()

num_points = 400
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

words = [vocab_rdict[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)