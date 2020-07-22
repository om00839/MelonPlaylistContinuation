import os
import json
import pickle
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.notebook import tqdm, trange
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class Configuration:
    def __init__(self, **kwargs):
        # model configuration
        self.song_hidden_dim = 64
        self.tag_hidden_dim = 64
        self.song_maxlen = None
        self.tag_maxlen = None
        self.n_songs = None
        self.n_tags = None
        
        # train configuration
        self.batch_size = 32
        self.lr = 1e-4
        
        for k, v in kwargs.items():
            if k == 'song_hidden_dim':
                self.song_hidden_dim = v
            elif k == 'tag_hidden_dim':
                self.tag_hidden_dim = v
            elif k == 'song_maxlen':
                self.song_maxlen = v
            elif k == 'tag_maxlen':
                self.tag_maxlen = v
            elif k == 'n_songs':
                self.n_songs = v
            elif k == 'n_tags':
                self.n_tags = v
            elif k == 'batch_size':
                self.batch_size = v
            elif k == 'lr':
                self.lr = v
            elif k == 'epochs':
                self.epochs = v
            else:
                raise(ValueError)

class EmbMLP(keras.Model):
    def __init__(self, config, song_vectors, tag_vectors):
        super(EmbMLP, self).__init__()
        self.config = config
        self.song_embedding = layers.Embedding(self.config.n_songs,
                                          self.config.song_hidden_dim, 
                                          embeddings_initializer=tf.keras.initializers.Constant(song_vectors))
        self.linear = layers.Dense(self.config.n_songs, 
                                   input_shape=(self.config.song_hidden_dim,))

    def call(self, input):
        song_in, tag_in = input
        song_nonzero = tf.math.count_nonzero(song_in)
        song_embedded = tf.nn.relu(self.song_embedding(song_in))
        song_embedded_mean = tf.math.reduce_sum(song_embedded[:song_nonzero], axis=1) / tf.cast(song_nonzero, tf.float32)
        logits = self.linear(song_embedded_mean)

        return logits

class Trainer:
    def __init__(self, config, dataset):
        self.config = config
        self.model = EmbMLP(self.config, dataset.song_vectors, dataset.tag_vectors)
        self.optimizer = keras.optimizers.Adam(config.lr)
        
        for epoch in trange(1, config.epochs):
            # train 
            generator = iter(dataset.generate_input('train', config.batch_size))
            N = len(dataset.train_plylst_list)
            steps_per_epoch = (N // config.batch_size) + 1
            loss_list = list()
            for step in trange(1, steps_per_epoch+1):
                input, label = next(generator)
                label = label[0]
                loss = self.train_batches(input, label)
                loss_list.append(loss.numpy())
                if step % 100 == 0:
                    print(f"epoch/step {epoch}/{step}\t|\tavg. loss: {np.mean(loss_list)}")
                
            print(f"epoch {epoch}\t|\tavg. loss: {np.mean(loss_list)}")
                
            # evaluate
            generator = iter(dataset.generate_input('val', config.batch_size))
            N = len(dataset.val_plylst_list)
            steps_per_epoch = (N // config.batch_size) + 1
            song_ndcg_list = list()
            tag_ndcg_list = list()
            for step in trange(steps_per_epoch):
                input, (song_label, tag_label) = next(generator)
                logits = self.eval_batches(input)
                song_logits, tag_logits = tf.split(logits, (config.n_songs, config.n_tags), -1)
                song_mask = -song_label+1
                song_masked_logits = song_mask * song_logits
                songs_top100 = tf.math.top_k(song_masked_logits, k=100)
                
                for i in range(songs_top100.shape[0]):
                    song_nDCG = self._ndcg(songs_top100[i,:].tolist())
                    song_ndcg_list.append(song_nDCG)
                    
            score = 0.85 * np.mean(song_ndcg_list)
            print(f"epoch {epoch}\t|\tscore: {score}")
            
    def train_batches(self, input, label):
        with tf.GradientTape() as tape:
            logits = self.model(input)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(label, logits)
            vars_ = self.model.trainable_variables
            grads = tape.gradient(loss, vars_)
            self.optimizer.apply_gradients(zip(grads, vars_))
        return loss
            
    def eval_batches(self, input):
        logits = self.model(input)
        
        return song_logits
    
    def _ndcg(self, gt, rec):
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)
        return dcg / self._idcgs[len(gt)]

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    DATA_DIR = '../data/'
    INFO_DIR = './info/'
    dataset = MPCDataset(DATA_DIR, INFO_DIR)
    config = Configuration(song_maxlen = dataset.song_maxlen, 
                        tag_maxlen=dataset.tag_maxlen,
                        n_songs=dataset.n_songs,
                        n_tags=dataset.n_tags, 
                        batch_size=2, 
                        epochs=10)
    trainer = Trainer(config, dataset)