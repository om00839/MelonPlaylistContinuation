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

class MPCDataset:
    def __init__(self, data_dir, info_dir='./info/', **kwargs):
        
        self.data_dir = data_dir if data_dir[-1] == '/' else data_dir + '/'
        self.info_dir = info_dir if info_dir[-1] == '/' else info_dir + '/'
        
        self.train_plylst_list = None
        self.val_plylst_list = None
        self.test_plylst_list = None
        
        self.song_tag_d2v = None
        self.tag_song_d2v = None
        
        self.tag_maxlen = None
        self.n_tags = None
        self.idx2tag = None
        self.tag2idx = None
        
        self.song_maxlen = None
        self.n_songs = None
        self.idx2song = None
        self.song2idx = None
        
        self.song_vectors = None
        self.tag_vectors = None
        
        self.song_window=100
        self.tag_window=5
        self.min_count=2
        self.negative=5
        self.worker=4
        self.vector_size=64
        for k,v in kwargs.items():
            if k == 'song_window':
                song_window=v
            if k == 'tag_window':
                tag_window=v
            if k == 'min_count':
                min_count=v
            if k == 'negative':
                negative=v
            if k == 'worker':
                worker=v
            if k == 'vector_size':
                vector_size=v
        
        if not os.path.isdir(self.info_dir):
            os.mkdir(self.info_dir)

        self.train_plylst_list = self.load_json(
            os.path.join(self.data_dir, 'train.json'))
        self.val_plylst_list = self.load_json(
            os.path.join(self.data_dir, 'val.json'))
        self.test_plylst_list = self.load_json(
            os.path.join(self.data_dir, 'test.json'))
        
        # load song_tag_d2v, tag_song_d2v
        self.song_tag_d2v, self.tag_song_d2v = self.get_d2v_models()    
        
        self.idx2song = ['<pad>', '<unk>'] + self.song_tag_d2v.wv.index2word
        self.song2idx = {song:idx for idx, song in enumerate(self.idx2song)}
        
        self.idx2tag = ['<pad>', '<unk>'] + self.tag_song_d2v.wv.index2word
        self.tag2idx = {tag:idx for idx, tag in enumerate(self.idx2tag)}
        
        self.song_maxlen, self.tag_maxlen = self.get_maxlen()
        self.n_songs = len(self.idx2song)
        self.n_tags = len(self.idx2tag)
        
        song_vectors_path = self.info_dir + 'song_vectors.npy'
        tag_vectors_path = self.info_dir + 'tag_vectors.npy'
        if not (os.path.isfile(song_vectors_path) and os.path.isfile(tag_vectors_path)):
            self.song_vectors = np.concatenate([np.zeros((2, self.vector_size)), 
                                  self.song_tag_d2v.wv.vectors], axis=0)
            self.tag_vectors = np.concatenate([np.zeros((2, self.vector_size)), 
                                  self.tag_song_d2v.wv.vectors], axis=0)
            
            np.save(song_vectors_path, self.song_vectors)
            np.save(tag_vectors_path, self.tag_vectors)
            
        elif os.path.isfile(song_vectors_path) and os.path.isfile(tag_vectors_path):
            self.song_vectors = np.load(song_vectors_path)
            self.tag_vectors = np.load(tag_vectors_path)

    def load_json(self, path):
        return json.load(open(path, 'r'))
    
    def get_d2v_models(self):
        song_tag_d2v_path = self.info_dir + 'song_tag_d2v.model'
        tag_song_d2v_path = self.info_dir + 'tag_song_d2v.model'
        
        song_tag_d2v = None
        tag_song_d2v = None
        if not(os.path.isfile(song_tag_d2v_path) and os.path.isfile(tag_song_d2v_path)):
            song_tag_doc_list = list()
            tag_song_doc_list = list()
            for plylst in chain(self.train_plylst_list, 
                                self.val_plylst_list, 
                                self.test_plylst_list):
                songs = list()
                for song in plylst['songs']:
                    songs.append(str(song))
                    
                tags = list()
                for tag in plylst['tags']:
                    tags.append(str(tag))
                    
                song_tag_doc_list.append(TaggedDocument(songs, tags))
                tag_song_doc_list.append(TaggedDocument(tags, songs))

            song_tag_d2v = Doc2Vec(song_tag_doc_list, window=self.song_window, 
                                   min_count=self.min_count, negative=self.negative, 
                                   worker=self.worker, vector_size=self.vector_size)
            tag_song_d2v = Doc2Vec(tag_song_doc_list, window=self.tag_window, 
                                   min_count=self.min_count, negative=self.negative, 
                                   worker=self.worker, vector_size=self.vector_size)
            
            song_tag_d2v.save(song_tag_d2v_path)
            tag_song_d2v.save(tag_song_d2v_path)

            
        elif os.path.isfile(song_tag_d2v_path) and os.path.isfile(tag_song_d2v_path):
            song_tag_d2v = Doc2Vec.load(song_tag_d2v_path)
            tag_song_d2v = Doc2Vec.load(tag_song_d2v_path)
        
        return song_tag_d2v, tag_song_d2v
    
    def get_maxlen(self):
        song_maxlen = -1
        tag_maxlen = -1
        for plylst in chain(self.train_plylst_list, 
                            self.val_plylst_list, 
                            self.test_plylst_list):
            song_maxlen = max(song_maxlen, len(plylst['songs']))
            tag_maxlen = max(tag_maxlen, len(plylst['tags']))
            
        return song_maxlen, tag_maxlen
    
    def generate_input(self, mode, batch_size):
        def _encode(plylst, feature):
            
            if feature == 'tag':
                feat_idxs = list()
                for tag in plylst['tags']:
                    if self.tag2idx.setdefault(str(tag), False):
                        feat_idxs.append(self.tag2idx[str(tag)])
                    else:
                        feat_idxs.append(self.tag2idx['<unk>'])
                feat_maxlen = self.tag_maxlen
                n_feats = self.n_tags
            else:
                feat_idxs = list()
                for song in plylst['songs']:
                    if self.song2idx.setdefault(str(song), False):
                        feat_idxs.append(self.song2idx[str(song)])
                    else:
                        feat_idxs.append(self.song2idx['<unk>'])
                feat_maxlen = self.song_maxlen
                n_feats = self.n_songs

            feat_in = np.zeros((feat_maxlen,), dtype=np.int32)
            for i, idx in enumerate(feat_idxs):
                feat_in[i] = idx
            feat_out = np.zeros((n_feats,), dtype=np.float32)
            feat_out[feat_in[:n_feats]] = 1
            
            return feat_in, feat_out
        
        if mode == 'train':
            plylst_list = self.train_plylst_list
        elif mode == 'val':
            plylst_list = self.val_plylst_list
        elif mode == 'test':
            plylst_list = self.test_plylst_list
        else:
            raise(ValueError)
            
        song_in_list = list()
        tag_in_list = list()
        song_out_list = list()
        tag_out_list = list()
        for i, plylst in enumerate(plylst_list, 1):
            song_in, song_out = _encode(plylst, 'song')
            tag_in, tag_out = _encode(plylst, 'tag')
            song_in_list.append(song_in)
            tag_in_list.append(tag_in)
            song_out_list.append(song_out)
            tag_out_list.append(tag_out)
            
            if i%batch_size == 0:
                song_in = np.stack(song_in_list, axis=0)
                tag_in = np.stack(tag_in_list, axis=0)
                song_in_list = list()
                tag_in_list = list()
                
                song_out = np.stack(song_out_list, axis=0)
                tag_out = np.stack(tag_out_list, axis=0)
                song_out_list = list()
                tag_out_list = list()
                
                yield (song_in, tag_in), (song_out, tag_out)

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