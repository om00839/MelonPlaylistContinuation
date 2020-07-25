import os
import sys
import random

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as K

from .dataset import MPCDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append("/content/drive/My Drive/MelonPlaylistContinuation")


DATA_DIR = "/content/drive/My Drive/MelonPlaylistContinuation/data"
INFO_DIR = "/content/drive/My Drive/MelonPlaylistContinuation/info"
CKPT_DIR = "/content/drive/My Drive/MelonPlaylistContinuation/checkpoint/d2v_dense"
if not os.path.isdir(CKPT_DIR):
    os.mkdir(CKPT_DIR)
SEED = 200722
dataset = MPCDataset(DATA_DIR, INFO_DIR, vector_size=256)

generator = iter(dataset.generate_input('train', batch_size=1))
song_plylst_vec_list = list()
tag_plylst_vec_list = list()
song_labels_list = list()
tag_labels_list = list()

for step in range(N):
    (song_inputs, tag_inputs), (song_labels, tag_labels) = next(generator)
    song_nonzero = np.count_nonzero(song_inputs)
    tag_nonzero = np.count_nonzero(tag_inputs)

    song_doc = [dataset.idx2song[song_idx] for song_idx in song_inputs[:song_nonzero]]
    tag_doc = [dataset.tag[tag_idx] for tag_idx in tag_inputs[:tag_nonzero]]

    song_plylst_vec = dataset.song_tag_d2v.infer_vector(song_doc)
    song_plylst_vec_list.append(song_plylst_vec)
    tag_plylst_vec = dataset.tag_song_d2v.infer_vector(tag_doc)
    tag_plylst_vec_list.append(tag_plylst_vec)

    song_labels_list.append(song_labels)
    tag_labels_list.append(tag_labels)

X_song = np.stack(song_plylst_vec_list, axis=0)
X_song_train, X_song_val = train_test_split(X_song, test_size=0.3, random_state=SEED)
X_tag = np.stack(tag_plylst_vec_list, axis=0)
X_tag_train, X_tag_val = train_test_split(X_tag, test_size=0.3, random_state=SEED)

y_song = np.stack(song_labels_list, axis=0)
y_song_train, y_song_val = train_test_split(y_song, test_size=0.3, random_state=SEED)
y_tag = np.stack(tag_labels_list, axis=0)
y_tag_train, y_tag_val = train_test_split(y_tag, test_size=0.3, random_state=SEED)

# define model, opt
song_model = K.Sequential()
song_model.add(K.layers.Dense(128, input_shape=(256,)))
song_model.add(K.layers.Activation('relu'))
song_model.add(K.layers.Dense(64))
song_model.add(K.layers.Activation('relu'))
song_model.add(K.layers.Dense(dataset.n_songs))
song_opt = K.optimizers.Adam()
song_model.compile(optimizer=song_opt, loss="binary_crossentropy")

song_ckpt_path = os.path.join(CKPT_DIR, 'song', 'checkpoint')
song_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=song_ckpt_path,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

# train model
song_history = song_model.fit(
    X_song_train, y_song_train, 
    batch_size=512, 
    epochs=30, 
    verbose=1,
    validation_data = (X_song_val, y_song_val),
    shuffle = True,
    callbacks = [song_model_checkpoint_callback]
)

tag_model = K.Sequential()
tag_model.add(K.layers.Dense(128, input_shape=(256,)))
song_model.add(K.layers.Activation('relu'))
tag_model.add(K.layers.Dense(64))
song_model.add(K.layers.Activation('relu'))
tag_model.add(K.layers.Dense(dataset.n_tags))
tag_opt = K.optimizers.Adam()
tag_model.compile(optimizer=tag_opt, loss="binary_crossentropy")

tag_ckpt_path = os.path.join(CKPT_DIR, 'tag', 'checkpoint')
tag_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=tag_ckpt_path,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

# train model
tag_history = tag_model.fit(
    X_tag_train, y_tag_train, 
    batch_size=512, 
    epochs=30, 
    verbose=1,
    validation_data = (X_tag_val, y_tag_val),
    shuffle = True,
    callbacks = [tag_model_checkpoint_callback]
)