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
        self.vector_size=256
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

            # feat_included
            random.shuffle(feat_idxs)
            n_included = len(feat_idxs) // 2
            feat_included = feat_idxs[:n_included]

            # feat_input: padded label encoding(feat_included)
            feat_input = np.zeros((feat_maxlen,), dtype=np.int32)
            feat_input[:n_included] = feat_included

            # feat_label: one-hot encoding(feat_included & feat_excluded)
            feat_label = np.zeros((n_feats,), dtype=np.float64)
            feat_label[feat_idxs] = 1
            
            return feat_input, feat_label
        
        if mode == 'train':
            plylst_list = self.train_plylst_list
        elif mode == 'val':
            plylst_list = self.val_plylst_list
        elif mode == 'test':
            plylst_list = self.test_plylst_list
        else:
            raise(ValueError)
            
        song_input_list = list()
        tag_input_list = list()
        song_label_list = list()
        tag_label_list = list()
        for i, plylst in enumerate(plylst_list, 1):
            song_input, song_label = _encode(plylst, 'song')
            tag_input, tag_label = _encode(plylst, 'tag')
            song_input_list.append(song_input)
            tag_input_list.append(tag_input)
            song_label_list.append(song_label)
            tag_label_list.append(tag_label)
            
            if i%batch_size == 0:
                song_inputs = np.stack(song_input_list, axis=0)
                tag_inputs = np.stack(tag_input_list, axis=0)
                song_input_list = list()
                tag_input_list = list()
                
                song_labels = np.stack(song_label_list, axis=0)
                tag_labels = np.stack(tag_label_list, axis=0)
                song_label_list = list()
                tag_label_list = list()
                
                yield (song_inputs, tag_inputs), (song_labels, tag_labels)