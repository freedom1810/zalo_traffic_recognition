import torch
from torch.utils.data.dataset import Dataset

import numpy as np
import six
import pandas as pd

class BengaliAIDataset(Dataset):
    def __init__(self, 
                    path = '', 
                    transform=None, 
                    indices=None,
                    # mode = 'train' #train valid inference 
                    ):
        
        self.path = path


        self.load_data()
        
        self.transform = transform
        
        if indices is None:
            indices = np.arange(len(self.images))

        self.indices = indices
        self.train = self.labels is not None
    
    def __getitem__(self, index):
        """Returns an example or a sequence of examples."""
        if torch.is_tensor(index):
            index = index.tolist()

        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))

            return [self.get_example_wrapper(i) for i in
                    six.moves.range(current, stop, step)]

        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example_wrapper(i) for i in index]

        else:
            return self.get_example_wrapper(index)
    
    def __len__(self):
        """return length of this dataset"""
        return len(self.indices)
    
    def get_example_wrapper(self, i):
        """Wrapper of `get_example`, to apply `transform` if necessary"""
        example = self.get_example(i)

        if self.transform:
            example = self.transform(example)
        return example

    def get_example(self, i):
        """Return i-th data"""
        i = self.indices[i]
        x = self.images[i]
        # Opposite white and black: background will be white and
        # for future Affine transformations
        if self.train:
            y = self.labels[i]
            return x, y
        else:
            return x

    def load_data(self):
        #train
        df = pd.read_csv(self.path)
        self.labels = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
        images = []
        images = df['image_id'].values

        self.images = np.array(images)

    @staticmethod
    def load_data_1295(train_path = '/home/asilla/sonnh/k/data/train_split.csv'):

        #train
        train = pd.read_csv(train_path)
        train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
        
        train_images = train['image_id'].values
        train_paths = train['image_id'].values

        labels = set()
        for i, path in enumerate(train_paths):
            labels.add(str(train_labels[i][0]) + '_' + str(train_labels[i][1]) + '_' + str(train_labels[i][2]))

        labels = list(labels)
        final_train_labels = []
        for i, path in enumerate(train_paths):
            label = str(train_labels[i][0]) + '_' + str(train_labels[i][1]) + '_' + str(train_labels[i][2])

            final_train_labels.append(labels.index(label))

        return train_images, np.array(final_train_labels)