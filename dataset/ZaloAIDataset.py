import torch
from torch.utils.data.dataset import Dataset

import numpy as np
import six
import pandas as pd

class ZaloAIDataset(Dataset):
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
        self.labels = df['label'].values
        images = []
        images = df['image_file_name'].values

        self.images = np.array(images)