# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import numpy as np
from torch.utils.data import Dataset

from semilearn.datasets.utils import get_onehot


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for FixMatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 is_ulb=False,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets
        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot
        self.transform = None

    def random_choose_sen(self):
        return random.randint(1,2)
    
    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """

        # set idx-th target
        if self.targets is None:
            target = None
            
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)
        sen = self.data[idx]
        
        # set augmented images
        if self.is_ulb == False:
            return {'idx':idx, 'text':sen[0], 'label':target} 
        else:
            return {'idx':idx, 'text':sen[0], 'text_s':sen[self.random_choose_sen()], 'lab':target}
                
    def __len__(self):
        return len(self.data)