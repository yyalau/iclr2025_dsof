# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple
from torchvision import transforms


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, n_tasks=1, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        # self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        # self.attributes = ['examples', 'labels', 'logits', 'task_labels']
    def is_full(self):
        return self.num_seen_examples >= self.buffer_size

    def init_tensors(self, **kwargs) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        self.attributes = list(kwargs.keys())
        for attr_str, attr in kwargs.items():
            # attr = eval(attr_str)
            if isinstance(attr, torch.Tensor) and not hasattr(self, attr_str):
                setattr(self, attr_str, torch.zeros(
                    (self.buffer_size, *attr.shape[1:]),
                    dtype=attr.dtype, 
                    device=attr.device))

    def add_data(self, batch_x, **kwargs):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        attributes = {'batch_x': batch_x}
        attributes.update(**kwargs)
        
        if not hasattr(self, 'batch_x'):
            self.init_tensors(**attributes)
        
        n = batch_x.shape[0]

        for i in range(n):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                for key, value in attributes.items():
                    getattr(self, key)[index] = value[i]
                # self.examples[index] = examples[i].to(self.device)
                # if labels is not None:
                #     self.labels[index] = labels[i].to(self.device)
                # if logits is not None:
                #     self.logits[index] = logits[i].to(self.device)
                # if task_labels is not None:
                #     self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, ) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        n = min(self.num_seen_examples, self.batch_x.shape[0])
        if size > n:
            size = n

        choice = np.random.choice(n, size=size, replace=False)
        # if transform is None: transform = lambda x: x
        # ret_tuple = (torch.stack([transform(ee.cpu())
        #                     for ee in self.examples[choice]]).to(self.device),)
        ret_tuple = ()
        # import ipdb; ipdb.set_trace()
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        # if transform is None: transform = lambda x: x
        # ret_tuple = (torch.stack([transform(ee.cpu())
        #                     for ee in self.examples]).to(self.device),)
        ret_tuple = ()

        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0



class BufferFIFO:
    """
    """
    def __init__(self, buffer_size, n_tasks=1, ):
        self.buffer_size = buffer_size
        self.num_seen_examples = 0

    def is_full(self):
        return self.num_seen_examples >= self.buffer_size

    def init_tensors(self, **kwargs) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        self.attributes = list(kwargs.keys())
        for attr_str, attr in kwargs.items():
            # attr = eval(attr_str)
            if isinstance(attr, torch.Tensor) and not hasattr(self, attr_str):
                setattr(self, attr_str, torch.zeros(
                    (self.buffer_size, *attr.shape[1:]),
                    dtype=attr.dtype, 
                    device=attr.device))

    def add_data(self, batch_x, **kwargs):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        attributes = {'batch_x': batch_x}
        attributes.update(**kwargs)
        
        if not hasattr(self, 'batch_x'):
            self.init_tensors(**attributes)
        
        n = batch_x.shape[0]

        for i in range(n):
            self.num_seen_examples += 1
            if self.is_full():
                for key, value in attributes.items():
                    
                    # import ipdb; ipdb.set_trace()
                    getattr(self, key)[self.num_seen_examples % self.buffer_size] = value[i]

    def get_data(self, size: int, ) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        n = min(self.num_seen_examples, self.batch_x.shape[0])
        if size > n:
            size = n

        choice = np.random.choice(n, size=size, replace=False)
        # if transform is None: transform = lambda x: x
        # ret_tuple = (torch.stack([transform(ee.cpu())
        #                     for ee in self.examples[choice]]).to(self.device),)
        ret_tuple = ()
        # import ipdb; ipdb.set_trace()
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        # if transform is None: transform = lambda x: x
        # ret_tuple = (torch.stack([transform(ee.cpu())
        #                     for ee in self.examples]).to(self.device),)
        ret_tuple = ()

        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
