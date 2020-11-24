import torch as t
import torch.nn as nn
import numpy as np

class PadCollate:
    """
    a variant of collate_fn that zero pads according to the longest sequence in a batch of sequence
    """
    def __init__(self, dim=0):
        """
        Args:
            dim: the dimension to be padded (dimensions of time in sequence)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        Args:
            batch: list of tensor

        Returns:
            xs: a tensor of all examples in 'batch' after padding
            ys: a LongTensor of all labels in Batch
        """
        ## the original length of samples in batch
        org_len = list(map(lambda x: x[0].shape[self.dim], batch))
        #
        ## find the longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        ## pad according to max_len
        batch = list(map(lambda x: (self.pad_tensor(x[0], padding=max_len, dim=self.dim), x[1]), batch))
        xs = t.stack(list(map(lambda x: x[0], batch)), dim=self.dim)
        # used pack_padded_sequence
        xs = nn.utils.rnn.pack_padded_sequence(xs, org_len, batch_first=True)
        ys = t.LongTensor(list(map(lambda x: x[1], batch)))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

    def pad_tensor(self, tensor, padding, dim):
        """
        Args:
            tensor: tensor to pad
            padding: the size to pad to
            dim: dimension to pad

        Returns: the padded tensor
        """
        pad_size = list(tensor.shape)
        pad_size[dim] = padding - tensor.size(dim)
        return t.cat([tensor, t.zeros(*pad_size, dtype=t.double)], dim=dim)

