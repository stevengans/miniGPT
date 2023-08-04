import numpy as np
import tensorflow as tf
import math


class CharDataset:

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("Unique CHARS: ", chars)
        print("Text Size: ", data_size)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __iter__(self):
        for _ in range(self.__len__()):
            i = np.random.randint(0, len(self.data) - (self.block_size + 1))
            chunk = self.data[i:i + self.block_size + 1]
            dix = [self.stoi[s] for s in chunk]
            x = tf.convert_to_tensor(dix[:-1], dtype=tf.int32)
            y = tf.convert_to_tensor(dix[1:], dtype=tf.int32)
            yield x, y

    __call__ = __iter__


class TikTokenDataset:
    def __init__(self, block_size):
        self.data = np.memmap('train.bin', dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.vocab_size = 50304

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __iter__(self):
        for _ in range(self.__len__()):
            i = np.random.randint(0, len(self.data) - (self.block_size + 1))
            dix = self.data[i:i + self.block_size + 1]
            x = tf.convert_to_tensor(dix[:-1], dtype=tf.int32)
            y = tf.convert_to_tensor(dix[1:], dtype=tf.int32)
            yield x, y

    __call__ = __iter__
