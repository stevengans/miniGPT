import tensorflow as tf
import tiktoken
from minigpt.trainer import Trainer
from minigpt.tokenizer import CharDataset, TikTokenDataset


data_type = 'char'
block_size = 128
if data_type == 'char':
    text = open('data/tiny_shakespeare.txt', 'r').read()
    train_dataset_gen = CharDataset(text, block_size)
else:
    train_dataset_gen = TikTokenDataset(block_size)
train_dataset = tf.data.Dataset.from_generator(train_dataset_gen, (tf.int32, tf.int32))
trainer = Trainer(train_dataset, len(train_dataset_gen), train_dataset_gen.vocab_size, block_size)
trainer.train()

if data_type == 'char':
    context = "O God, O God!"
    x = tf.convert_to_tensor([train_dataset_gen.stoi[s] for s in context], dtype=tf.int32)[None, ...]
    y = trainer.sample(x, 2000, temperature=0.9, sample=True, top_k=5)[0]
    completion = ''.join([train_dataset_gen.itos[int(i)] for i in y])
    print(completion)
else:
    prompt = "O God, O God!"
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    context = encode(prompt)
    x = tf.convert_to_tensor(context, dtype=tf.int32)[None, ...]
    y = trainer.sample(x, 100, temperature=0.9, sample=True, top_k=5)[0]
    completion = decode(y.numpy())
    print(completion)
