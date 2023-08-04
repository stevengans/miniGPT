import math
import tensorflow as tf
from minigpt.model import GPT
from fastprogress import master_bar, progress_bar


class Trainer:
    def __init__(self, train_dataset, train_dataset_len, vocab_size, block_size):
        self.batch_size = 64
        self.block_size = block_size
        self.train_dataset = train_dataset.batch(self.batch_size)
        self.train_dataset_len = train_dataset_len
        self.max_epochs = 10
        self.betas = (0.9, 0.95)
        self.grad_norm_clip = 1.0
        self.warmup_tokens = 512 * 20
        self.final_tokens = 200 * train_dataset_len * self.block_size
        self.ckpt_path = None
        self.num_workers = 4
        self.tokens = 0
        self.gpt = GPT(vocab_size, self.block_size, n_embed=512, n_head=8, n_layer=8)
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=6e-4)
        self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def save_checkpoints(self):
        if self.ckpt_path is not None:
            self.gpt.model.save_weights(self.ckpt_path)

    def train_step(self, inputs):
        X, Y = inputs
        with tf.GradientTape() as tape:
            logits = self.gpt.model(X, training=True)
            num_labels = tf.shape(logits)[-1]
            label_mask = tf.math.logical_not(Y < 0)
            label_mask = tf.reshape(label_mask, (-1,))
            logits = tf.reshape(logits, (-1, num_labels))
            logits_masked = tf.boolean_mask(logits, label_mask)
            label_ids = tf.reshape(Y, (-1,))
            label_ids_masked = tf.boolean_mask(label_ids, label_mask)
            cross_entropy = self.cce(label_ids_masked, logits_masked)
            loss = tf.reduce_sum(cross_entropy) * (1.0 / self.batch_size)
        grads = tape.gradient(loss, self.gpt.model.trainable_variables)
        self.optimizer.apply_gradients(list(zip(grads, self.gpt.model.trainable_variables)))
        return cross_entropy

    def train(self):
        train_pb_max_len = math.ceil(float(self.train_dataset_len)/float(self.batch_size))
        epoch_bar = master_bar(range(self.max_epochs))
        for epoch in epoch_bar:
            for inputs in progress_bar(self.train_dataset, total=train_pb_max_len, parent=epoch_bar):
                loss = self.train_step(inputs)
                labels = inputs[-1]
                self.tokens += tf.reduce_sum(tf.cast(labels >= 0, tf.int32)).numpy()
                epoch_bar.child.comment = f'training loss : {loss}'
            print(f"Epoch {epoch+1}: train loss {loss:.5f}.")
            self.save_checkpoints()

    def sample(self, x, steps, temperature=1.0, sample=False, top_k=None):
        block_size = self.gpt.block_size
        for k in range(steps):
            x_cond = x if x.shape[1] <= block_size else x[:, -block_size:]
            logits = self.gpt.model(x_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = tf.where(tf.math.cumsum(tf.ones_like(logits)) <= top_k, logits, -float('inf'))
            probs = tf.nn.softmax(logits, axis=-1)
            if sample:
                ix = tf.random.categorical(logits, 1, dtype=tf.int32)
            else:
                _, ix = tf.math.top_k(probs, k=1)
            x = tf.concat((x, ix), axis=1)
        return x
