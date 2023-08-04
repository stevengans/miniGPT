import tensorflow as tf


class GPT:
    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_head = n_head
        self.depth = self.n_embed // self.n_head
        self.n_layer = n_layer
        self.wq = tf.keras.layers.Dense(self.n_embed, name="query")
        self.wk = tf.keras.layers.Dense(self.n_embed, name="key")
        self.wv = tf.keras.layers.Dense(self.n_embed, name="value")
        self.attn_drop = tf.keras.layers.Dropout(rate=0.1)
        self.resid_drop = tf.keras.layers.Dropout(rate=0.1)
        self.dense = tf.keras.layers.Dense(self.n_embed, name="projection")
        self.ffn = self.point_wise_feed_forward_network()
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.tok_emb = tf.keras.layers.Embedding(vocab_size, self.n_embed)
        self.pos_emb = tf.keras.layers.Embedding(self.block_size, self.n_embed)
        self.drop = tf.keras.layers.Dropout(0.1)
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.head = tf.keras.layers.Dense(self.vocab_size)
        self.model = None
        self.build()

    def point_wise_feed_forward_network(self):
        return tf.keras.Sequential([tf.keras.layers.Dense(self.n_embed * 4, activation='gelu'),
                                    tf.keras.layers.Dense(self.n_embed),
                                    tf.keras.layers.Dropout(0.1)
                                    ])

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_head, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def multi_head_attention(self, x, mask, training):
        batch_size = tf.shape(x)[0]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.attn_drop(attention_weights, training=training)
        scaled_attention = tf.matmul(attention_weights, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.n_embed))
        output = self.dense(concat_attention)
        output = self.resid_drop(output, training=training)
        return output

    def block(self, x, mask, training):
        x = x + self.multi_head_attention(self.layernorm1(x), mask, training=training)
        x = x + self.ffn(self.layernorm2(x), training=training)
        return x

    def build(self, training=False):
        input_layer = tf.keras.Input(shape=(None,))
        t = tf.shape(input_layer)[1]
        token_embeddings = self.tok_emb(input_layer)
        pos = tf.range(0, tf.shape(token_embeddings)[1])
        position_embeddings = self.pos_emb(pos)
        x = self.drop(token_embeddings + position_embeddings, training=training)
        mask = 1 - tf.linalg.band_part(tf.ones((t, t)), -1, 0)
        for i in range(self.n_layer):
            x = self.block(x, mask, training=training)
        x = self.ln_f(x)
        logits = self.head(x)
        self.model = tf.keras.Model(inputs=input_layer, outputs=logits)
        return
