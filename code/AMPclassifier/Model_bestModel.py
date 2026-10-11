import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from multimodal_fusion_bestModel import MultiModalFeatureFusion

# ======================= 小工具 =======================
def _ensure_dense_edges(edges):
    if isinstance(edges, tf.RaggedTensor):
        edges = edges.to_tensor()
    return tf.cast(edges, tf.int32)


class TokenDropout(layers.Layer):
    """
    简化版：仅支持 rank-2 输入 [N, D]（我们在展平后就是这种形状），
    按 token（行）随机置零。
    """
    def __init__(self, rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)

    def call(self, x, training=False):
        if (not training) or self.rate <= 0.0:
            return x
        n = tf.shape(x)[0]                          # N
        keep = tf.cast(tf.random.uniform([n, 1]) > self.rate, x.dtype)  # [N,1]
        return x * keep


# =============================================================================
# Graph Attention (GAT) / MultiHead —— 支持“边权 + 边类型偏置 + 训练态 DropEdge”
# =============================================================================
class GraphAttention(layers.Layer):
    def __init__(self,
                 units,
                 n_edge_types=5,
                 seq_dropedge=0.3,
                 struct_dropedge=0.1,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=keras.regularizers.l2(1e-4),
                 attn_dropout=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.n_edge_types = n_edge_types
        self.seq_dropedge = float(seq_dropedge)
        self.struct_dropedge = float(struct_dropedge)
        self.ki = keras.initializers.get(kernel_initializer)
        self.kr = kernel_regularizer
        self.attn_drop = layers.Dropout(attn_dropout)

    def build(self, input_shape):
        # input_shape: ((N,D), (E,2), (E,), (E,))
        dim = input_shape[0][-1]
        self.W = self.add_weight("gat_W",
                                 shape=(dim, self.units),
                                 initializer=self.ki,
                                 regularizer=self.kr)
        self.a_src = self.add_weight("gat_a_src",
                                     shape=(self.units, 1),
                                     initializer=self.ki,
                                     regularizer=self.kr)
        self.a_dst = self.add_weight("gat_a_dst",
                                     shape=(self.units, 1),
                                     initializer=self.ki,
                                     regularizer=self.kr)
        self.bias = self.add_weight("gat_bias",
                                    shape=(1,),
                                    initializer="zeros")
        # 边权缩放系数
        self.edge_w_scale = self.add_weight("edge_w_scale",
                                            shape=(1,),
                                            initializer=keras.initializers.Constant(1.0))
        # 边类型偏置（Embedding 方式）
        self.edge_type_bias = self.add_weight("edge_type_bias",
                                              shape=(self.n_edge_types,),
                                              initializer="zeros")
        super().build(input_shape)

    def _drop_edges(self, edges, w, etype, training):
        if (not training) or edges.shape.rank is None:
            return edges, w, etype
        if edges.shape.rank != 2:
            return edges, w, etype
        if tf.shape(edges)[0] == 0:
            return edges, w, etype
        # etype: 0=顺序边；1..=结构边
        rand = tf.random.uniform(tf.shape(etype), dtype=tf.float32)
        keep_prob = tf.where(tf.equal(etype, 0),
                             1.0 - self.seq_dropedge,
                             1.0 - self.struct_dropedge)
        keep = rand < keep_prob
        edges = tf.boolean_mask(edges, keep)
        w = tf.boolean_mask(w, keep)
        etype = tf.boolean_mask(etype, keep)
        return edges, w, etype

    def call(self, inputs, training=False):
        # inputs: (x, edges, edge_w, edge_type)
        x, edges, edge_w, edge_type = inputs
        edges = _ensure_dense_edges(edges)  # [E,2]
        edge_w = tf.cast(edge_w, tf.float32)
        edge_type = tf.cast(edge_type, tf.int32)

        # 训练态 DropEdge（顺序边更高丢弃率）
        edges, edge_w, edge_type = self._drop_edges(edges, edge_w, edge_type, training)

        n = tf.shape(x)[0]
        h = x @ self.W  # [N, units]

        # + 自环（类型视为结构最近桶1；权重=1）
        self_loop = tf.range(n, dtype=edges.dtype)[:, None]
        self_loop = tf.concat([self_loop, self_loop], axis=1)  # [N,2]
        loop_w = tf.ones((n,), dtype=edge_w.dtype)
        loop_t = tf.ones((n,), dtype=edge_type.dtype)  # 当作结构短距（桶=1）

        edges = tf.concat([edges, self_loop], axis=0)
        edge_w = tf.concat([edge_w, loop_w], axis=0)
        edge_type = tf.concat([edge_type, loop_t], axis=0)

        src, dst = edges[:, 0], edges[:, 1]
        h_src = tf.gather(h, src)
        h_dst = tf.gather(h, dst)

        # 注意力打分：节点对 + 边权偏置 + 边类型偏置
        e = tf.nn.leaky_relu(
            tf.squeeze(h_src @ self.a_src, -1) +
            tf.squeeze(h_dst @ self.a_dst, -1) +
            self.bias
        )  # [E]
        e = e + tf.squeeze(self.edge_w_scale) * edge_w
        e = e + tf.gather(self.edge_type_bias, edge_type)

        # per-source 稳定 softmax
        seg_max = tf.math.unsorted_segment_max(e, src, n)
        e = e - tf.gather(seg_max, src)
        a = tf.exp(e)
        a = self.attn_drop(a, training=training)
        denom = tf.math.unsorted_segment_sum(a, src, n) + tf.keras.backend.epsilon()
        alpha = a / tf.gather(denom, src)

        out = tf.math.unsorted_segment_sum(h_dst * alpha[:, None], src, n)  # [N, units]
        return tf.nn.relu(out)


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads,
                 n_edge_types=5,
                 seq_dropedge=0.3,
                 struct_dropedge=0.1,
                 merge_type="concat",
                 **kwargs):
        super().__init__(**kwargs)
        self.heads = [
            GraphAttention(units,
                           n_edge_types=n_edge_types,
                           seq_dropedge=seq_dropedge,
                           struct_dropedge=struct_dropedge)
            for _ in range(num_heads)
        ]
        self.merge = merge_type

    def call(self, inputs, training=False):
        x, edges, edge_w, edge_type = inputs
        edges = _ensure_dense_edges(edges)
        outs = [h([x, edges, edge_w, edge_type], training=training) for h in self.heads]
        if self.merge == "concat":
            return tf.concat(outs, axis=-1)
        else:
            return tf.reduce_mean(tf.stack(outs, axis=-1), axis=-1)


# =============================================================================
# Readout: CLS Token 读出
# =============================================================================
class CLSReadout(layers.Layer):
    def __init__(self, num_heads, embed_dim, dense_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.cls_token = self.add_weight(
            "cls_token", shape=(1, 1, embed_dim),
            initializer="zeros"
        )
        self.attn = layers.MultiHeadAttention(num_heads, key_dim=embed_dim // num_heads, dropout=dropout)
        self.n1 = layers.LayerNormalization()
        self.n2 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(dense_dim, activation="relu",
                         kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.Dense(embed_dim, kernel_regularizer=keras.regularizers.l2(1e-4))
        ])
        self.do = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x, mol_ind = inputs  # x:[N,D], mol_ind:[N]
        mol_ind = tf.cast(mol_ind, tf.int32)
        nb = tf.reduce_max(mol_ind) + 1
        lengths = tf.math.bincount(mol_ind, minlength=nb, maxlength=nb)
        rag = tf.RaggedTensor.from_row_lengths(x, lengths)
        padded = rag.to_tensor(shape=[nb, tf.reduce_max(lengths), tf.shape(x)[-1]])  # [B,T,D]
        key_mask = tf.sequence_mask(lengths, maxlen=tf.shape(padded)[1])            # [B,T]

        B = tf.shape(padded)[0]
        cls = tf.tile(self.cls_token, [B, 1, 1])  # [B,1,D]

        attn_mask = tf.expand_dims(key_mask, axis=1)  # [B,1,T]

        y = self.attn(query=cls, value=padded, key=padded, attention_mask=attn_mask, training=training)  # [B,1,D]
        y = self.n1(cls + y)
        y = self.do(y, training=training)
        z = self.n2(y + self.ffn(y, training=training))  # [B,1,D]
        return tf.squeeze(z, axis=1)  # [B,D]


# =============================================================================
# GATModel: 融合→TokenDrop→GAT(残差, 支持边权/类型/DropEdge)→CLS Readout
# =============================================================================
class GATModel(keras.Model):
    def __init__(self,
                 u50_dim, dim_aaindex, dim_paac, dim_pc6,
                 dim_blosum62, dim_onehot,
                 conv_channels_aaindex,
                 conv_channels_paac, conv_channels_pc6,
                 conv_channels_blosum62, conv_channels_onehot,
                 lstm_units, hidden_units, num_heads, num_layers,
                 token_drop_rate=0.1,
                 n_edge_types=5,
                 seq_dropedge=0.3,
                 struct_dropedge=0.1,
                 batch_size=32, num_classes=1, **kwargs):
        super().__init__(**kwargs)

        self.seq_dim = (
            u50_dim + dim_aaindex + dim_paac +
            dim_pc6 + dim_blosum62 + dim_onehot
        )

        self.fusion = MultiModalFeatureFusion(
            u50_dim=u50_dim,
            dim_aaindex=dim_aaindex,
            dim_paac=dim_paac,
            dim_pc6=dim_pc6,
            dim_blosum62=dim_blosum62,
            dim_onehot=dim_onehot,
            conv_channels_aaindex=conv_channels_aaindex,
            conv_channels_paac=conv_channels_paac,
            conv_channels_pc6=conv_channels_pc6,
            conv_channels_blosum62=conv_channels_blosum62,
            conv_channels_onehot=conv_channels_onehot,
            lstm_units=lstm_units
        )

        self.fused_dim = hidden_units * num_heads
        self.seq_proj = layers.Dense(self.fused_dim, activation="relu",
                                     kernel_regularizer=keras.regularizers.l2(1e-4))

        self.token_drop = TokenDropout(rate=token_drop_rate)

        # 用显式层替代 Sequential，避免图模式下的秩推断问题
        self.pre_dense = layers.Dense(self.fused_dim, activation="relu",
                                      kernel_regularizer=keras.regularizers.l2(1e-4))
        self.pre_do = layers.Dropout(0.1)

        self.gat_layers = []
        for _ in range(num_layers):
            self.gat_layers.append(
                MultiHeadGraphAttention(hidden_units, num_heads,
                                        n_edge_types=n_edge_types,
                                        seq_dropedge=seq_dropedge,
                                        struct_dropedge=struct_dropedge)
            )
            self.gat_layers.append(layers.Dropout(0.1))

        self.readout = CLSReadout(
            num_heads=num_heads,
            embed_dim=self.fused_dim,
            dense_dim=self.fused_dim * 4
        )

        self.classifier = layers.Dense(
            num_classes, activation="sigmoid",
            kernel_regularizer=keras.regularizers.l2(1e-4)
        )

    def call(self, inputs, training=False):
        # inputs: (node_rt, edges, edge_w, edge_type, mol_ind)
        node_rt, edges, edge_w, edge_type, mol_ind = inputs

        # Ragged -> Dense（仅为分割 seq_feats）
        full = node_rt.to_tensor(default_value=0.0)            # [B, T, F]
        row_lengths = node_rt.row_lengths()                    # [B]

        seq_feats = full[..., :self.seq_dim]                   # [B,T,S]

        # 多模态融合 + 投影
        fused_seq = self.fusion(seq_feats)                     # [B,T,2*lstm_units]
        fused_seq = self.seq_proj(fused_seq)                   # [B,T,fused_dim]

        # Dense -> Ragged -> flat
        rag = tf.RaggedTensor.from_tensor(fused_seq, lengths=row_lengths)
        x = rag.flat_values                                     # [N, fused_dim]
        # 强制静态约束最后一维，避免后续层的形状不确定
        x = tf.ensure_shape(x, [None, self.fused_dim])

        # TokenDrop + 前投影
        x = self.token_drop(x, training=training)
        x = self.pre_dense(x)                  # Dense 不区分 training
        x = self.pre_do(x, training=training)

        # 多层 GAT（支持边权/类型 + DropEdge）
        for layer in self.gat_layers:
            if isinstance(layer, MultiHeadGraphAttention):
                res = layer([x, edges, edge_w, edge_type], training=training)
                x = x + res
            else:
                x = layer(x, training=training)

        # Readout + 分类
        emb = self.readout([x, mol_ind], training=training)
        return self.classifier(emb)
