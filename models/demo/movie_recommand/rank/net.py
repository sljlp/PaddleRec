# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np

import os

from paddle.distributed import moe
Moe_layer = moe.MoeLayer

class ExpertLayer(nn.Layer):
    def __init__(self, d_model, d_hidden, name=None,rank=0, windex = 0, num_expert=1):
        super(ExpertLayer, self).__init__()

        self.htoh4 = nn.Linear(d_model, d_hidden, \
            weight_attr=paddle.nn.initializer.KaimingUniform(), \
             bias_attr=nn.initializer.Constant(value=0.0))
        self.h4toh = nn.Linear(d_hidden, d_model, \
            weight_attr=paddle.nn.initializer.KaimingUniform(), \
             bias_attr=nn.initializer.Constant(value=0.0))
        self.htoh4.weight.name = "expert_" + self.htoh4.weight.name
        self.h4toh.weight.name = "expert_" + self.h4toh.weight.name
        self.htoh4.bias.name = "expert_" + self.htoh4.bias.name
        self.h4toh.bias.name = "expert_" + self.h4toh.bias.name
        self.act = paddle.nn.GELU()
        
    def forward(self, x):
        x = self.htoh4(x)
        # x = F.gelu(x, approximate=True, inplace=False)
        x = self.act(x)
        x = self.h4toh(x)
        return x

class ExpLayer2(nn.Layer):
    def __init__(self, fc_sizes, sizes, acts):
        super(ExpLayer2, self).__init__()
        self._layers = []
        for i in range(len(fc_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._layers.append(act)
            if acts[i] == 'sigmoid':
                act = paddle.nn.layer.Sigmoid()
                self.add_sublayer('act_%d' % i, act)
                self._layers.append(act)
        
    def forward(self, x):
        features = x
        for n_layer in self._layers:
            # if isinstance(n_layer, paddle.nn.Linear):
            #     print("weight shape:", n_layer.weight.shape)
            features = n_layer(features)
        return features

class DNNLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim, fc_sizes):
        super(DNNLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.fc_sizes = fc_sizes

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            padding_idx=0,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        sizes = [63] + self.fc_sizes + [1]
        acts = ["relu" for _ in range(len(self.fc_sizes))] + ["sigmoid"]

        if os.environ.get("USING_MOE", "False").lower() == "true":
            expert_count = int(os.environ.get("EXPERT_COUNT", 64))
            moe_layer_list = nn.LayerList()
            for _ in range(expert_count):
                expert = ExpertLayer(d_model = sizes[1], d_hidden=sizes[1])
                # expert = ExpLayer2(self.fc_sizes, sizes, acts)
                moe_layer_list.append(expert)
            self.moe_layer = Moe_layer(d_model=sizes[1], experts=moe_layer_list)
        else:
            self.moe_layer = None

        self._layers = []
        for i in range(len(self.fc_sizes) + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._layers.append(act)
            if acts[i] == 'sigmoid':
                act = paddle.nn.layer.Sigmoid()
                self.add_sublayer('act_%d' % i, act)
                self._layers.append(act)

    def forward(self, batch_size, user_sparse_inputs, mov_sparse_inputs,
                label_input):

        user_sparse_embed_seq = []
        for s_input in user_sparse_inputs:
            emb = self.embedding(s_input)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            user_sparse_embed_seq.append(emb)

        mov_sparse_embed_seq = []
        for s_input in mov_sparse_inputs:
            s_input = paddle.reshape(s_input, shape=[batch_size, -1])
            emb = self.embedding(s_input)
            emb = paddle.sum(emb, axis=1)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            mov_sparse_embed_seq.append(emb)

        features = paddle.concat(
            user_sparse_embed_seq + mov_sparse_embed_seq, axis=1)
        # if self.moe_layer:
        #     features = features.unsqueeze(0)
        #     features = self.moe_layer(features)
        #     features = features.squeeze(0)
            # print(features)
        for i, n_layer in enumerate(self._layers):
            if i == 1 and self.moe_layer:
                features = features.unsqueeze(0)
                features = self.moe_layer(features)
                features = features.squeeze(0)
            #if isinstance(n_layer, paddle.nn.Linear):
            #    print("weight shape:", n_layer.weight.shape)
            features = n_layer(features)

        predict = paddle.scale(features, scale=5)

        return predict
