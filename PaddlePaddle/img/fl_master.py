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

import paddle.fluid as fluid
import paddle_fl.paddle_fl as fl
from paddle_fl.paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory


def conv_pool(input, num_filters, filter_size, pool_size, pool_stride, act="relu"):
        conv_out = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            act=act)
        out = fluid.layers.pool2d(
            input=conv_out,
            pool_size=pool_size,
            pool_stride=pool_stride)
        return out

class Model(object):
    def __init__(self):
        pass


    def cnn(self):
        self.inputs = fluid.layers.data(
            name='img', shape=[1, 150, 150], dtype="float32")
        self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        self.conv_pool_1 = conv_pool(
            input=self.inputs,
            filter_size=5,
            num_filters=20,
            pool_size=2,
            pool_stride=2,
            act='relu')
        self.conv_pool_1 = fluid.layers.batch_norm(self.conv_pool_1) 
        #self.droped1 = fluid.layers.dropout(x = self.conv_pool_1, dropout_prob=0.5)
        self.conv_pool_2 = fluid.nets.simple_img_conv_pool(
            input=self.conv_pool_1,
            filter_size=5,
            num_filters=50,
            pool_size=2,
            pool_stride=2,
            act='relu')
        self.conv_pool_2 = fluid.layers.batch_norm(self.conv_pool_2) 
        #self.droped2 = fluid.layers.dropout(x = self.conv_pool_2, dropout_prob=0.2)
        self.conv_pool_3 = fluid.nets.simple_img_conv_pool(
            input=self.conv_pool_2,
            filter_size=5,
            num_filters=70,
            pool_size=2,
            pool_stride=2,
            act='relu')
        self.conv_pool_3 = fluid.layers.batch_norm(self.conv_pool_3)
        self.droped3 = fluid.layers.dropout(x = self.conv_pool_3, dropout_prob=0.2) 
        self.fc1 = fluid.layers.fc(input=self.droped3, size=512, act=None)
        self.bn = fluid.layers.batch_norm(input=self.fc1, act='relu')
        self.drop1 = fluid.layers.dropout(x=self.bn, dropout_prob=0.5)
        self.fc2 = fluid.layers.fc(input=self.drop1, size=512, act=None)
        self.predict = self.predict = fluid.layers.fc(input=self.fc2,
                                                      size=2,
                                                      act='softmax')
        self.cost = fluid.layers.cross_entropy(
            input=self.predict, label=self.label)
        self.accuracy = fluid.layers.accuracy(
            input=self.predict, label=self.label)
        self.loss = fluid.layers.mean(self.cost)
        self.startup_program = fluid.default_startup_program()


model = Model()
model.cnn()

job_generator = JobGenerator()
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
job_generator.set_optimizer(optimizer)
job_generator.set_losses([model.loss])
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names(
    [model.inputs.name, model.label.name],
    [model.loss.name, model.accuracy.name])

build_strategy = FLStrategyFactory()
build_strategy.fed_avg = True
build_strategy.inner_step = 1
strategy = build_strategy.create_fl_strategy()

endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=2, output=output)
