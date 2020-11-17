import paddle.fluid as fluid
from paddle_fl.paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory



class Model(object):
    def __init__(self):
        pass

    def scs_network(self):
        self.inputs = fluid.layers.data(name='x', shape=[1, 7], dtype="float32")
        self.label = fluid.layers.data(name='label', shape=[1], dtype='float32')
        self.fc1 = fluid.layers.fc(input=self.inputs, size=7, act='relu')
        self.batch1 = fluid.layers.batch_norm(input=self.fc1)
        self.fc2 = fluid.layers.fc(input=self.batch1, size=256, act='relu')
        self.drop1 = fluid.layers.dropout(x=self.fc2, dropout_prob=0.2)
        self.batch2 = fluid.layers.batch_norm(input=self.drop1)
        self.fc3 = fluid.layers.fc(input=self.batch2, size=64, act='relu')
        self.drop2 = fluid.layers.dropout(x=self.fc3, dropout_prob=0.2)
        self.predict = fluid.layers.fc(input=self.drop2, size=5, act='softmax')
        self.sum_cost = fluid.layers.cross_entropy(
            input=self.predict, label=self.label)
        self.accuracy = fluid.layers.accuracy(input=self.predict, label=self.label)
        self.loss = fluid.layers.reduce_mean(self.sum_cost)
        self.startup_program = fluid.default_startup_program()

    def csa_network(self):
        self.inputs = fluid.layers.data(
            name='x', shape=[1, 14], dtype="float32")
        self.label = fluid.layers.data(name='label', shape=[1], dtype='float32')
        # self.concat = fluid.layers.concat(self.inputs, axis=1)
        self.fc1 = fluid.layers.fc(input=self.inputs, size=28, act='relu')
        self.predict = fluid.layers.fc(input=self.fc1, size=2, act='sigmoid')
        self.sum_cost = fluid.layers.cross_entropy(input=self.predict, label=self.label)
        self.accuracy = fluid.layers.accuracy(input=self.predict, label=self.label)
        self.loss = fluid.layers.reduce_mean(self.sum_cost)
        self.startup_program = fluid.default_startup_program()


model = Model()
model.car_network()

job_generator = JobGenerator()
optimizer = fluid.optimizer.SGD(learning_rate=0.001)
job_generator.set_optimizer(optimizer)
job_generator.set_losses([model.loss])
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names(
    [model.inputs.name, model.label.name],
    [model.loss.name, model.accuracy.name])

build_strategy = FLStrategyFactory()
build_strategy.fed_avg = True
build_strategy.inner_step = 10
strategy = build_strategy.create_fl_strategy()

# endpoints will be collected through the cluster
# in this example, we suppose endpoints have been collected
endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=2, output=output)
# fl_job_config will  be dispatched to workers

