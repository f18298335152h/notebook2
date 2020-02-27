#!/usr/bin/env python
# coding=utf-8
import mxnet as mx
import numpy as np
import logging 

logging.basicConfig(level=logging.DEBUG)

def get_symbol(num_classes):
    input_data = mx.sym.Variable(name="data")
    conv1 = mx.sym.Convolution(name='conv1',data=input_data, kernel=(11, 11), stride=(4, 4), num_filter=96)
    relu1 = mx.sym.Activation(data=conv1, act_type="relu")
    lrn1 = mx.sym.LRN(data=relu1, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    pool1 = mx.sym.Pooling(data=lrn1, pool_type="max", kernel=(3, 3), stride=(2,2))
    #stage2
    conv2 = mx.sym.Convolution(name='conv2',data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.sym.Activation(data=conv2, act_type="relu")
    lrn2 = mx.sym.LRN(data=relu2, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
    pool2 = mx.sym.Pooling(data=lrn2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    #stage3
    conv3 = mx.sym.Convolution(name='conv3',data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.sym.Activation(data=conv3, act_type="relu")
    conv4 = mx.sym.Convolution(name='conv4',data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.sym.Activation(data=conv4, act_type="relu")
    conv5 = mx.sym.Convolution(name='conv5',data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.sym.Activation(data=conv5, act_type="relu")
    pool3 = mx.sym.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    #stage4
    flatten = mx.sym.Flatten(data=pool3)
    fc1 = mx.sym.FullyConnected(name='fc1', data=flatten, num_hidden=4096)
    relu6 = mx.sym.Activation(data=fc1, act_type="relu")
    dropout1 = mx.sym.Dropout(data=relu6, p=0.5)
    #stage5
    fc2 = mx.sym.FullyConnected(name='fc2', data=dropout1, num_hidden=4096)
    relu7 = mx.sym.Activation(data=fc2, act_type="relu")
    dropout2 = mx.sym.Dropout(data=relu7, p=0.5)
    #stage6
    fc3 = mx.sym.FullyConnected(name='fc3', data=dropout2, num_hidden=num_classes)
    softmax = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
    return softmax

def train():
    batch_size = 1000
    num_classes = 10574
    num_examples = 455576 
    epoch_size = num_examples / batch_size
    lr_sch = mx.lr_scheduler.FactorScheduler(step=10, factor=0.9)
    optimizer_params = {
                'learning_rate': 0.1,
                'lr_scheduler': lr_sch
                }
    train = mx.io.ImageRecordIter(
                path_imgrec = 'train_train.rec',
                data_shape = (3,227,227),
                batch_size = batch_size,
                shuffle = True,
                )
    val = mx.io.ImageRecordIter(
                path_imgrec = 'train_val.rec',
                data_shape = (3,227,227),
                batch_size = batch_size,
                shuffle = True,
        )
    model = mx.mod.Module(
                context = [mx.gpu(i) for i in range(4)],
                symbol = get_symbol(num_classes),
                )
    model.fit(
                train_data = train,
                begin_epoch=0,
                num_epoch = 20,
                eval_data = val,
                optimizer = 'sgd',
                optimizer_params = optimizer_params,
                initializer = mx.init.Xavier(),
               batch_end_callback = mx.callback.Speedometer(batch_size,1),
                )
if __name__=='__main__':
    train()