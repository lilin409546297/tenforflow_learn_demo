# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
    模型的保存与训练加载：
        tf.train.Saver(<var_list>,<max_to_keep>)
            var_list: 指定要保存和还原的变量,作为一个dict或者list传递
            max_to_keep: 指示要保留的最大检查点文件个数。
            保存模型的文件：checkpoint文件/检查点文件
            method:
                save(<session>, <path>)
                restore(<session>, <path>)
    模型的独立加载：
        1、tf.train.import_meta_graph(<meta_graph_or_file>) 读取训练时的数据流图
            meta_graph_or_file: *.meta的文件
        2、saver.restore(<session>, tf.train.latest_checkpoint(<path>)) 加载最后一次检测点
            path: 含有checkpoint的上一级目录
        3、graph = tf.get_default_graph() 默认图谱
            graph.get_tensor_by_name(<name>) 获取对应数据传入占位符
                name: tensor的那么名称，如果没有生命name,则为(placeholder:0), 数字0依次往后推
            graph.get_collection(<name>) 获取收集集合
                return tensor列表
            补充：
                如果不知道怎么去获取tensor的相关图谱，可以通过
                graph.get_operations() 查看所有的操作符，最好断点查看
"""
import os
import tensorflow as tf

def model_save():
    # 1、准备特征值和目标值
    with tf.variable_scope("data"):
        # 占位符，用于数据传入
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x")
        # 矩阵相乘必须是二维(为了模拟效果而设定固定值来训练)
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    # 2、建立回归模型，随机给权重值和偏置的值，让他去计算损失，然后在当前状态下优化
    with tf.variable_scope("model"):
        # 模型 y = wx + b, w的个数根据特征数据而定，b随机
        # 其中Variable的参数trainable可以指定变量是否跟着梯度下降一起优化(默认True)
        w = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name="w", trainable=True)
        b = tf.Variable(0.0, name="b")
        # 预测值
        y_predict = tf.matmul(x, w) + b

    # 3、建立损失函数，均方误差
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 4、梯度下降优化损失
    with tf.variable_scope("optimizer"):
        # 学习率的控制非常重要，如果过大会出现梯度消失/梯度爆炸导致NaN
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    # 收集需要用于预测的模型
    tf.add_to_collection("y_predict", y_predict)

    # 定义保存模型
    saver = tf.train.Saver()

    # 通过绘画运行程序
    with tf.Session() as sess:
        # 存在变量时需要初始化
        sess.run(tf.global_variables_initializer())

        # 加载上次训练的模型结果
        if os.path.exists("model/model/checkpoint"):
            saver.restore(sess, "model/model/model")

        # 循环训练
        for i in range(100):
            # 读取数据（这里自己生成数据）
            x_train = sess.run(tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x"))

            sess.run(train_op, feed_dict={x: x_train})

            # 保存模型
            if (i + 1) % 10 == 0:
                print("第%d次训练保存，权重：%f, 偏值：%f" % (((i + 1) / 10), w.eval(), b.eval()))
                saver.save(sess, "model/model/model")

def model_load():
    with tf.Session() as sess:
        # 1、加载模型
        saver = tf.train.import_meta_graph("model/model/model.meta")
        saver.restore(sess, tf.train.latest_checkpoint("model/model"))
        graph = tf.get_default_graph()

        # 2、获取占位符
        x = graph.get_tensor_by_name("data/x:0")

        # 3、获取权重和偏置
        y_predict = graph.get_collection("y_predict")[0]

        # 4、读取测试数据
        x_test = sess.run(tf.random_normal([10, 1], mean=1.75, stddev=0.5, name="x"))
        # 5、预测
        for i in range(len(x_test)):
            predict = sess.run(y_predict, feed_dict={x: [x_test[i]]})
            print("第%d个数据，原值：%f, 预测值：%f" % ((i + 1), x_test[i], predict))

if __name__ == '__main__':
    # model_save()
    model_load()