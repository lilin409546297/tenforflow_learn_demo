# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
    线性回归步骤：
        1、准备特征数据和目标值
        2、建立模型 y = wx + b,主要求解w,b的值
        3、计算损失值：误差loss均方误差（y1-y1'）^2 + ... + (yn - yn')^2 / n 其中：yn为特征值矩阵，yn'为平均值矩阵
        4、梯度下降，优化损失过程，需要指定学习率
    矩阵运算：
        乘法：tf.matmul(x, y)
        平方：tf.square(error)
        均值：tf.reduce_mean(error)
    梯度下降：
        tf.train.GradientDescentOptimizer(learning_rate)
        method: minimize(loss)
        return: 梯度下降op
        学习率：
            如果学习率过大会出现梯度消失/梯度爆炸导致NaN
            优化：
                1、重新设计网络
                2、调整学习率
                3、使用梯度截断
                4、使用激活函数
    变量作用域：
        tf.variable_scope(<scope_name>)
        目的：将每一个步骤的变量集中化，达到更加清晰的目的
        如果作用域内名字相同，会增加_1的方式来呈现
        作用：让模型代码更加清晰，作用分明
    收集变量：
        tf.summary.scalar(<name>)
        增加可观察的数值
        1、收集变量
        2、合并写入事件文件
    自定义命令行参数(弃用)：
        声明：
            tf.flags.DEFINE_integer(<name>, <default_value>, <desc>)
            tf.flags.DEFINE_string(<name>, <default_value>, <desc>)
            ...
        获取参数：
            定义：
                FLAGS = tf.flags.FLAGS
            获取：
                FLAGS.<name>
        运行命令：
            python <py_name> --<name>=<value>
"""
import os
import tensorflow as tf

# tf.flags.DEFINE_integer("max_step", 2000, "最大训练次数")
#
# FLAGS = tf.flags.FLAGS

def tensorflow_linear_regression():
    with tf.variable_scope("data"):
        # 1、准备特征值和目标值
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x")
        # 矩阵相乘必须是二维(为了模拟效果而设定固定值来训练)
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("model"):
        # 2、建立回归模型，随机给权重值和偏置的值，让他去计算损失，然后在当前状态下优化
        # 模型 y = wx + b, w的个数根据特征数据而定，b随机
        # 其中Variable的参数trainable可以指定变量是否跟着梯度下降一起优化(默认True)
        w = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name="w", trainable=True)
        b = tf.Variable(0.0, name="b")
        # 预测值
        y_predict = tf.matmul(x, w) + b

    with tf.variable_scope("loss"):
        # 3、建立损失函数，均方误差
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    with tf.variable_scope("optimizer"):
        # 4、梯度下降优化损失
        # 学习率的控制非常重要，如果过大会出现梯度消失/梯度爆炸导致NaN
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    # 1）收集变量
    tf.summary.scalar("losses", loss)
    tf.summary.histogram("ws", w)

    # 2）合并变量
    merged = tf.summary.merge_all()

    tf.add_to_collection("y_predict", y_predict)

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 定义保存模型
    saver = tf.train.Saver()

    # 通过绘画运行程序
    with tf.Session() as sess:
        sess.run(init_op)
        print("运行前，权重值：%f, 偏置：%f" % (w.eval(), b.eval()))
        file_write = tf.summary.FileWriter("tmp/summary/regression", sess.graph)

        # 加载上次训练的模型结果
        if os.path.exists("model/checkpoint/checkpoint"):
            saver.restore(sess, "model/checkpoint/model")

        # 循环训练
        for i in range(2000):
        # python tensorflow_linear_regression_demo.py --max_step=1000
        # for i in range(FLAGS.max_step):
            sess.run(train_op)
            print("运行 %d 后，权重值：%f, 偏置：%f" % (i + 1, w.eval(), b.eval()))

            # 运行合并后的数据
            summary = sess.run(merged)
            file_write.add_summary(summary, i)

            # 保存模型
            if (i + 1) % 100 == 0:
                saver.save(sess, "model/checkpoint/model")


if __name__ == '__main__':
    tensorflow_linear_regression()