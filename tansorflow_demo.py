# !/usr/bin/python
# -*- coding: UTF-8 -*-


"""
    TensorFlow：
        神经网络（深度）
            图像：卷积神经网络
            自然语言处理：循环神经网络
        特点：
            引入各种计算设备（CPU/GPU/TPU）,以及能够很好的运行在移动端。
            合理的C++使用界面，易用的Python使用界面来构造个执行你的graphs，可以直接写Python/C++程序
            采用数据流图（data flow graphs）,用于数值计算的开源库，能够灵活的组装图，执行图
            谷歌支持，希望成为通用语言
        前后端系统：
            前端系统：定义程序图的机构
            后端系统：运算图结构
        会话：
            1、运算图的结构
            2、分配资源计算
            3、掌握资源（变量的资源，队列，线程）
    数据流图：
        tensor：张量（numpy中的数组，ndarray类型然后封装为tensor），简而言之，就是数组
        operation（op）：专门运算的操作节点，所有操作都是一个op
        图：你的这个程序架构
        会话：运算程序的图
    tensor（张量）:
        一个类型化的N维数组
        三部分：
            名字、形状、数据类型
        阶：和数组的维度类似
        属性：
            graph：张量的默认图
            op：张量的操作名
            name：张量的字符串描述
            shape：张量的形状
        动态形状和静态形状：
            动态形状：（动态形状,创建一个新的张量并且数据量大小不变）
                一种描述原始张量在执行过程中的一种形状（动态变化）
                tf.reshape(和numpy类似)，创建一个具有不同形态的新张量
            静态形状：（静态形状，一旦张量固定，不能再次设置静态形状，不能夸维度修改）
                创建一个张量，初始的形状
                tf.get_shape():获取静态形状
                tf.set_shape():更新对象的静态形状。通常用于不能推断的情况下
        张量操作：
            固定值张量：
                tf.zeros(shape, dtype, name)
                tf.ones()
                tf.constant()
            随机张量：（正太分布）
                tf.random_normal(shape, mean, stddev, dtype, seed, name)
                mean: 平均值
                stddev: 标准差
            类型变换：
                tf.cast(x, dtype, name)
            形状变换：
                tf.reshape()
                tf.get_shape()
                tf.set_shape()
            切片与扩展：
                tf.concat(values, axis, name)
        google提供的数据运算：
            地址：https://tensorflow.google.cn/api_docs/python/tf/math
    变量：
        也是一种op，是一种特殊的张量，能够进行储存持久化，它的值就是张量，默认被训练
        tf.Variable(initial_value, name, trainable)
        注：
            1、变量op能够持久化保存，普通张量不行
            2、当定义一个变量op的时候，一定要在会话中取运行初始化
            3、name参数：在tensortboard使用的时候展示名字，可以让相同op名字进行区分
    可视化tensorboard：
        通过读取TensorFlow事件文件来运行
        tf.summary.FileWriter(path, graph)
        读取（cmd中执行）：tensorboard --logdir "path"
"""
import tensorflow as tf

a = tf.constant(5.0)
b = tf.constant(4.0)

sum = tf.add(a, b)

# 默认的这张图，相当于一块内存
graph = tf.get_default_graph()
print(graph)

# 只能运行一个图
with tf.Session() as sess:
    print(sess.run(sum))

# 图的创建
# 创建一张图包含了一组op和tensor，上下文环境
# op：只要使用tensorflow的api定义的函数都是op
# tensor：指数数据
g = tf.Graph()
with g.as_default():
    c = tf.constant(12.0)
    # 有重载机制（默认给运算符重载成op类型）
    d = c + 1.0
print(g)

# 可以在会话中指定运行
# config:
#   log_device_placement: 查看运行设备信息
with tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
    # eval:只有在会话上下文才可以使用
    print(d.eval())

    # 占位符
    plt = tf.placeholder(tf.float32, [None, 3])
    print(plt)
    # 静态形状，一旦张量固定，不能再次设置静态形状，不能夸维度修改
    plt.set_shape([2, 3])
    print(plt)
    # 动态形状,创建一个新的张量并且数据量大小不变
    plt_reshape = tf.reshape(plt, [3,2])
    print(plt_reshape)

    print(sess.run(plt, feed_dict={plt: [[1,2,3], [4,5,6]]}))

    print("*" * 20)
    print(d.graph)
    print("-" * 20)
    print(d.op)
    print("-" * 20)
    print(d.name)
    print("-" * 20)
    # 形状表示维度大小，如果是（）表示0维，?代表不确定
    print(d.shape)
    print("*" * 20)


# 变量
e = tf.constant(1.0, name="e")
f = tf.constant(2.0, name="f")
g = tf.add(e, f, name="g")
var = tf.Variable(tf.random_normal([2,3], mean=0, stddev=1), name="var")
print(e)
print(var)

# 初始化所有变量的op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 必须初始化op，才可运行
    sess.run(init_op)
    # tensorboard 写入
    tf.summary.FileWriter("tmp/summary/test", graph=sess.graph)
    print(sess.run([g, var]))
