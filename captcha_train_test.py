# !/usr/bin/python
# -*- coding: UTF-8 -*-
import os

import tensorflow as tf


def captcha_train_test(n):
    # 读取数据
    image_batch, label_batch = tfrecords_read_decode()

    # 1、建立占位符(数据更具图片和目标数据而定)
    with tf.variable_scope("data"):
        x = tf.placeholder(dtype=tf.uint8, shape=[None, 50, 200, 3], name="x")
        label = tf.placeholder(dtype=tf.uint8, shape=[None, 5], name="label")

    # 2、建立模型
    y_predict = model(x)

    # y_predict为[None, 5 * 36] label_batch为[None, 5], 因此需要one-hot[None, 5, 36]
    y_true = tf.one_hot(label, depth=36, axis=2, on_value=1.0, name="one_hot")
    # 需要变形为[None 5 * 36]
    y_true_reshape = tf.reshape(y_true, [-1, 5 * 36], name="y_true_reshape")

    # 4、计算损失值
    with tf.variable_scope("loss"):
        softmax_cross = tf.nn.softmax_cross_entropy_with_logits(labels=y_true_reshape, logits=y_predict)
        loss = tf.reduce_mean(softmax_cross)

    # 5、训练
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=loss)

    # 6、计算准确率
    with tf.variable_scope("accuracy"):
        # 因为这里的真实值是2维结果，所以需要把y_predict，转化为2位数据
        equal_list = tf.equal(tf.argmax(y_true, axis=2), tf.argmax(tf.reshape(y_predict, [-1, 5, 36]), 2))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 收集数据
    tf.add_to_collection("y_predict", y_predict)
    if n == 1:
        # 7、会话

        with tf.Session() as sess:
            # 变量初始化
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver = tf.train.Saver()
            # 如果模型存在，则加载模型
            if os.path.exists("model/captcha/checkpoint"):
                saver.restore(sess, "model/captcha/captcha")

            for i in range(2000):
                # 读取数据
                # 训练，记住这里的数据类型为uint8需要转换为tf.float32
                image_train, label_train = sess.run([image_batch, label_batch])
                sess.run(train_op, feed_dict={x: image_train, label: label_train})
                # 保存模型
                if (i + 1) % 100 == 0:
                    saver.save(sess, "model/captcha/captcha")
                acc = sess.run(accuracy, feed_dict={x: image_train, label: label_train})
                print("第%d次，准确率%f" % ((i + 1), acc))

            coord.request_stop()
            coord.join(threads)
    else:
        # 1、读取指定目录下图片数据
        file_names_test = os.listdir("data/captcha_test")
        file_list_test = [os.path.join("data/captcha_test", file_name) for file_name in file_names_test]
        file_queue_test = tf.train.string_input_producer(file_list_test, shuffle=False)
        # 2、读取和解码数据
        reader = tf.WholeFileReader()
        key, value = reader.read(file_queue_test)
        image = tf.image.decode_png(value)
        image.set_shape([50, 200, 3])
        # 3、批处理
        image_batch_test = tf.train.batch([tf.cast(image, tf.uint8)], batch_size=len(file_names_test), capacity=len(file_names_test))
        # 4、加载模型
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # 1、加载模型
            saver = tf.train.Saver()
            saver.restore(sess, "model/captcha/captcha")

            # 4、预测[None, 5 * 36]
            predict = sess.run(y_predict, feed_dict={x: sess.run(image_batch_test)})
            captcha_list = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            for i in range(len(file_names_test)):
                predict_reshape = tf.reshape(predict, [-1, 5, 36])
                captcha = ""
                for j in range(5):
                    captcha += captcha_list[tf.argmax(predict_reshape[i][j], 0).eval()]
                print("预测值：%s, 真实值：%s" % (captcha, file_names_test[i].split(".")[0]))
            coord.request_stop()
            coord.join(threads)

def model(x):
    # # 第一层卷积
    # with tf.variable_scope("conv_1"):
    #     # 卷积[None, 50, 200, 3] -> [None, 50, 200, 32]
    #     w_1 = gen_weight([5, 5, 3, 32])
    #     b_1 = gen_bias([32])
    #     # 在进行模型计算的时候需要使用tf.float32数据进行计算
    #     x_conv_1 = tf.nn.conv2d(tf.cast(x, tf.float32), filter=w_1, strides=[1, 1, 1, 1], padding="SAME") + b_1
    #     # 激活
    #     x_relu_1 = tf.nn.relu(x_conv_1)
    #     # 池化[None, 50, 200, 32] -> [None, 25, 100, 32]
    #     x_pool_1 = tf.nn.max_pool(x_relu_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # # 第二层卷积
    # with tf.variable_scope("conv_2"):
    #     # 卷积[None, 25, 100, 32] -> [None, 25, 100, 64]
    #     w_2 = gen_weight([5, 5, 32, 64])
    #     b_2 = gen_bias([64])
    #     x_conv_2 = tf.nn.conv2d(x_pool_1, filter=w_2, strides=[1, 1, 1, 1], padding="SAME") + b_2
    #     # 激活
    #     x_relu_2 = tf.nn.relu(x_conv_2)
    #     # 池化[None, 25, 100, 64] -> [None, 13, 50, 64]
    #     x_pool_2 = tf.nn.max_pool(x_relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # # 全连接层
    # with tf.variable_scope("full_connection"):
    #     # 生成权重和偏置
    #     # 为什么是5 * 36主要是我们的字符串个数36个，使用one-hot.每个值为36个值，一个验证码5个值，所以为5 * 36个
    #     w_fc = gen_weight([13 * 50 * 64, 5 * 36])
    #     b_fc = gen_bias([5 * 36])
    #     # 修改数据形状
    #     x_fc = tf.reshape(x_pool_2, shape=[-1, 13 * 50 * 64])
    #     # [None, 5 * 36]
    #     y_predict = tf.matmul(x_fc, w_fc) + b_fc
    with tf.variable_scope("model"):
        # 生成权重和偏置
        # 为什么是5 * 36主要是我们的字符串个数36个，使用one-hot.每个值为36个值，一个验证码5个值，所以为5 * 36个
        w_fc = gen_weight([50 * 200 * 3, 5 * 36])
        b_fc = gen_bias([5 * 36])
        # 修改数据形状
        x_fc = tf.reshape(tf.cast(x, tf.float32), shape=[-1, 50 * 200 * 3])
        # [None, 5 * 36]
        y_predict = tf.matmul(x_fc, w_fc) + b_fc
    return y_predict

# 生成权重值
def gen_weight(shape):
    return tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32))

# 生成偏值
def gen_bias(shape):
    return tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=shape))

# 读取数据
def tfrecords_read_decode():
    # 将文件加入队列
    file_queue = tf.train.string_input_producer(["data/tf_records/captcha.tfrecords"])
    # 读取tfrecords文件
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)
    # value的格式为example
    features = tf.parse_single_example(value, features={
        "image": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.string)
    })
    # 解码
    image_data = tf.decode_raw(features["image"], tf.uint8)
    label_data = tf.decode_raw(features["label"], tf.uint8)
    # 改变形状
    image_reshape = tf.reshape(image_data, [50, 200, 3])
    label_reshape = tf.reshape(label_data, [5])
    # 获取批次数据
    image_batch, label_batch = tf.train.batch([image_reshape, label_reshape], batch_size=200, num_threads=1, capacity=200)
    return image_batch, label_batch


if __name__ == '__main__':
    captcha_train_test(2)