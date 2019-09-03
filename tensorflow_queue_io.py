# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
    tensorflow:
        多线程是真正的多线程执行。
        队列：
            tf.FIFOQueue(<capacity>, <dtypes>, <name>), 先进先出
            tf.RandomShuffleQueue, 随机出队列
        多线程：
            当数据量很大时，入队操作从硬盘中读取，放入内存。主线程需要等待操作完成，才能训练。
            使用多线程，可以达到边处理，边训练的异步效果。
        队列管理器(弃用)：
            tf.train.QueueRunner(<queue>, <enqueue_ops>)
            enqueue_ops: 添加线程的队列操作列表[]*2为开启2个线程，[]内为操作部分
            method:
                create_threads(<sess>, <coord>, <start>):
                    创建线程来运行给定的入队操作。
                    start: 布尔值，是否启动线程
                    coord: 线程协调器
                    return: 线程实例
        线程协调器：
            协调线程之间终止
    文件io:
        1、csv文件读取一行
        2、二进制文件指定bytes
        3、图片文件一张一张
        流程：
            1、构造一个文件队列
            2、读取文件内容
            3、解码文件内容
            4、批处理
        api:
            1、文件队列构造
                tf.train.string_input_producer(<string_tensor>, <shuffle=True>)
                string_tensor: 含有文件名的一阶张量
                num_epochs: 过几遍数据，默认无数遍
            2、文件阅读器
                tf.TextLineReader、csv文件格式类型
                tf.FixedLengthRecordReader(record_bytes)、读取固定值的二进制文件
                tf.TFRecordReader、读取TfRecords
                共同：
                    read(file_queue): 队列中指定数量
                    return: Tensors 元组（key：文件名， value默认行内容）
            3、文件解码器：
                tf.decode_csv(<records>, <record_defaults=None>, <field_delim=None>, <name=None>)
                将CSV转换为张量，与tf.TextLineReader搭配使用
                records: tensor型字符串，每一个字符串为CSV中的记录
                record_defaults: 参数决定了所有张量的类型，并设置一个值在输入字符串中缺少使用默认值
                tf.decode_raw(<bytes>, <out_type>, <little_endian=None>, <name=None>)
                将字节转换为一个向量表示，字节为一字符串类型的张量，与函数rf.FixedLengthRecordReader搭配使用，二进制读取为utf-8格式
        csv文件读取：
            1、找到文件，构建列表
            2、构造文件队列
            3、构造阅读器，读取队列内容
            4、解码内容
            5、批处理
        管道读端批处理：
            tf.train.batch(<tensors>, <batch_size>, <num_threads=1>, <capacity=32>, <name=None>)
            tensors: 张量列表
            tf.train.shuffle_batch(<tensors>, <batch_size>, <capacity>, <min_dequeue>)
            min_dequeue: 留下队列里的张量个数，能够保持随机打乱
        图片读取：
            每一个样本必须保证特征数量一样
            特征值：像素值
                单通道：灰度值（黑白图片，像素中只有一个值）
                三通道：RGB(每个像素都有3个值)
            三要素：长度宽度、通道值
            图像的基本操作：
                目的：
                    1、增加图片数据的统一性
                    2、所有图片装换成指定大小
                    3、缩小图片数据量，防止增加开销
                操作：
                    缩小图片大小
                api：
                    图片缩放：
                        tf.image.resize_images(<images>, <size>)
                        <images>:4-D形状[batch, height, width, channels]/3-D[height, width, channels]
                        <size>:1-D int32张量：new_height, new_width, 图像的新尺寸
                        return：4-D/3-D格式图片
            图片读取api：
                tf.WholeFileReader:
                    将文件的全部内容作为输入的读取器
                    return:读取器实例
                    read(<file_queue>):输出将一个文件名（key）和该文件的内容值
            图像解码器：
                tf.image.decode_jpeg(<contents>):
                    将JPEG编码的图像解码为unit8张量
                    return：uint8张量，3-D形状[height, width, channels]
                tf.image.decode_png():
                    将PNG编码的图像解码为uint8/uint16的张量
                    return:张量类型，3-D[height, width, channels]
        二进制文件读取：
            api:
                tf.FixedLengthRecordReader(<record_bytes>)
                record_bytes:数据长度
            解码器：
                tf.decode_raw(<bytes>, <out_type>, <little_endian=None>, <name=None>)
                bytes：数据
                out_type：输出类型
        tf.TFRecordReader
            一种内置文件格式，是一种二进制文件，它可以更好的利用内存，更方便的复制和移动
            为了将二进制数据和标签（训练类别标签），数据存储在同一文件中
            分析、存取
            文件格式：*.threcords
            写入文件内容：example协议块
            TF存储：
                TFRecord存储器
                    tf.python_io.TFRecordWriter(<path>)
                    method:
                        write(record)
                        close
                Example协议块：
                    tf.train.Example(<features=None>)
                    features:tf.train.Features(<feature=None>)实例
                        feature:字典数据，key为要保存的数据
                            tf.train.Feature(<**options>)
                                **options:
                                    tf.train.ByteList(<value=[Bytes]>)
                                    tf.train.IntList(<value=[Value]>)
                                    tf.train.FloatList(<value=[Value]>)
                        return:Features实例
                    return:Example协议块
            TF读取：
                tf.parse_example(<serialized>, <features=None>, <name=None>)
                    serialized:标量字符串Tensor,一个序列化的Example
                    features:dict字典数据，键为读取的名字，值为FixedLenFeature
                    return:一个键值对组成的字典，键为读取的名字
                    tf.FixedLenFeature(<shape>, <dtype>)
                        shape:形状
                        dtype:数据类型（float32/int64/string）
"""
import os
import tensorflow as tf

def queue_demo():

    # 1、声明队列
    queue = tf.FIFOQueue(3, dtypes=tf.float32)

    # 2、加入数据
    init_queue = queue.enqueue_many([[0.1, 0.2, 0.3]])

    # 3、取出数据
    data = queue.dequeue()

    # 4、处理数据
    en_queue = queue.enqueue(data + 1)

    with tf.Session() as sess:
        # 初始化操作
        sess.run(init_queue)
        # 循环
        for i in range(10):
            sess.run(en_queue)
        for i in range(queue.size().eval()):
            print(queue.dequeue().eval())


def queue_thread_demo():
    # 1、声明队列
    queue = tf.FIFOQueue(100, dtypes=tf.float32)

    # 2、加入数据
    for i in range(100):
        queue.enqueue((i + 1)/100)

    # 3、操作
    data = queue.dequeue()
    en_queue = queue.enqueue(data + 1)

    # 3、定义队列管理器
    qr = tf.train.QueueRunner(queue, enqueue_ops=[en_queue] * 2)

    with tf.Session() as sess:
        # 开启线程协调器
        coord = tf.train.Coordinator()
        # 开启线程
        threads = qr.create_threads(sess, coord=coord, start=True)
        for i in range(100):
            print(sess.run(queue.dequeue()))
        # 注：没有线程协调器，主线程结束，会结束session，导致异常。
        coord.request_stop()
        coord.join(threads)

def csv_io():
    # 1、找到文件，加入队列
    file_names = os.listdir("data/csv")
    file_list = [os.path.join("data/csv", file_name) for file_name in file_names]
    file_queue = tf.train.string_input_producer(file_list)
    # 2、读取一行数据
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)
    # 3、解码csv
    records = [[-1], [-1]]
    num1, num2 = tf.decode_csv(value, record_defaults=records)
    # 4、批处理
    num1_batch, num2_batch = tf.train.batch([num1, num2], batch_size=9, num_threads=1, capacity=9)

    with tf.Session() as sess:
        # 加入线程协调器
        coord = tf.train.Coordinator()
        # 线程运行
        threads = tf.train.start_queue_runners(sess, coord=coord)
        print(sess.run([num1_batch, num2_batch]))

        # 子线程回收
        coord.request_stop()
        coord.join(threads)

def image_io():
    # 1、读取文件放入队列
    image_names = os.listdir("data/captcha")
    image_files = [os.path.join("data/captcha", image_name) for image_name in image_names]
    image_queue = tf.train.string_input_producer(image_files)

    # 2、读取一张图片数据
    reader = tf.WholeFileReader()
    # value:一整张图片的数据
    key, value = reader.read(image_queue)

    # 3、解码
    image = tf.image.decode_jpeg(value)
    print(image)

    # 4、处理图片的大小
    new_image = tf.image.resize_images(image, [350, 350])
    print(new_image)
    # 注意一定要固定形状,批处理的时候所有数据必须固定
    new_image.set_shape([350, 350, 3])
    print(new_image)

    # 5、批处理
    image_batch = tf.train.batch([new_image], batch_size=2, num_threads=1, capacity=2)

    # 6、运行
    with tf.Session() as sess:
        # 加入线程协调器
        coord = tf.train.Coordinator()
        # 线程运行
        threads = tf.train.start_queue_runners(sess, coord=coord)
        print(sess.run([image_batch]))

        # 子线程回收
        coord.request_stop()
        coord.join(threads)

def cifar_io():
    # 1、读取文件加入队列
    cifar_names = os.listdir("data/cifar")
    cifar_files = [os.path.join("data/cifar", cifar_name) for cifar_name in cifar_names if cifar_name.endswith(".bin") and cifar_name != "test_batch.bin"]
    file_queue = tf.train.string_input_producer(cifar_files)

    # 2、读取二进制文件
    reader = tf.FixedLengthRecordReader(record_bytes=(32 * 32 * 3 + 1))
    key, value = reader.read(file_queue)

    # 3、解码数据(二进制数据)
    # 样本数据集根据具体数据处理，这里的数据为第一个数据为目标值，后面的为图片数据
    target_image = tf.decode_raw(value, tf.uint8)

    # 4、分割数据
    target = tf.slice(target_image, [0], [1])
    image = tf.slice(target_image, [1], [32 * 32 * 3])

    # 5、特征数据形状改变
    new_image = tf.reshape(image, [32, 32, 3])
    print(new_image)

    # 6、批处理
    image_batch, target_batch = tf.train.batch([new_image, target], batch_size=10, capacity=10)
    print(image_batch, target_batch)

    # 7、运行
    with tf.Session() as sess:
        # 线程协调器
        coord = tf.train.Coordinator()
        # 线程运行
        threads = tf.train.start_queue_runners(sess, coord=coord)
        print(sess.run([image_batch, target_batch]))

        # 子线程回收
        coord.request_stop()
        coord.join(threads)

# tfrecords文件读写
def tf_records_io():
    # 1、读取文件加入队列
    cifar_names = os.listdir("data/cifar")
    cifar_files = [os.path.join("data/cifar", cifar_name) for cifar_name in cifar_names if
                   cifar_name.endswith(".bin") and cifar_name != "test_batch.bin"]
    file_queue = tf.train.string_input_producer(cifar_files)

    # 2、读取二进制文件
    reader = tf.FixedLengthRecordReader(record_bytes=(32 * 32 * 3 + 1))
    key, value = reader.read(file_queue)

    # 3、解码数据(二进制数据)
    # 样本数据集根据具体数据处理，这里的数据为第一个数据为目标值，后面的为图片数据
    target_image = tf.decode_raw(value, tf.uint8)

    # 4、分割数据
    target = tf.slice(target_image, [0], [1])
    image = tf.slice(target_image, [1], [32 * 32 * 3])

    # 5、特征数据形状改变
    new_image = tf.reshape(image, [32, 32, 3])
    print(new_image)

    # 6、批处理
    image_batch, target_batch = tf.train.batch([new_image, target], batch_size=10, capacity=10)
    print(image_batch, target_batch)

    # 7、tf文件写入
    with tf.Session() as sess:
        if not os.path.exists("data/tf_records/cifar.tfrecords"):
            # 1)存进tfRecords文件
            print("开始存储")
            with tf.python_io.TFRecordWriter(path="data/tf_records/cifar.tfrecords") as writer:
                # 2)循环次数为批次数
                for i in range(10):
                    # 获取对应值
                    image_data = image_batch[i].eval().tostring()
                    target_data = int(target_batch[i].eval()[0])
                    # 3)产生实例
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                        "target": tf.train.Feature(int64_list=tf.train.Int64List(value=[target_data]))
                    }))
                    # 4)写入数据
                    writer.write(example.SerializeToString())
            print("结束存储")

    # 8、tf文件读取
    # 1)读取tfRecords文件
    tf_queue = tf.train.string_input_producer(["data/tf_records/cifar.tfrecords"])

    # 2)读取数据
    tf_reader = tf.TFRecordReader()
    key, value = tf_reader.read(tf_queue)

    # 3)解析example
    features = tf.parse_single_example(value, features={
        "image": tf.FixedLenFeature([], dtype=tf.string),
        "target": tf.FixedLenFeature([], dtype=tf.int64)
    })
    print(features["image"], features["target"])

    # 4)解码数据
    image = tf.decode_raw(features["image"], tf.uint8)
    image_reshape = tf.reshape(image, [32, 32, 3])
    target = tf.cast(features["target"], tf.int32)
    print(image_reshape, target)
    # 5)批处理
    image_batch, target_batch = tf.train.batch([image_reshape, target], batch_size=10, capacity=10)

    # 9、运行
    with tf.Session() as sess:
        # 线程协调器
        coord = tf.train.Coordinator()
        # 线程运行
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # tf文件读取
        print(sess.run([image_batch, target_batch]))

        # 子线程回收
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # queue_demo()
    # queue_thread_demo()
    # csv_io()
    # image_io()
    # cifar_io()
    tf_records_io()