# !/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import random

import tensorflow as tf
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# 生成验证码训练集
def gen_captcha():
    captcha_char_list = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    font = ImageFont.truetype(font="font/msyh.ttf", size=36)
    for n in range(2000):
        image = Image.new('RGB', (200, 50), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        chars = ""
        for i in range(5):
            captcha_char = captcha_char_list[random.randint(0, 25)]
            chars += captcha_char
            draw.text((10 + i * 40, 0), captcha_char, (0, 0, 0), font=font)
        image.save(open("data/captcha/" + chars + ".png", 'wb'), 'png')

# 将图片数据和目标值写到tfrecords文件中
def captcha_data_write():
    # 获取图片名称和所有数据
    file_names, image_batch = get_image_name_batch()

    # 获取目标值
    target = get_target(file_names)

    # 写入文件
    write_tf_records(image_batch, target)

def get_image_name_batch():
    # 1、读取图片目录，生成文件名称列表
    file_names = os.listdir("data/captcha")
    file_list = [os.path.join("data/captcha", file_name) for file_name in file_names]
    print(file_list)
    # 2、放入队列（shuffle=False，一定要设置，不然会乱序）
    file_queue = tf.train.string_input_producer(file_list, shuffle=False)
    # 3、读取图片数据
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    # 4、解码图片数据
    image = tf.image.decode_png(value)
    # 改变形状(根据具体的图片大小来)
    image.set_shape([50, 200, 3])
    # 5、获取图片批次
    image_batch = tf.train.batch([image], batch_size=2000, num_threads=1, capacity=2000)
    return file_names, image_batch

def get_target(file_names):
    # 6、获取目标值
    labels = [file_name.split(".")[0] for file_name in file_names]
    print(labels)
    # 7、将目标值装换成具体数值
    captcha_char = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # 转换成字典,然后反转(0:0, 1:1, ...,z:35)
    num_char = dict(enumerate(list(captcha_char)))
    char_num = dict(zip(num_char.values(), num_char.keys()))
    # 8、构建标签列表
    array = []
    for label in labels:
        nums = []
        for char in label:
            nums.append(char_num[char])
        array.append(nums)
    # [[2, 11, 8, 2, 7] ...]
    print(array)
    # 9、转换为tensor张量
    target = tf.constant(array, dtype=tf.uint8)
    return target

def write_tf_records(image_batch, target):
    # 10、上面主要准备图片数据和目标值，下面主要是写入到*.tfrecords中
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 运算获取图片批次数据
        image_batch = sess.run(image_batch)
        # 转换一下数据为uint8
        image_batch = tf.cast(image_batch, tf.uint8)
        # 写入数据
        with tf.python_io.TFRecordWriter("data/tf_records/captcha.tfrecords") as writer:
            for i in range(2000):
                # 全部使用string保存
                image_string = image_batch[i].eval().tostring()
                label_string = target[i].eval().tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_string]))
                }))
                writer.write(example.SerializeToString())
                print("写入第%d数据" % (i + 1))
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    # gen_captcha()
    captcha_data_write()