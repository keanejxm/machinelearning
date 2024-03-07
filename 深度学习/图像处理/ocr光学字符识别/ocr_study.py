#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  ocr_study.py
:time  2024/3/6 14:30
:desc  
"""
import os
import math
import random
import shutil
import time

import matplotlib.pyplot as plt
import six
import paddle
import codecs
import logging
import numpy as np
from PIL import Image, ImageEnhance
import paddle.fluid as fluid
from common_utils import DATA_PATH


class BaseClass:
    @staticmethod
    def init_data():
        return {
            "input_size": [1, 48, 512],  # 输入数据维度
            "data_dir": "word-recognition",  # 数据集路径
            "train_dir": "trainImageSet",  # 训练数据目录
            "eval_dir": "evalImageSet",  # 评估数据目录
            "train_list": "train.txt",  # 训练集文件
            "eval_list": "eval.txt",  # 评估集文件
            "label_list": "label_list.txt",  # 标签文件
            "class_dim": -1,
            "label_dict": {},  # 标签字典
            "image_count": -1,
            "continue_train": False,
            "pretrained": True,  # 预训练
            "pretrained_model_dir": "./pretrained-model",  # 预训练模型目录
            "save_model_dir": "./crnn-model",  # 模型保存目录
            "num_epochs": 400,  # 训练轮次
            "train_batch_size": 256,  # 训练批次大小
            "use_gpu": True,  # 是否使用gpu
            "ignore_thresh": 0.7,  # 阈值
            "mean_color": 127.5,  #
            "mode": "train",  # 模式
            "multi_data_reader_count": 4,  # reader数量
            "apply_distort": True,  # 是否进行扭曲
            "image_distort_strategy": {  # 扭曲策略
                "expand_prob": 0.5,  # 放大比率
                "expand_max_ratio": 2,  # 最大放大比率
                "hue_prob": 0.5,  # 色调
                "hue_delta": 18,
                "contrast_prob": 0.5,  # 对比度
                "contrast_delta": 0.5,
                "saturation_prob": 0.5,  # 饱和度
                "saturation_delta": 0.5,
                "brightness_prob": 0.5,  # 亮度
                "brightness_delta": 0.125
            },
            "rsm_strategy": {  # 梯度下降配置
                "learning_rate": 0.0005,
                "lr_epochs": [70, 120, 170, 220, 270, 320],  # 学习率衰减分段（6个数字分为7段）
                "lr_decay": [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],  # 每段采用的学习率，对应lr_epochs参数7段
            },
            "early_stop": {  # 控制训练停止条件
                "sample_frequency": 50,
                "successive_limit": 5,
                "min_instance_error": 0.1
            }
        }

    @staticmethod
    def init_log_config():
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_path = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_name = os.path.join(log_path, 'train.log')
        sh = logging.StreamHandler()
        fh = logging.FileHandler(log_name, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.addHandler(fh)


class CRNNModule(BaseClass):
    def __init__(self, num_classes, label_dict):
        self.name = "CRNN模型"
        self.outputs = None  # 输出
        self.num_classes = num_classes  # 类别数量
        self.label_dict = label_dict  # 标签字典

    def conv(self, input_data, group, out_ch, act="relu", param=None, bias=None, param_0=None, is_test=False,
             pooling=True, use_cudnn=False):
        tmp = input_data
        for i in range(group):
            # 卷积层
            tmp = fluid.layers.conv2d(
                input=tmp,
                num_filters=out_ch[i],
                filter_size=3,
                padding=1,
                param_attr=param if param_0 is None else param_0,
                act=None,
                use_cudnn=use_cudnn
            )
            # batchNormal
            tmp = fluid.layers.batch_norm(
                input=tmp,
                act=act,
                param_attr=param,
                bias_attr=bias,
                is_test=is_test
            )
        if pooling:
            tmp = fluid.layers.pool2d(
                input=tmp,
                pool_size=2,
                pool_type="max",
                pool_stride=2,
                use_cudnn=use_cudnn,
                ceil_mode=True,
            )
        return tmp

    def conv_layers(self, input_data, regularizer=None, gradient_clip=None, is_test=False, use_cudnn=False):
        """
        卷积层
        :return:
        """
        b = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0)
        )
        w0 = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02)
        )
        w1 = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.01)
        )
        tmp = input_data
        # 第一组池化
        tmp = self.conv(
            tmp,
            2,
            [16, 16],
            param=w1,
            bias=b,
            param_0=w0,
            is_test=is_test,
            use_cudnn=use_cudnn
        )
        # 第二组池化
        tmp = self.conv(
            tmp,
            2,
            [32, 32],
            param=w1,
            bias=b,
            param_0=w0,
            is_test=is_test,
            use_cudnn=use_cudnn
        )
        # 第三组池化
        tmp = self.conv(
            tmp,
            2,
            [64, 64],
            param=w1,
            bias=b,
            param_0=w0,
            is_test=is_test,
            use_cudnn=use_cudnn
        )
        # 第四组池化
        tmp = self.conv(
            tmp,
            2,
            [128, 128],
            param=w1,
            bias=b,
            param_0=w0,
            is_test=is_test,
            use_cudnn=use_cudnn
        )
        return tmp

    def net(self, images, rnn_hidden_size=200, regularizer=None, gradient_clip=None, is_test=False, use_cudnn=True):
        """
        CRNN由卷积层、循环层、转录层构成
        :param images:
        :param rnn_hidden_size:
        :param regularizer:
        :param gradient_clip:
        :param is_test:
        :param use_cudnn:
        :return:
        """
        # 卷积池化
        conv_features = self.conv_layers(
            images,
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            is_test=is_test,
            use_cudnn=use_cudnn
        )
        # 转序列
        sliced_features = fluid.layers.im2sequence(
            input=conv_features,
            stride=[1, 1],
            filter_size=[conv_features.shape[2], 1]
        )
        para_attr = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02)
        )
        bias_attr = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02)
        )
        bias_attr_nobias = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02)
        )
        # 全连接层
        fc_1 = fluid.layers.fc(
            input=sliced_features,
            size=rnn_hidden_size * 3,
            param_attr=para_attr,
            bias_attr=bias_attr_nobias
        )
        fc_2 = fluid.layers.fc(
            input=sliced_features,
            size=rnn_hidden_size * 3,
            param_attr=para_attr,
            bias_attr=bias_attr_nobias
        )
        # gru(门控循环单元)，LSTM变种
        # 对检测到的字符连接成字符串序列
        gru_forward = fluid.layers.dynamic_gru(
            input=fc_1,
            size=rnn_hidden_size,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu'
        )
        gru_backward = fluid.layers.dynamic_gru(
            input=fc_2,
            size=rnn_hidden_size,
            is_reverse=True,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu'
        )
        w_attr = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.02)
        )
        b_attr = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0)
        )
        fc_out = fluid.layers.fc(
            input=[gru_forward, gru_backward],
            size=self.num_classes + 1,
            param_attr=w_attr,
            bias_attr=b_attr
        )
        self.outputs = fc_out
        return fc_out

    def get_infer(self):
        return fluid.layers.ctc_greedy_decoder(input=self.outputs, blank=self.num_classes)


class ImageDeal:
    # 重设图像大小
    @staticmethod
    def resize_img(img, input_size):
        target_size = input_size
        percent_h = float(target_size[1]) / img.size[1]
        percent_w = float(target_size[2]) / img.size[0]

        percent = min(percent_h, percent_w)

        resized_width = int(round(img.size[0] * percent))
        resized_height = int(round(img.size[1] * percent))

        w_off = (target_size[2] - resized_width) / 2
        h_off = (target_size[1] - resized_height) / 2

        img = img.resize((resized_width, resized_height), Image.ANTIALIAS)
        array = np.ndarray((target_size[1], target_size[2], 3), np.uint8)

        array[:, :, 0] = 127
        array[:, :, 1] = 127
        array[:, :, 2] = 127
        ret = Image.fromarray(array)
        ret.paste(img, (np.random.randint(0, w_off + 1), int(h_off)))
        return ret

    # 调节亮度
    @staticmethod
    def random_brightness(img, bright_prob, bright_delta):
        """"""
        prob = np.random.uniform(0, 1)
        if prob < bright_prob:
            delta = np.random.uniform(-bright_delta, bright_delta) + 1
            img = ImageEnhance.Brightness(img).enhance(delta)
        return img

    # 调节对比度
    @staticmethod
    def random_contrast(img, contrast_prob, contrast_delta):
        prob = np.random.uniform(0, 1)
        if prob < contrast_prob:
            delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
            img = ImageEnhance.Contrast(img).enhance(delta)
        return img

    # 饱和度
    @staticmethod
    def random_saturation(img, saturation_prob, saturation_delta):
        prob = np.random.uniform(0, 1)
        if prob < saturation_prob:
            delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
            img = ImageEnhance.Color(img).enhance(delta)
        return img

    # 色调
    @staticmethod
    def random_hue(img, hue_prob, hue_delta):
        prob = np.random.uniform(0, 1)
        if prob < hue_prob:
            delta = np.random.uniform(-hue_delta, hue_delta)
            img_hsv = np.array(img.convert('HSV'))
            img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
            img = Image.fromarray(img_hsv, mode="HSV").convert("RGB")
        return img

    # 图像增强随机旋转角度
    @staticmethod
    def rotate_image(img):
        """
        图像增强，增加随机旋转角度
        :param img:
        :return:
        """
        prob = np.random.uniform(0, 1)
        if prob > 0.5:
            angle = np.random.randint(-8, 8)
            img = img.rotate(angle)
        return img

    @staticmethod
    def random_expand(img, expand_prob, max_ratio, mean_color, keep_ratio=True, ):
        """图像填充"""
        if np.random.uniform(0, 1) < expand_prob:
            return img
        w, h = img.size
        c = 3
        ratio_x = random.uniform(1, max_ratio)
        if keep_ratio:
            ratio_y = ratio_x
        else:
            ratio_y = random.uniform(1, max_ratio)
        oh = int(h * ratio_y)
        ow = int(w * ratio_x)
        off_x = random.randint(0, ow - w)
        off_y = random.randint(0, oh - h)
        out_img = np.zeros((oh, ow, c), np.uint8)
        for i in range(c):
            out_img[:, :, i] = mean_color
        out_img[off_y:off_y + h, off_x:off_x + w, :] = img
        return Image.fromarray(out_img)


class OcrOperate(BaseClass):
    def __init__(self):
        self.base_data = self.init_data()
        self.logger = self.init_log_config()
        self.orc_data_dir = os.path.join(DATA_PATH, "ocr_data")

    # *******************************reader代码开始******************************************
    def pretreatment_data(self):
        """
        对数据处理，分测试集和训练集，分别保存
        :return:
        """
        train_ration = 9 / 10
        data_file_dir = os.path.join(self.orc_data_dir, self.base_data["data_dir"])
        images_file_dir = os.path.join(data_file_dir, "imageSet")

        # 创建训练集路径
        train_image_dir = os.path.join(data_file_dir, "trainImageSet")
        if not os.path.exists(train_image_dir):
            os.mkdir(train_image_dir)
        eval_image_dir = os.path.join(data_file_dir, "evalImageSet")
        if not os.path.exists(eval_image_dir):
            os.mkdir(eval_image_dir)

        train_file = codecs.open(os.path.join(data_file_dir, "train.txt"), 'w')
        eval_file = codecs.open(os.path.join(data_file_dir, "eval.txt"), 'w')
        label_list = os.path.join(data_file_dir, "image_label.txt")  # 标签文件

        train_count = 0
        eval_count = 0
        class_set = set()
        # 读取标签文件
        label_file_path = os.path.join(data_file_dir, "image_label.txt")
        with open(label_file_path, "r") as f:
            for line in f:
                image_name, label = line.strip().split()
                flag = [True if i in label else False for i in
                        ["/", "\'", ".", "!", "-", '$', "&", "@", "?", "%", "(", ")", "~"]]
                if True in flag:
                    continue
                for e in label:
                    class_set.add(e)
                if random.uniform(0, 1) <= train_ration:
                    shutil.copyfile(os.path.join(images_file_dir, image_name),
                                    os.path.join(train_image_dir, image_name))
                    train_file.write("{0}\t{1}\n".format(os.path.join(train_image_dir, image_name), label))
                    train_count += 1
                else:
                    shutil.copyfile(os.path.join(images_file_dir, image_name), os.path.join(eval_image_dir, image_name))
                    eval_file.write("{0}\t{1}\n".format(eval_image_dir, label))
                    eval_count += 1
        self.logger.info("train image count: {0} eval image count: {1}".format(train_count, eval_count))
        class_list = list(class_set)
        class_list.sort()
        print("class num: {0}".format(len(class_list)))
        print(class_list)
        with codecs.open(os.path.join(data_file_dir, "label_list.txt"), "w") as label_list:
            label_id = 0
            for c in class_list:
                label_list.write("{0}\t{1}\n".format(c, label_id))
                label_id += 1

    @staticmethod
    def distort_image(img, distort_strategy):
        """"""
        img_deal = ImageDeal()
        prob = np.random.uniform(0, 1)
        if prob > 0.5:
            img = img_deal.random_brightness(img, distort_strategy["brightness_prob"],
                                             distort_strategy["brightness_delta"])
            img = img_deal.random_contrast(img, distort_strategy["contrast_prob"],
                                           distort_strategy["contrast_delta"])
            img = img_deal.random_saturation(img, distort_strategy["saturation_prob"],
                                             distort_strategy["saturation_delta"])
            img = img_deal.random_hue(img, distort_strategy["hue_prob"],
                                      distort_strategy["hue_delta"])
        else:
            img = img_deal.random_brightness(img, distort_strategy["brightness_prob"],
                                             distort_strategy["brightness_delta"])
            img = img_deal.random_saturation(img, distort_strategy["saturation_prob"],
                                             distort_strategy["saturation_delta"])
            img = img_deal.random_hue(img, distort_strategy["hue_prob"],
                                      distort_strategy["hue_delta"])
            img = img_deal.random_contrast(img, distort_strategy["contrast_prob"],
                                           distort_strategy["contrast_delta"])
        return img

    def preprocess(self, img):
        """对图片进行预处理"""
        distort_strategy = self.base_data["strategy"]
        if self.base_data['apply_distort']:
            img = self.distort_image(img, distort_strategy)
        # 填充

        img = ImageDeal.random_expand(
            img, distort_strategy["expand_prob"],
            self.base_data["expand_max_ratio"],
            self.base_data["mean_color"]
        )
        #
        img = ImageDeal.rotate_image(img)
        return img

    def custom_reader(self, file_list, data_dir, input_size, mode):
        """"""

        def reader():
            np.random.shuffle(file_list)
            for line in file_list:
                parts = line.split()
                image_path = parts[0]
                img = Image.open(image_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                label = [int(self.base_data['label_dict'][c]) for c in parts[-1]]
                if len(label) == 0:
                    continue
                if mode == "train":
                    img = self.preprocess(img)
                img = ImageDeal.resize_img(img, input_size)
                img = img.convert("L")
                img = np.array(img).astype("float32") - self.base_data["mean_color"]
                img = img[np.newaxis, ...]
                yield img, label

        return reader

    # *******************************reader代码结束******************************************

    def load_pretrained_params(self, exe, program):
        # 如果设置了增量训练，加载之前训练的模型
        if self.base_data["continue_train"] and os.path.exists(self.base_data["save_model_dir"]):
            self.logger.info("load param from retrain model")
            fluid.io.load_persistables(
                executor=exe,
                dirname=self.base_data["save_model_dir"],
                main_program=program
            )
        # 如果设置了预训练，则加载预训练模型
        elif self.base_data["pretrained"] and os.path.exists(self.base_data["pretrained_model_dir"]):
            self.logger.info("load param from pretrained model")

            def if_exist(var):
                return os.path.exists(os.path.join(self.base_data["pretrained_model_dir"], var.name))

            fluid.io.load_vars(
                executor=exe,
                dirname=self.base_data["pretrained_model_dir"],
                main_program=program,
                predicate=if_exist
            )

    def train(self):
        """"""
        train_program = fluid.Program()
        start_program = fluid.Program()
        eval_program = fluid.Program()
        place = fluid.CUDAPlace(0) if self.base_data['use_gpu'] else fluid.CPUPlace()

        # ********************************定义异步读取器、预测、构建损失函数及优化器********************************
        with fluid.program_guard(main_program=train_program, startup_program=start_program):
            img = fluid.layers.data(name="img", shape=self.base_data["input_size"], dtype='float32')
            gt_label = fluid.layers.data(name="gt_label", shape=[1], dtype="int32", lod_level=1)

            # 创建reader
            data_reader = fluid.layers.create_py_reader_by_data(
                capacity=self.base_data["train_batch_size"],
                feed_list=[img, gt_label],
                name="train"
            )
            # 数据集路径
            data_set_dir = os.path.join(self.orc_data_dir, "word-recognition")
            # 训练数据集路径(train.txt:用来存放数据集的路径，TODO:使用的时候需要修改)
            train_data_dir = os.path.join(self.orc_data_dir, "train.txt")
            num_readers = self.base_data["multi_data_reader_count"]
            # 创建多进程reader
            readers = []
            images = [line.strip() for line in open(train_data_dir)]
            n = int(math.ceil(len(images) // num_readers))
            image_lists = [images[i:i + n] for i in range(0, len(images), n)]
            train_path = os.path.join(data_set_dir, self.base_data["train_dir"])
            for l in image_lists:
                reader = paddle.batch(
                    self.custom_reader(l, train_path, input_size=self.base_data["input_size"], mode="train"),
                    batch_size=self.base_data["train_batch_size"])
                readers.append(paddle.reader.shuffle(reader, self.base_data["train_batch_size"]))
            multi_reader = paddle.reader.multiprocess_reader(readers, False)
            data_reader.decorate_paddle_reader(multi_reader)
            with fluid.unique_name.guard():  # 更换namespace
                img, gt_label = fluid.layers.read_file(data_reader)
                model = CRNNModule(self.base_data["class_dim"], self.base_data["label_dict"])
                fc_out = model.net(img)
                cost = fluid.layers.warpctc(
                    input=fc_out,
                    label=gt_label,
                    blank=self.base_data["class_dim"],
                    norm_by_times=True
                )
                loss = fluid.layers.reduce_sum(cost)
                # 设置优化器
                batch_size = self.base_data["train_batch_size"]
                iters = self.base_data["image_count"] // batch_size
                learning_strategy = self.base_data["rsm_strategy"]
                lr = learning_strategy["learning_rate"]
                boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
                values = [i * lr for i in learning_strategy["lr_decay"]]
                # 均方根传播（RMSProp）法
                optimizer = fluid.optimizer.RMSProp(
                    learning_rate=fluid.layers.piecewise_decay(boundaries, values),
                    regularization=fluid.regularizer.L2Decay(0.00005)
                )
                optimizer.minimize(loss)

                # 执行CTC去重
                decoded_out = fluid.layers.ctc_greedy_decoder(
                    input=fc_out,
                    blank=self.base_data["class_dim"]
                )
                casted_label = fluid.layers.cast(x=gt_label, dtype="int64")

                distances, seq_num = fluid.layers.edit_distance(decoded_out, casted_label)
                # return data_reader, loss, distances, seq_num, decoded_out
                # train_reader,        loss, distances, seq_num, decoded_out

        # ********************************评估********************************
        with fluid.program_guard(main_program=eval_program, startup_program=start_program):
            img = fluid.layers.data(name="img", shape=self.base_data["input_size"], dtype="float32")
            gt_label = fluid.layers.data(name="gt_label", shape=[1], dtype="int32", lod_level=1)
            feeder = fluid.DataFeeder(feed_list=[img, gt_label], place=place, program=eval_program)

            # 创建reader
            eval_file_path = os.path.join(data_set_dir, "eval.txt")  # 评估集文件
            images = [line.strip() for line in open(eval_file_path)]
            eval_path = os.path.join(data_set_dir, self.base_data["eval_dir"])
            eval_reader = paddle.batch(self.custom_reader(
                images,
                eval_path,
                input_size=self.base_data["input_size"],
                mode="eval"
            ), batch_size=self.base_data["train_batch_size"])
            with fluid.unique_name.guard():
                eval_model = CRNNModule(self.base_data["class_dim"], self.base_data["label_dict"])
                outputs = eval_model.net(img)
                # return feeder,       reader,    outputs, gt_label
                #       eval_feeder, eval_reader, output, gt_label
        eval_program = eval_program.clone(for_test=True)

        # 训练
        exe = fluid.Executor(place)
        exe.run(start_program)
        train_fetch_list = [loss.name, distances.name, seq_num.name, decoded_out.name]
        eval_fetch_list = [outputs.name]
        # 加载增量训练模型或者预训练模型
        self.load_pretrained_params(exe, train_program)
        stop_strategy = self.base_data["early_stop"]
        successive_limit = stop_strategy["successive_limit"]
        sample_freq = stop_strategy["sample_frequency"]
        min_instance_error = stop_strategy["min_instance_error"]

        stop_train = False
        successive_count = 0
        total_batch_count = 0
        distance_evaluator = fluid.metrics.EditDistance("edit-distance")

        # 执行训练
        for pass_id in range(self.base_data["num_epochs"]):
            self.logger.info("current pass %d,start read image", pass_id)
            batch_id = 0
            data_reader.start()  # 启动reader线程
            distance_evaluator.reset()
            try:
                while True:
                    t1 = time.time()
                    loss, distances, seq_num, decoded_out = exe.run(
                        train_program,
                        fetch_list=train_fetch_list,
                        return_numpy=False
                    )
                    distances = np.array(distances)
                    seq_num = np.array(seq_num)
                    distance_evaluator.update(distances, seq_num)
                    period = time.time() - t1
                    loss = np.mean(np.array(loss))
                    batch_id += 1
                    total_batch_count += 1
                    if batch_id % 10 == 0:
                        distance, instance_error = distance_evaluator.eval()
                        self.logger.info("Pass {0}, trainbatch {1}, loss {2} distance {3} instance error {4} time {5}"
                                         .format(pass_id, batch_id, loss, distance, instance_error,
                                                 "%2.2f sec" % period))
                    # 采用简单的定时采样停止办法，可以调整为更为精细的保存策略
                    if total_batch_count % 100 == 0:
                        self.logger.info("temp save {0} batch train result".format(total_batch_count))
                        fluid.io.save_persistables(
                            dirname=self.base_data["save_model_dir"],
                            main_program=train_program,
                            executor=exe
                        )
                    if total_batch_count % sample_freq == 0:
                        if instance_error <= min_instance_error:
                            successive_count += 1
                            self.logger.info(
                                "instance error {0} successive count {1}".format(instance_error, successive_count))
                            if successive_count >= successive_limit:
                                stop_train = True
                                break
                        else:
                            successive_count = 0
            except fluid.core.EOFException:
                data_reader.reset()
            distance, instance_error = distance_evaluator.eval()
            self.logger.info("Pass {0} distance {1} instance error {2}".format(pass_id, distance, instance_error))
            if stop_train:
                self.logger.info("early stop")
                break
        self.logger.info("training till last, end training")
        fluid.io.save_persistables(dirname=self.base_data["save_model_dir"], main_program=train_program, executor=exe)

    def freeze_model(self):
        """保存推理模型"""
        exe = fluid.Executor(fluid.CPUPlace())
        image = fluid.layers.data(name='image', shape=self.base_data["input_size"], dtype="float32")
        with codecs.open(os.path.join(self.orc_data_dir, "label_list.txt")) as label_list:
            class_dim = len(label_list.readlines())
        model = CRNNModule(class_dim, {})
        pred = model.net(image)
        out = model.get_infer()
        freeze_program = fluid.default_main_program()
        fluid.io.load_persistables(
            executor=exe,
            dirname=self.base_data["freeze_model_dir"],
            main_program=freeze_program
        )
        freeze_program = freeze_program.clone(for_test=True)
        fluid.io.save_inference_model(
            dirname="./freeze-model",
            feeded_var_names=["image"],
            target_vars=out,
            executor=exe,
            main_program=freeze_program
        )

    # 预测
    def infer_model(self):
        """"""
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        infer_model_dir = "./freeze-model"
        # 加载模型
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
            dirname=infer_model_dir,
            executor=exe
        )
        label_dict = {}
        # 初始化训练参数主要是初始化图片数量类别数
        label_list_path = os.path.join(self.orc_data_dir, self.base_data["label_list"])
        with codecs.open(label_list_path, encoding="utf-8") as flist:
            lines = [line.strip() for line in flist]
            for line in lines:
                parts = line.split()
                label_dict[int(parts[1])] = parts[0]
        # 测试图片
        img_path = "1.png"
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = ImageDeal.resize_img(img, self.base_data["input_size"])
        img = img.convert('L')
        img = np.array(img).astype('float32') - self.base_data["mean_rgb"]
        img = img[..., np.newaxis]
        img = img.transpose((2, 0, 1))
        img = img[np.newaxis, ...]
        label = exe.run(
            program=inference_program,
            feed={feed_target_names[0]: img},
            fetch_list=fetch_targets,
            return_numpy=False
        )
        label = np.array(label[0])
        ret = ''
        if label[0] != -1:
            ret = ret.join([label_dict[int(c[0])] for c in label])
        self.logger.info("infer result:{0}".format(ret))

        img = Image.open(img_path)
        plt.imshow(img)
        plt.show()
