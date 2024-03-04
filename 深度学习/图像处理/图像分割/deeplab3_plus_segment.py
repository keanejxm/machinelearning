#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/26 22:01
# @Author  : keane
# @Site    : 
# @File    : deeplab3_fenge.py
# @Software: PyCharm
import os
import cv2
import time
import logging
import contextlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import paddle.fluid as fluid
from common_utils import DATA_PATH


# 导入模型
# from deeplab3_plus_model import Deeplab3PlusModel, BaseClass


class BaseClass:
    @staticmethod
    def init_data():
        return {
            "data_dir": 'data/iccv09Data/',  # 数据所在文件夹
            "batch_size": 10,  # 设置训练时的batch_size
            "use_gpu": True,  # 是否使用gpu进行训练
            "train_model_dir": "train model",  # 训练阶段暂时保存模型的路径，可重新加载再训练
            "infer_model_dir": "infer model",  # 最终模型保存的路径
            "pretrained_model_dir": "pretrained model/deeplabv3plus_gn",  # 预训练模型存在的地方
            "eval_file_path": 'data/eval_list.txt',  # 验证集的路径
            "continue_train": False,  # 是否接着上次的继续训练
            "paddle_flag": False,  # 是否使用paddle上预训练好的模型进行微调，如果要接着上次断点训练，这里需要改为False
            "num_classes": 8,  # 标签的类数
            "weight_decay": 0.00004,
            "base_lr": 0.0001,  # 初始的学习率
            "num_epochs": 500,  # 总的epochs数
            "total_step": 20000,  # 总的步数，计算方式：num_epochs * (样本总数 / batch_size)
            "image_shape": [240, 320],  # 图像的大小
            "enable_ce": False,
            "bn_momentum": 0.9997,
            "dropout_keep_prop": 0.9,
            "default_norm_type": 'gn',  # 默认的归一化方式
            "decode_channel": 48,
            "encode_channel": 256,
            "default_epsilon": 1e-3,
            "default_group_number": 32,
            "depthwise_use_cudnn": False,
            "is_train": True,
            "data_augmentation_config": {
                "use_augmentation": False,
                "min_resize": 0.5,
                "max_resize": 4,
                "crop_size": 240
            }
        }


name_scope = ""  # 网络中各层的命名空间


# 递归的方式对网络每一层进行命名
@contextlib.contextmanager
def scope(name):
    global name_scope
    bk = name_scope
    name_scope = name_scope + name + '/'
    yield  # 这里并不是为了把这个函数变为生成器，而是为了在这里产生中断，不让后面那个命令执行，知道最后with的生存周期结束，执行最后那句
    name_scope = bk


def check(data, number):
    """
    对输入数据检查，保证数据是一个列表而不是一个单纯的数字，
    如果是数字就将数据复制number份然后形成一个列表
    :param data:
    :param number:
    :return:
    """
    if isinstance(data, int):
        return [data] * number
    elif isinstance(data, list):
        assert len(data) == number
        return data
    else:
        raise "error data type " + data


class Deeplab3PlusModel(BaseClass):
    def __init__(self):
        base_data = self.init_data()
        self.decode_channel = base_data["decode_channel"]
        self.encode_channel = base_data["encode_channel"]
        self.label_number = base_data["num_classes"]
        self.bn_momentum = base_data["bn_momentum"]
        self.dropout_keep_prop = base_data["dropout_keep_prop"]
        self.is_train = base_data["is_train"]
        self.default_epsilon = base_data["default_epsilon"]
        self.default_norm_type = base_data["default_norm_type"]
        self.default_group_number = base_data["default_group_number"]
        self.depth_wise_use_cudnn = base_data["depth_wise_use_cudnn"]

        self.bn_regularizer = fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0)
        self.depth_wise_regularizer = fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0)

        self.op_results = {}

    # 对操作进行自动名字空间处理
    def append_op_result(self, result, name):
        op_index = len(self.op_results)
        name = name_scope + name + str(op_index)
        self.op_results[name] = result
        return result

    def conv(self, *args, **kwargs):
        """
        卷积:设置初始化标准差-->根据初始化标准差指定权重参数属性对象-->指定偏置参数属性对象-->卷积
        :param args:
        :param kwargs:
        :return:
        """
        # 先根据不同层设置参数初始化标准差
        if "xception" in name_scope:
            init_std = 0.09
        elif "logit" in name_scope:
            init_std = 0.01
        elif name_scope.endswith("depthwise/"):
            init_std = 0.33
        else:
            init_std = 0.06

        # 根据是不是深度可分离卷积来决定是否要使用正则化
        if name_scope.endswith("depthwise/"):
            regularizer = self.depth_wise_regularizer
        else:
            regularizer = None

        # 根据本层的名字来对参数的属性进行设置，包括名字、正则化初始化方式这三个
        init_val = fluid.initializer.TruncatedNormal(loc=0.0, scale=init_std)
        # # 指定权重参数属性的对象
        kwargs["param_attr"] = fluid.ParamAttr(
            name=name_scope + 'weights',
            regularizer=regularizer,
            initializer=init_val
        )

        #  指定偏置参数属性的对象
        if "bias_attr" in kwargs and kwargs['bias_attr']:
            init_val = fluid.initializer.ConstantInitializer(value=0.0)
            kwargs["bias_attr"] = fluid.ParamAttr(
                name=name_scope + "biases",
                regularizer=regularizer,
                initializer=init_val
            )
        else:
            kwargs["bias_attr"] = False
        kwargs["name"] = name_scope + "conv"

        return self.append_op_result(fluid.layers.conv2d(*args, **kwargs), "conv")

    @staticmethod
    def group_norm(input, G, eps=1e-5, param_attr=None, bias_attr=None):
        N, C, H, W = input.shape
        if C % G != 0:
            # print "group can not divide channle:", C, G
            for d in range(10):
                for t in [d, -d]:
                    if G + t <= 0: continue
                    if C % (G + t) == 0:
                        G = G + t
                        break
                if C % G == 0:
                    # print "use group size:", G
                    break
        assert C % G == 0

        x = fluid.layers.group_norm(input,
                                    groups=G,
                                    param_attr=param_attr,
                                    bias_attr=bias_attr,
                                    name=name_scope + 'group_norm')
        return x

    def bn(self, *args, **kwargs):
        """
        归一化处理
        :param args:
        :param kwargs:
        :return:
        """
        # 在这里选择采用BN还是全新的GN归一化方式
        if self.default_norm_type == "bn":
            # 使用BN归一化
            with scope("BatchNormal"):
                return self.append_op_result(
                    fluid.layers.batch_norm(
                        *args,
                        epsilon=self.default_epsilon,
                        momentum=self.bn_momentum,
                        param_attr=fluid.ParamAttr(name=name_scope + "gamma", regularizer=self.bn_regularizer),
                        bias_attr=fluid.ParamAttr(name=name_scope + "beta", regularizer=self.bn_regularizer),
                        moving_mean_name=name_scope + "moving_mean",
                        moving_variance_name=name_scope + "moving_variance",
                        **kwargs
                    ),
                    "bn")
        elif self.default_norm_type == "gn":
            # 使用GN(GroupNormal)归一化
            with scope("GroupNorm"):
                return self.append_op_result(
                    self.group_norm(
                        args[0],
                        groups=self.default_group_number,
                        epsilon=self.default_epsilon,
                        param_attr=fluid.ParamAttr(name=name_scope + "gamma", regularizer=self.bn_regularizer),
                        bias_attr=fluid.ParamAttr(name=name_scope + "beta", regularizer=self.bn_regularizer)
                    )
                    , "gn")
        else:
            raise "Unsupport norm type:" + self.default_norm_type

    def bn_relu(self, data):
        """
        添加激活函数
        :return:
        """
        return self.append_op_result(fluid.layers.relu(self.bn(data)), "relu")

    def relu(self, data):
        """
        激活运算
        :param data:
        :return:
        """
        return self.append_op_result(fluid.layers.relu(
            data,
            name=name_scope + "relu"
        ),
            "relu")

    def seperate_conv(self, input_data, channel, stride, filter, dilation=1, act=None):
        """
        空洞卷积，具体实现被分成了 可分离卷积-->分为depthwise（先对通道进行卷积）和ponitwise（再对点进行卷积）
        :param input_data:
        :param channel:
        :param stride:
        :param filter:
        :param dilation:
        :param act:
        :return:
        """
        with scope("depthwise"):  # 按通道卷积
            data = self.conv(
                input_data,
                input_data.shape[1],
                filter,
                stride,
                groups=input_data.shape[1],
                padding=(filter // 2) * dilation,
                dilation=dilation,
                use_cudnn=self.depth_wise_use_cudnn
            )
            data = self.bn(data)
            if act:
                data = act(data)

        with scope("pointwise"):  # 按点卷积
            data = self.conv(
                data,
                channel,
                1,
                1,
                groups=1,
                padding=0
            )
            data = self.bn(data)
            if act:
                data = act(data)
        return data

    def xception_block(self,
                       input_data,
                       channels,
                       strides=1,
                       filters=3,
                       dilation=1,
                       skip_conv=True,
                       has_skip=True,
                       activation_fn_in_separable_conv=False):
        """
        深度可分离卷积快
        :param input_data:
        :param channels:
        :param strides:
        :param filters:
        :param dilation:
        :param skip_conv:
        :param has_skip:
        :param activation_fn_in_separable_conv:
        :return:
        """
        repeat_num = 3
        # 将channels、filter、strides重复repeat_number次返回一个列表
        channels = check(channels, repeat_num)
        filters = check(filters, repeat_num)
        strides = check(strides, repeat_num)
        results = []
        for i in range(repeat_num):
            with scope("separable_conv" + str(i + 1)):
                if not activation_fn_in_separable_conv:
                    # 先做激活运算再做深度可分离卷积
                    data = self.relu(data)
                    data = self.seperate_conv(
                        data,
                        channels[i],
                        strides[i],
                        filters[i],
                        dilation=dilation
                    )
                else:
                    # 直接做深度可分离卷积
                    data = self.seperate_conv(
                        data,
                        channels[i],
                        strides[i],
                        filters[i],
                        dilation=dilation,
                        act=self.relu
                    )
                results.append(data)
        # 循环完成后有一次跳跃计算
        if not has_skip:
            return self.append_op_result(data, "xception_block"), results
        if skip_conv:
            with scope("shortcut"):
                skip = self.bn(self.conv(
                    input_data,
                    channels[-1],
                    1,
                    strides[-1],
                    groups=1,
                    padding=0
                ))
        else:
            skip = input_data
        return self.append_op_result(data + skip, "xception_block"), results

    def entry_flow(self, input_data):
        """
        1、32个卷积核为3*3，步长为2
        2、64个卷积核为3*3
        :param input_data:
        :return:
        """
        with scope("entry_flow"):
            # 第一层：32个大小为3*3，步长为2，填充为1的卷积核进行卷积
            with scope("conv1"):
                data = self.conv(input_data, 32, 3, stride=2, padding=1)
                data = self.bn_relu(data)
            # 第二层：64个大小为3*3,步长为1，填充为1的卷积核进行卷积
            with scope("conv2"):
                data = self.conv(data, 64, 3, stride=1, padding=1)
                data = self.bn_relu(data)
            # 深度可分离卷积层1：第三、四、五层：深度可分离卷积循环三次（卷积核数量128,卷积核大小3*3）
            with scope("block1"):
                data, _ = self.xception_block(
                    data,
                    128,  # 卷积核数量
                    [1, 1, 2]  # 卷积核步长,三次的步长不相同
                )

            # 深度可分离卷积层2：第六、七、八层：深度可分离卷积循环三次(256个卷积核，大小3*3)
            with scope("block2"):
                data, results = self.xception_block(
                    data,
                    256,  # 卷积核数量
                    [1, 1, 2]  # 卷积核步长,三次的步长不相同
                )

            # 深度可分离卷积层3：第九、十、十一层：深度可分离卷积循环三次（728个卷积核，大小3*3）
            with scope("block3"):
                data, _ = self.xception_block(
                    data,
                    728,  # 卷积核数量
                    [1, 1, 2]  # 卷积核步长,三次的步长不相同
                )
        return data, results[1]

    def middle_flow(self, input_data):
        """
        中间表部分：深度可分离卷积块循环16次，每次有3层,每层728个卷积核，卷积核大小3*3
        :param input_data:
        :return:
        """
        with scope("middle_flow"):
            for i in range(16):
                with scope("block" + str(i)):
                    data, _ = self.xception_block(
                        input_data,
                        728,
                        [1, 1, 1],
                        skip_conv=False
                    )
        return data

    def exit_flow(self, input_data):
        """
        两个深度可分离卷积块，一次跳跃结构，最后一个深度分离卷积块没有跳跃结构
        :param input_data:
        :return:
        """
        with scope("exit_flow"):
            with scope("block1"):
                data, _ = self.xception_block(
                    input_data,
                    [728, 1024, 1024],
                    [1, 1, 1]
                )
            with scope("block2"):
                data, _ = self.xception_block(
                    data,
                    [1536, 1536, 2048],
                    [1, 1, 1],
                    dilation=2,
                    has_skip=False,
                    activation_fn_in_separable_conv=True
                )
        return data

    def dropout(self, x, keep_rate):
        """"""
        if self.is_train:
            return fluid.layers.dropout(x, 1 - keep_rate) / keep_rate
        else:
            return x

    def encoder(self, input_data):
        """
        膨胀率为0,6,12,18和imagePooling进行特征融合
        :param input_data:
        :return:
        """
        with scope("encoder"):
            channel = 256
            with scope("image_pool"):
                image_avg = fluid.layers.reduce_mean(input_data, [2, 3], keep_dim=True)  # 这个函数是用来求给定维度平均值
                # 寻找图像平均值
                self.append_op_result(image_avg, "reduce_mean")
                image_avg = self.bn_relu(
                    self.conv(
                        image_avg,
                        channel,
                        1,
                        1,
                        groups=1,
                        padding=0
                    )
                )
                # 图像缩放（双线性插值）
                image_avg = fluid.layers.resize_bilinear(image_avg, input_data.shape[2:])

            # 进行膨胀率为0,6,12,18的膨胀卷积
            with scope("aspp0"):
                aspp0 = self.bn_relu(self.conv(input_data, channel, 1, 1, groups=1, padding=0))

            with scope("asspp6"):
                aspp6 = self.bn_relu(self.conv(input_data, channel, 1, 3, dilation=6, act=self.relu))

            with scope("aspp12"):
                aspp12 = self.bn_relu(self.conv(input_data, channel, 1, 3, dilation=12, act=self.relu))

            with scope("aspp18"):
                aspp18 = self.bn_relu(self.conv(input_data, channel, 1, 3, dilation=18, act=self.relu))

            # 特征融合
            with scope("concat"):
                data = self.append_op_result(
                    fluid.layers.concat([image_avg, aspp0, aspp6, aspp12, aspp18], axis=1), "concat"
                )
            # 特征融合后进行1*1的卷积
            data = self.bn_relu(self.conv(data, channel, 1, 1, groups=1, padding=0))
            data = self.dropout(data, self.dropout_keep_prop)
            return data

    def decoder(self, encode_data, decode_shortcut):
        """"""
        with scope("decoder"):
            with scope("concat"):
                # 骨干网输出部分卷积，压缩通道数量
                decode_shortcut = self.bn_relu(
                    self.conv(decode_shortcut, self.decode_channel, 1, 1, groups=1, padding=0)
                )
                # 编码输出的部分数据上采样
                encode_data = fluid.layers.resize_bilinear(
                    encode_data, decode_shortcut.shape[2:]
                )
                # 特征融合
                encode_data = fluid.layers.concat([encode_data, decode_shortcut], axis=1)
                self.append_op_result(encode_data, "concat")
        # 再接两个卷积层
        with scope("separable_conv1"):
            encode_data = self.seperate_conv(
                encode_data,
                self.encode_channel,
                1,
                3,
                dilation=1,
                act=self.relu
            )
        with scope("separable_conv2"):
            encode_data = self.seperate_conv(
                encode_data,
                self.encode_channel,
                1,
                3,
                dilation=1,
                act=self.relu
            )
        return encode_data

    def net(self, img):
        """
        主干网部分：采用更深的xception网络，所有max pooling结构为stride=2的深度可分离卷积代替；
                  每个3*3的depthwise卷积都跟BN和Relu。
                  主干网部分由三部分组成：Entry flow ; Middle flow ; Exit flow
        :param img:
        :return:
        """
        self.append_op_result(img, "img")
        with scope("xception_65"):  # xception共65个卷积层
            # entry flow 共11层
            self.default_epsilon = 1e-3
            data, decode_shortcut = self.entry_flow(img)

            # middle flow 共48层
            data = self.middle_flow(data)

            # 出口部分 共6层
            data = self.exit_flow(data)
        # 接一个encode、decode
        self.default_epsilon = 1e-5
        encode_data = self.encoder(data)
        decode_data = self.decoder(data, decode_shortcut)

        # 输出层
        with scope("logit"):
            logit = self.conv(
                encode_data,
                self.label_number,
                1,
                stride=1,
                padding=0,
                bias_attr=True
            )
            # 上采样层调节输出矩阵大小
            logit = fluid.layers.resize_bilinear(
                logit,
                img.shape[2:]  # 原图的高度、宽度
            )
        return logit


class PictureSegment(BaseClass):
    def __init__(self):
        self.base_data = self.init_data()
        self.data_path = os.path.join(DATA_PATH, "iccv09Data")
        self.deeplab3_plus = Deeplab3PlusModel()

    @staticmethod
    def init_log_config(log_nm):
        """
        初始化日志相关配置
        :return:
        """

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_path = os.path.join(os.getcwd(), 'logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_name = os.path.join(log_path, log_nm)
        sh = logging.StreamHandler()
        fh = logging.FileHandler(log_name, mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.addHandler(fh)
        return logger

    @staticmethod
    def loss(logit, label, num_classes):
        """
        损失函数
        :param logit:
        :param label:
        :param num_classes:
        :return:
        """
        # assign :将输入张量或数组拷贝到新张量中（或拷贝生成一个新张量）
        # less_than:主元素比较两个张量，返回一个同维度布尔类型张量
        label_nignore = fluid.layers.less_than(
            label.astype("float32"),
            fluid.layers.assign(np.array([num_classes], "float32")),
            force_cpu=False
        ).astype("float32")
        # transpose:根据perm对输入多维的Tensor进行数据重排。返回多维Tensor的第i维对应输入Tensor的perm[i]维
        logit = fluid.layers.transpose(logit, [0, 2, 3, 1])  # 将预测结果的维度进行重排，将通道维度放在最后
        logit = fluid.layers.reshape(logit, [-1, num_classes])
        label = fluid.layers.reshape(label, [-1, 1])
        label = fluid.layers.cast(label, 'int64')
        label_nignore = fluid.layers.reshape(label_nignore, [-1, 1])
        logit = fluid.layers.softmax(logit, use_cudnn=False)
        loss = fluid.layers.cross_entropy(logit, label, ignore_index=8)  # 不考虑值为8的元素
        label_nignore.stop_gradient = True
        label.stop_gradient = True
        return loss, label_nignore

    @staticmethod
    def optimizer_momentum_setting(**kwargs):
        learning_rate = fluid.layers.polynomial_decay(
            kwargs["base_lr"],
            kwargs["total_step"],
            end_learning_rate=0,
            power=0.9
        )
        optimizer = fluid.optimizer.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.1,
            regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=kwargs["weight_decay"])
        )
        return optimizer

    def load_pretrained_model(self, exe, program, save_model_dir, logger):
        """
        加载增量训练模型
        :param exe:
        :param program:
        :param save_model_dir:
        :param logger:
        :return:
        """
        if self.base_data["continue_train"] and os.path.exists(save_model_dir):
            fluid.io.load_persistables(executor=exe, dirname=save_model_dir, main_program=program)
            logger.info("***************已读入增量训练模型，并准备继续训练***************")
        else:
            pass

    @staticmethod
    def load_paddle_model(exe, program, paddle_model_dir, logger):
        """
        加载与训练模型
        :param exe:
        :param program:
        :param paddle_model_dir:
        :param logger:
        :return:
        """
        load_vars = [x for x in program.list_vars() if
                     isinstance(x, fluid.framework.Parameter) and x.name.find("logit") == -1]
        fluid.io.load_vars(exe, dirname=paddle_model_dir, vars=load_vars)
        logger.info("***************已读入预先训练的模型，并准备继续训练***************")

    # ***************************************reader部分开始***************************************
    @staticmethod
    def custom_batch_reader(batch_size, reader):
        batch_img = []
        batch_label = []
        for img, label in reader:
            if img.shape[0] == 240 and img.shape[1] == 320:
                batch_img.append(img)
                batch_label.append(label)
            if len(batch_img) == batch_size:
                yield batch_img, batch_label
                batch_img = []
                batch_label = []
        if len(batch_img) != 0:
            yield batch_img, batch_label

    def general_train_file(self):
        """
        生成训练集文件
        :return:
        """
        image_path = os.path.join(self.data_path, "images")
        label_path = os.path.join(self.data_path, "labels")

        def file_list(dirname, ext=".jpg"):
            """
            获取目录下所有特定后缀的文件
            :param dirname:str 目录的完整路径
            :param ext:str 后缀名，以点号开头
            :return: list（str)所有文件夹名（不包含路径组成的列表）
            """
            return list(filter(lambda filename: os.path.splitext(filename)[1] == ext, os.listdir(dirname)))

        # 对文件列表进行打乱
        image_list = file_list(image_path)
        np.random.shuffle(image_list)

        # 取其中的80%作为训练集
        total_num = len(image_list)
        train_num = int(total_num * 0.8)
        # 创建训练文件
        with open(os.path.join(self.data_path, "train_list.txt"), "w") as train_f:
            for i in range(train_num):
                train_data_path = os.path.join("images/", image_list[i])
                train_label_path = os.path.join("labels/", image_list[i])
                train_label_path = train_label_path.replace("jpg", "region.txt")
                lines = train_data_path + "\t" + train_label_path + "\n"
                train_f.write(lines)
        with open(os.path.join(self.data_path, "eval_list.txt"), "w") as eval_f:
            for i in range(train_num, total_num):
                eval_data_path = os.path.join("images/", image_list[i])
                eval_label_path = os.path.join("labels/", image_list[i])
                eval_label_path = eval_label_path.replace("jpg", "regions.txt")
                lines = eval_data_path + "\t" + eval_label_path + "\n"
                eval_f.write(lines)

    def slice_with_pad(self, a, s, value=0):
        pads = []
        slices = []
        for i in range(len(a.shape)):
            if i > len(s):
                pads.append([0, 0])
                slices.append([0, a.shape[i]])
            else:
                l, r = s[i]
                if l < 0:
                    pl = -l
                    l = 0
                else:
                    pl = 0
                if r > a.shape[1]:
                    pr = r - a.shape[i]
                    r = a.shape[i]
                else:
                    pr = 0
                pads.append([pl, pr])
                slices.append([l, r])
        slices = list(map(lambda x: slice(x[0], x[1], 1), slices))
        a = a[slices]
        a = np.pad(a, pad_width=pads, mode="constant", constant_values=value)
        return a

    def custom_reader(self, file_list, mode="train"):
        def reader():
            np.random.shuffle(file_list)
            for lines in file_list:
                if mode == "train" or mode == "eval":
                    image_path, label_path = lines.strip("\t")
                    image_path = os.path.join(self.data_path, image_path)
                    label_path = os.path.join(self.data_path, label_path)
                    # 数据读入进来后，类型为numpy.ndarray
                    img = cv2.imread(image_path)

                    # 读标签
                    lab = []
                    with open(label_path, "r") as of_data:
                        for line in of_data.readlines():
                            temp = line.strip().split(" ")
                            temp_lab = []
                            for i in range(len(temp)):
                                if int(temp[i]) == -1:
                                    temp_lab.append(8)
                                else:
                                    temp_lab.append(int(temp[i]))
                            lab.append(temp_lab)
                    lab = np.array(lab)
                    if not self.base_data["data_augmentation_config"]["use_augmentation"]:
                        yield img, lab
                    else:
                        if np.random.rand() > 0.5:
                            range_l = 1
                            range_r = self.base_data["data_augmentation_config"]["max_resize"]
                        else:
                            range_l = self.base_data["data_augmentation_config"]["min_resize"]
                            range_r = 1
                        random_scale = np.random.rand(1) * (range_r - range_l) + range_l
                        crop_size = int(self.base_data["data_augmentation_config"]["crop_size"] / random_scale)
                        bb = crop_size // 2
                        offset_x = np.random.randint(bb, max(bb + 1, img.shape[0] - bb)) - crop_size // 2
                        offset_y = np.random.randint(bb, max(bb + 1, img.shape[1] - bb)) - crop_size // 2
                        img_crop = self.slice_with_pad(img, [[offset_x, offset_x + crop_size],
                                                             [offset_y, offset_y + crop_size]], 128)
                        img = cv2.resize(img_crop, (self.base_data["image_shape"][0], self.base_data["image_shape"][1]))
                        label_crop = self.slice_with_pad(lab, [[offset_x, offset_x + crop_size],
                                                               [offset_y, offset_y + crop_size]], 8)
                        lab = cv2.resize(label_crop,
                                         (self.base_data["image_shape"][0], self.base_data["image_shape"][1]),
                                         interpolation=cv2.INTER_NEAREST)
                        yield img, lab
                if mode == "test":
                    image_path = os.path.join(self.data_path, lines)
                    yield cv2.imread(image_path)

        return reader()

    @staticmethod
    def label2img(lab):
        x, y = lab.shape
        label_img = np.zeros((x, y, 3))
        label_dic = {1: [0.0, 0.0, 255.0], 2: [255.0, 0.0, 0.0], 3: [0.0, 255.0, 0.0], 4: [255.0, 255.0, 0.0],
                     5: [255.0, 0.0, 255.0], 6: [0.0, 255.0, 255.0], 7: [255.0, 255.0, 255.0], 0: [0.0, 0.0, 0.0],
                     8: [100, 100, 100]}
        for i in range(lab.shape[0]):
            for j in range(lab.shape[1]):
                key = lab[i][j]
                if key < 0:
                    key = 8
                label_img[i][j][0] = label_dic[key][0]
                label_img[i][j][1] = label_dic[key][1]
                label_img[i][j][2] = label_dic[key][2]
        return label_img

    # ***************************************reader部分结束***************************************
    def train(self):
        """
        训练模型
        :return:
        """
        # 创建parser对象
        parser = argparse.ArgumentParser(description="Short sample app")
        parser.add_argument("--use_gpu", type=int, default=0, required=True, help="0:not use_gpu,1:use_gpu")
        parser.add_argument("--num_epochs", type=int, default=None)
        parser.add_argument("--continue_train", type=int, default=0, required=True,
                            help="0:not continue_train,1:continue_train")
        parser.add_argument("--paddle_flag", type=int, default=0, required=True,
                            help="0:not use paddle model zoo's pretrained model,1:use paddle model zoo's pretrained model")
        mlh_args = parser.parse_args()
        self.base_data["use_gpu"] = bool(mlh_args.use_gpu)
        self.base_data["continue_train"] = bool(mlh_args.continue_train)
        self.base_data["paddle_flag"] = bool(mlh_args.paddle_flag)

        image_shape = self.base_data["image_shape"]
        enable_ce = self.base_data["enable_ce"]
        # fluid.default_startup_program()：初始化启动程序，返回默认全局的program（静态图）；执行参数初始化操作
        # fluid.default_main_program()：获取当前用于存储op（操作）和variable（变量）描述信息的program；
        #                               fluid.layers中添加的ope和variable会存储在mainProgram中
        sp = fluid.Program()  # 初始化参数
        tp = fluid.Program()  # mainProgram全局
        if enable_ce:
            seed = 102
            # random_seed：随机种子，用于产生相同的随机数序列
            sp.random_seed = seed
            tp.random_seed = seed
        # 开始训练:program_guard:声明静态图
        with fluid.program_guard(main_program=tp, startup_program=sp):
            # 采用同步方式读取数据
            img = fluid.layers.data(name='image', shape=[3] + image_shape, dtype="float32")  # 定义数据
            label = fluid.layers.data(name='label', shape=image_shape, dtype="int32")  # 定义标签

            # 模型计算输出(输出结果为4D张量)
            logit = self.deeplab3_plus.net(img)

            # 对每个结果在深度方向进行比较，返回最大值索引
            pred = fluid.layers.argmax(logit, axis=1).astype("int32")
            # 求损失函数
            loss, mask = self.loss(logit, label, num_classes=self.base_data["num_classes"])

            area = fluid.layers.elementwise_max(
                fluid.layers.reduce_mean(mask),
                fluid.layers.assign(np.array([0.1], dtype=np.float32))
            )

            loss_mean = fluid.layers.reduce_mean(loss) / area

            loss_mean.persistable = True

            # 优化器
            optimizer = self.optimizer_momentum_setting(
                base_lr=self.base_data["base_lr"],
                total_step=self.base_data["total_step"],
                weight_decay=self.base_data["weight_decay"]
            )
            optimize_ops, param_grads = optimizer.minimize(loss_mean, startup_program=sp)

            for p, g in param_grads:
                g.persistable = True

        place = fluid.CUDAPlace(0) if self.base_data["use_gpu"] else fluid.CPUPlace()

        train_path = os.path.join(self.data_path, "train_list.txt")
        # data_dir = ('data/iccv09Data')
        file_list = []
        with open(train_path, 'r') as f:
            for line in f.readlines():
                lines = line.strip()
                file_list.append(lines)
        exe = fluid.Executor(place)
        exe.run(sp)  # 初始化

        now = time.strftime("%Y-%m-%d")
        log_name = "train_log_" + now
        logger = self.init_log_config(log_name)
        logger.info("train params:%s", str(self.base_data))

        if self.base_data["paddle_flag"]:
            self.load_paddle_model(exe, tp, self.base_data["pretrained_model_dir"], logger)  # 加载预训练模型
        else:
            self.load_pretrained_model(exe, tp, self.base_data["train_model_dir"], logger)  # 加载增量训练模型
        logger.info("*********************************训练开始*********************************")
        if mlh_args.num_epochs:
            num_epochs = mlh_args.num_epochs
        else:
            num_epochs = self.base_data["num_epochs"]
        batch_size = self.base_data["batch_size"]
        for pass_id in range(num_epochs):
            step_num = 1
            total_time = []
            total_loss = []
            for imgs, labs in self.custom_batch_reader(batch_size, self.custom_reader(file_list, mode="train")):
                t1 = time.time()
                imgs = np.array(imgs)
                labs = np.array(labs)
                imgs = imgs.transpose([0, 3, 1, 2])
                img = imgs.astype(np.float32)
                labs = labs.astype(np.int32)
                loss = exe.run(tp, feed={"image": imgs, "label": labs}, fetch_list=[loss_mean])
                period = time.time() - t1
                loss = np.mean(np.array(loss))
                if step_num % 10 == 0:
                    logger.info("epoch:{0} step:{1} loss:{2} period:{3}".format(pass_id, step_num * batch_size, loss,
                                                                                "%2.2f sec" % period))
                step_num += 1
                total_time.append(period)
                total_loss.append(loss)
            logger.info("{0}'s epoch_total_time:{1} && mean_loss:{2}".format(pass_id, "%2.2f sec" % sum(total_time),
                                                                             sum(total_loss) / len(total_loss)))
            if pass_id % 10 == 0:
                # 每隔10个epoch保存一次模型
                logger.info("暂时存储第{0}个epoch的训练结果".format(pass_id))
                fluid.io.save_persistables(dirname=self.base_data["train_model_dir"], main_program=tp, executor=exe)
        logger.info("****************训练完成****************")
        fluid.io.save_persistables(dirname=self.base_data["train_model_dir"], main_program=tp, executor=exe)

    def mean_iou(self, pred, label):
        label = fluid.layers.elementwise_min(
            label,
            fluid.layers.assign(np.array([self.base_data["num_classes"]], dtype=np.int32))
        )
        label_ignore = (label == self.base_data["num_classes"]).astype("int32")
        label_nignore = (label != self.base_data["num_classes"]).astype("int32")

        pred = pred * label_nignore + label_ignore * self.base_data["num_classes"]
        miou, wrong, correct = fluid.layers.mean_iou(pred, label, self.base_data["num_classes"] + 1)
        return miou, wrong, correct

    def eval_model(self):
        """
        对模型进行评估
        :return:
        """
        image_shape = self.base_data["image_shape"]
        sp = fluid.Program()
        tp = fluid.Program()
        batch_size = 1
        deeplab_3p_model = Deeplab3PlusModel()
        with fluid.program_guard(main_program=tp, startup_program=sp):
            img = fluid.layers.data(name="img", shape=[3] + image_shape, dtype="float32")
            label = fluid.layers.data(name="label", shape=image_shape, dtype="int32")
            logit = deeplab_3p_model.net(img)
            pred = fluid.layers.argmax(logit, axis=1).astype("int32")
            miou, out_wrong, out_correct = self.mean_iou(pred, label)
        tp = tp.clone(True)
        fluid.memory_optimize(
            tp,
            print_log=False,
            skip_opt_set=set([pred.name, miou, out_wrong, out_correct]),
            level=1
        )

        place = fluid.CPUPlace()
        if self.base_data["use_gpu"]:
            place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(sp)
        fluid.io.load_persistables(exe, self.base_data("train_model_dir"), tp)
        file_list = []
        with open(self.base_data["eval_file_path"], "r") as f:
            for line in f.readlines():
                lines = line.strip()
                file_list.append(lines)
        sum_iou = 0
        all_correct = np.array([0], dtype=np.int64)
        all_wrong = np.array([0], dtype=np.int64)
        step = 1
        mean_result = []
        for imgs, labs in self.custom_batch_reader(batch_size, self.custom_reader(file_list, mode="train")):
            imgs = np.array(imgs)
            labs = np.array(labs)
            imgs = imgs.transpose([0, 3, 1, 2])
            imgs = imgs.astype(np.float32)
            labs = labs.astype(np.int32)
            result = exe.run(
                tp,
                feed={"img": imgs, "label": labs},
                fetch_list=[pred, miou, out_wrong, out_correct]
            )
            wrong = result[2][:-1] + all_wrong
            right = result[3][:-1] + all_correct
            all_wrong = wrong.copy()
            all_correct = right.copy()
            mp = (wrong + right) != 0
            miou2 = np.mean(right[mp] * 1.0 / (right[mp] + wrong[mp]))
            mean_result.append(miou2)
            step += 1
        print('eval done! total number of image is {}, mean iou: {}'.format(str(step), str(np.mean(mean_result))))

    def freeze_model(self):
        """
        保存推理模型
        :return:
        """
        deeplabv3p = Deeplab3PlusModel()
        exe = fluid.Executor(fluid.CPUPlace())
        image = fluid.layers.data(name="image", shape=[3] + self.base_data["image_shape"], dtype="float32")
        pred = deeplabv3p.net(image)
        freeze_program = fluid.default_main_program()
        fluid.io.load_persistables(exe, self.base_data["train_model_dir"], freeze_program)
        freeze_program = freeze_program.clone(for_test=True)
        fluid.io.save_inference_model(self.base_data["infer_model_dir"], ["image"], [pred], exe, freeze_program)

    def read_image(self, img_path):
        origin = cv2.imread(img_path)
        if origin.shape[0] != self.base_data["image_shape"][0] or origin.shape[1] != self.base_data["image_shape"][
            1] or len(origin.shape) != 3:
            print(
                "输入图片的大小不合适，我们需要的是{}，但是输入的是{}".format(str(self.base_data["image_shape"] + [3]),
                                                                            str(origin.shape)))
            exit()
        img = origin.astype("float32").transpose(2, 0, 1)
        return origin, [img]

    def infer_model(self):
        """
        对数据进行推理
        :return:
        """
        image_shape = self.base_data["image_shape"]
        deeplabv3p = Deeplab3PlusModel()
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
            dirname=self.base_data["infer_model_dir"],
            executor=exe
        )
        train_path = self.base_data["eval_file_path"]
        data_dir = self.base_data["data_dir"]
        file_list = []
        with open(train_path, "r") as f:
            for line in f.readlines():
                lines = line.strip()
                file_list.append(lines)
        test1 = file_list[15].split()
        origin, img = self.read_image(os.path.join(data_dir, test1[0]))
        img = np.array(img)
        output = exe.run(
            inference_program,
            feed={feed_target_names[0]: img},
            fetch_list=fetch_targets
        )
        output = output[0][0]
        output = output.transpose(1, 2, 0)
        label_out = np.zeros(image_shape)
        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                temp = output[i][j]
                temp_index = temp.tolist().index(max(temp))
                label_out[i][j] = temp_index
        label_out = self.label2img(label_out)
        cv2.imwrite("origin.jpg", origin)
        cv2.imwrite("result.jpg", label_out)
        print("运行结束，origin为预测图片，result为分割后的图片")
        # 显示分割结果

        img1 = plt.imread('origin.jpg')
        img2 = plt.imread('result.jpg')
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(img1)
        ax.set_title('origin')
        ax.axis('off')
        ax = fig.add_subplot(122)
        ax.imshow(img2)
        ax.set_title('result')
        ax.axis('off')
        plt.show()


if __name__ == '__main__':
    obj = PictureSegment()
    # 训练
    obj.train()
    # 评估
    obj.eval_model()
    # 保存预测模型
    obj.freeze_model()
    # 预测
    obj.infer_model()
