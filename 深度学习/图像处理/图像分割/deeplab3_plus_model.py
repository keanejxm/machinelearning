#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  deeplab3_model.py
:time  2024/2/29 11:26
:desc  deeplab3网络模型
"""
import paddle
from PIL import Image
import cv2
import paddle.fluid as fluid
import numpy as np
import contextlib  # 上下文管理的库
import os


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

# if __name__ == '__main__':
# obj = Deeplab3Model()
