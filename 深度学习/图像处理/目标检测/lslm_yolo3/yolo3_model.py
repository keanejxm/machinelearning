#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  yolo3_model.py
:time  2024/1/5 10:00
:desc  构建yolo3模型
"""
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay


class Yolo3Model:
    def __init__(self, class_num, anchors, anchor_mask):
        self.outputs = []  # 网络最终模型
        self.down_sample_ratio = 1  # 下采样率
        self.anchor_mask = anchor_mask
        self.anchors = anchors  # 锚点
        self.class_num = class_num  # 类别数量

        self.yolo_anchors = []
        self.yolo_classes = []

        for mask_pair in self.anchor_mask:
            mask_anchors = []
            for mask in mask_pair:
                mask_anchors.append(self.anchors[2 * mask])
                mask_anchors.append(self.anchors[2 * mask + 1])
            self.yolo_anchors.append(mask_anchors)
            self.yolo_classes.append(class_num)

    def get_anchors(self):
        return self.anchors

    def get_anchor_mask(self):
        return self.anchor_mask

    def get_class_num(self):
        return self.class_num

    def get_down_sample_ratio(self):
        return self.down_sample_ratio

    def get_yolo_anchors(self):
        return self.yolo_anchors

    def get_yolo_classes(self):
        return self.yolo_classes

    @staticmethod
    def conv_dbl(input_data, num_filters, filter_size, stride, padding, use_cudnn=True):
        # 卷积+batchNorm
        conv = fluid.layers.conv2d(
            input=input_data,
            num_filters=num_filters,  # 卷积核数量
            filter_size=filter_size,  # 卷积核的大小
            stride=stride,  # 步长
            padding=padding,  # 填充
            # act=None, # 激活函数
            use_cudnn=use_cudnn,  # 是否使用cudnn，cudnn利用cuda进行了加速处理
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),  # 指定权重参数属性的对象
            bias_attr=False  # 指定偏置参数属性的对象
        )
        # batch_norm:批量正则化
        # batch_norm中的参数不需要参与正则化，所以主动使用正则系数为0的正则项屏蔽掉
        # 在batch_norm中使用leaky的话，只能使用默认的alpha = 0.02；如果需要设置值，必须提出去单独来
        # 正则化的目的，是为了防止过拟合，较小的l2值能防止过拟合

        # ParamAttr:创建一个参数属性对象，用户可设置参数的名称、初始化方式、学习率、正则化规则、是否需要训练、梯度裁剪方式、是否做模型平均等属性
        # L2Decay: 实现 L2 权重衰减正则化，用于模型训练，有助于防止模型对训练数据过拟合
        param_attr = ParamAttr(initializer=fluid.initializer.Normal(0., 0.02), regularizer=L2Decay(0.))
        bias_attr = ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.))
        # batch_norm加上权重和偏置
        out = fluid.layers.batch_norm(input=conv, act=None, param_attr=param_attr, bias_attr=bias_attr)
        # leaky_relu：LeakeyReLU是给所有负值赋予一个非零斜率
        out = fluid.layers.leaky_relu(out, 0.1)
        return out

    # 通过卷积实现降采样（代替池化层）
    def down_sample(self, input_data, num_filters, filter_size=3, stride=2, padding=1):
        self.down_sample_ratio *= 2
        return self.conv_dbl(
            input_data=input_data,
            num_filters=num_filters,  # 卷积和数量
            filter_size=filter_size,  # 卷积核大小
            stride=stride,
            padding=padding)

    # 残差块res_unit(包含两个卷积层一个跳跃结构)
    def res_unit_block(self, input_data, num_filters):
        conv1 = self.conv_dbl(input_data, num_filters=num_filters, filter_size=1, stride=1, padding=0)
        conv2 = self.conv_dbl(conv1, num_filters=num_filters * 2, filter_size=3, stride=1, padding=1)
        # 输入 +输出
        out = fluid.layers.elementwise_add(x=input_data, y=conv2, act=None)
        return out

    # 基本块，一个填充zero_padding,一个dbl（conv_dbl）,n个残差块
    def resn_block(self, input_data, num_filters, count):
        for i in range(count):
            input_data = self.res_unit_block(input_data, num_filters)
        return input_data

    def yolo_detection_block(self, conv, num_filters):
        """检测模块"""
        # 创建4个卷积层
        for j in range(2):
            conv = self.conv_dbl(conv, num_filters=num_filters, filter_size=1, stride=1, padding=0)
            conv = self.conv_dbl(conv, num_filters=num_filters * 2, filter_size=3, stride=1, padding=1)
        # 返回倒数第二层结果
        route_ = self.conv_dbl(conv, num_filters=num_filters, filter_size=1, stride=1, padding=0)
        # 返回倒数第一层结果
        tip = self.conv_dbl(route_, num_filters=num_filters * 2, filter_size=3, stride=1, padding=1)
        return route_, tip

    # 上采样
    def up_sample(self, input, scale=2):
        shape_nchw = fluid.layers.shape(input)
        shape_hw = fluid.layers.slice(shape_nchw,
                                      axes=[0],
                                      starts=[2],
                                      ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype="int32")
        out_shape = in_shape * scale  # 计算输出数据形状
        out_shape.stop_gradient = True
        # 矩阵放大（最邻插值法）
        out = fluid.layers.resize_nearest(input=input,
                                          scale=scale,
                                          actual_shape=out_shape)
        return out

    # 构建darknet-53网络
    def net(self, img):
        stages = [1, 2, 8, 8, 4]
        assert len(self.anchor_mask) <= len(stages), "anchor masks can't bigger than down_sample times"
        # 第一个卷积层
        conv1 = self.conv_dbl(input_data=img, num_filters=32, filter_size=3, stride=1, padding=1)
        # 第二个卷积层
        out_data = self.down_sample(conv1, num_filters=conv1.shape[1] * 2)
        # 12884卷积层
        blocks = []
        for i, stage_count in enumerate(stages):
            # n个残差块（1,2,8,8,4）
            block = self.resn_block(input_data=out_data,
                                    num_filters=32 * (2 ** i),
                                    count=stage_count)  # 基本块数量
            blocks.append(block)
            # 用来代替池化层（池化层做降采样）的卷积层，使用步长为2的卷积代替池化层做降采样
            if i < len(stages) - 1:
                # 如果不是最后一组做降采样，用来代替池化层
                out_data = self.down_sample(block, num_filters=block.shape[1] * 2)
        blocks = blocks[-1:-4:-1]  # 取倒数三层，并且逆序，后面特征融合使用
        # ---------------------------Darknet-53骨干网络结束---------------------------
        route_ = None
        for i, block in enumerate(blocks):
            if i > 0 and route_:
                route_ = self.conv_dbl(route_, 256 // (2 ** i), filter_size=1, stride=1, padding=0)
                # 上采样
                route_ = self.up_sample(route_)
                block = fluid.layers.concat(input=[route_, block], axis=1)  # 上采样进行特征融合
            # 里面一共有6个卷积层，返回5个卷积后的结果和6个卷积后的结果
            route_, tip = self.yolo_detection_block(block, num_filters=512 // (2 ** i))
            param_attr = ParamAttr(initializer=fluid.initializer.Normal(0., 0.02))
            bias_attr = ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.))
            block_out = fluid.layers.conv2d(input=tip,
                                            num_filters=len(self.anchor_mask[i]) * (self.class_num + 5),
                                            filter_size=1,
                                            stride=1,
                                            padding=0,
                                            # act=None,
                                            param_attr=param_attr,
                                            bias_attr=bias_attr
                                            )
            self.outputs.append(block_out)
        return self.outputs


if __name__ == '__main__':
    print(fluid.initializer.Normal(0., 0.02))
    param_attr_ = ParamAttr(initializer=fluid.initializer.Normal(0., 0.02), regularizer=L2Decay(0.))
    print(param_attr_)
