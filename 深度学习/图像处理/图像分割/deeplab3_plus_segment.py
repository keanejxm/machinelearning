#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/26 22:01
# @Author  : keane
# @Site    : 
# @File    : deeplab3_fenge.py
# @Software: PyCharm
import os
import time
import logging
import argparse
import numpy as np
import paddle.fluid as fluid
from common_utils import DATA_PATH

# 导入模型
from deeplab3_plus_model import Deeplab3PlusModel


class PictureSegment:
    def __init__(self):
        self.base_data = self.init_data()
        self.deeplab3_plus = Deeplab3PlusModel()

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

    def load_paddle_model(self, exe, program, paddle_model_dir, logger):
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

    def train(self):
        """
        训练模型
        :return:
        """
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

        train_path = "data/train_list.txt"
        data_dir = 'data/iccv09Data'
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
