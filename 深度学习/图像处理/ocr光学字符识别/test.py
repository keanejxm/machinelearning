#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  test.py
:time  2024/3/7 9:48
:desc  
"""
train_params = {
    "input_size": [1, 48, 512],  # 输入数据维度
    "data_dir": "data/data6927/word-recognition",  # 数据集路径
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
    "early_stop": { # 控制训练停止条件
        "sample_frequency": 50,
        "successive_limit": 5,
        "min_instance_error": 0.1
    }
}