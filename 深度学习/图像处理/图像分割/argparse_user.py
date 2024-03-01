#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  argparse_user.py
:time  2024/2/29 10:29
:desc  argparse用法
"""
import argparse

"""
1、argparse模块是Python内置的命令用于命令项与参数解析的模块。
2、作用：argparse可以让用户轻松编写用户友好的命令行接口，帮助用户为模型定义参数
        定义命令行参数，解析命令行参数
"""

# 1.定义命令行解析器对象
parse = argparse.ArgumentParser(description="Demo of argparse")

# 2.添加命令行参数
parse.add_argument('--epochs', type=int, default=30)
parse.add_argument("--batch", type=int, default=4)

# 3.从命令行中结构化   解析参数
args_ = parse.parse_args()
print(args_)
epochs = args_.epochs
batch = args_.batch
print(f"epochs:{epochs};batch:{batch}")
