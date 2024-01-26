#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  lslm_codes.py
:time  2024/1/26 9:21
:desc  
"""
import paddle.fluid as fluid

# default_startup_program：只运行一次来初始化参数
# default_main_program：在每个mini batch中运行并更新权重
# fluid.unique_name.guard：共享训练阶段和测试阶段的模型参数

"""
fluid内部执行流程：
    1、用户编写Python代码，向一段program中添加变量和对变量的操作
    2、program转为中间描述语言programDesc
    3、Transpiler接受一段programDesc输出一段变化后的programDesc,后端执行fluid program
    4、后端接受然后依次执行program中的指令
    使用时会默认创建startup_program和main_program
    startup_program：定义模型参数，输入输出，和模型中可学习参数的初始化操作
    main_program：定义神经网络
"""
