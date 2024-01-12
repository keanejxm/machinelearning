#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
:author: keane
:file  train_predict.py
:time  2024/1/5 15:12
:desc  模型的训练与预测
"""
import os.path
import time

import numpy as np
import paddle.reader

from deal_image_data import DealData
from yolo3_model import Yolo3Model
import paddle.fluid as fluid
from PIL import Image, ImageDraw


class TrainModel:
    def __init__(self, logger):
        self._logger = logger

    # 读取器
    def paddle_data_reader(self):
        def reader():
            for img, ret_boxes, ret_lbls in DealData().read_img():
                yield img, ret_boxes, ret_lbls

        return reader

    # 定义优化器
    def optimizer_sgd_setting(self):
        batch_size = ""
        iters = 1
        iters = 1 if iters < 1 else iters
        # 学习率
        learning_strategy = ""
        lr = learning_strategy

        boundaries = [i * iters for i in learning_strategy[""]]
        values = [i * lr for i in learning_strategy[""]]
        self._logger.info(f"origin learning rate:{lr};boundaries:{boundaries};values:{values}")
        optimizer = fluid.optimizer.SGDOptimizer(
            learning_rate=fluid.layers.piecewise_decay(boundaries, values),  # 分段衰减学习率
            regularization=fluid.regularizer.L2Decay(0.00005)  # L2权重衰减正则化
        )
        return optimizer

    # 构造损失函数
    def get_losses(self, model, outputs, gt_box, gt_label):
        losses = []
        downsample_ratio = 1
        for i, out in enumerate(outputs):
            loss = fluid.layers.yolov3_loss(
                x=out,
                gt_box=gt_box,
                gt_label=gt_label,
                anchors=model.get_anchors(),
                anchor_mask=model.get_anchor_mask()[i],
                class_num=model.get_class_num(),
                ignore_thresh=0.7,
                use_label_smooth=False,
                downsample_ratio=downsample_ratio
            )
            losses.append(fluid.layers.reduce_mean(loss))
            downsample_ratio //= 2
        loss = sum(losses)
        optimizer = self.optimizer_sgd_setting()
        optimizer.minimize(loss)
        return loss

    def load_pretrained_params(self, exe, program):
        if "continue_train" and os.path.exists("save_model_dir"):
            self._logger.info("load param from retrain model")
            fluid.io.load_persistables(
                executor=exe,
                dirname="",
                main_program=program
            )
        elif "pretrained" and os.path.exists(""):
            self._logger.info("load parm from pretrained model")

            def if_exist(var):
                return os.path.exists(os.path.join("", var.name))

            fluid.io.load_vars(exe, "", main_program=program, predicate=if_exist())

    def train_model(self):
        place = fluid.CPUPlace()
        # 创建2个program
        train_program = fluid.Program()
        start_program = fluid.Program()

        # "max_box_num": 20,  # 一幅图上最多有多少个目标
        max_box_num = 20
        with fluid.program_guard(train_program, start_program):
            img = fluid.layers.data(name="img", shape=[3, 448, 448], dtype="float32")
            gt_box = fluid.layers.data(name="gt_box", shape=[max_box_num, 4], dtype="float32")
            gt_label = fluid.layers.data(name="gt_label", shape=[max_box_num], dtype="int32")
            feeder = fluid.DataFeeder(feed_list=[img, gt_box, gt_label], place=place, program=train_program)
            reader = self.paddle_data_reader()
            reader = paddle.reader.shuffle(reader, buf_size=32)
            reader = paddle.batch(reader, batch_size=32)
        # 创建模型
        with fluid.unique_name.guard():
            # 创建yolo模型
            anchors = [7, 10, 12, 22, 24, 17, 22, 45, 46, 33, 43, 88, 85, 66, 115, 146, 275, 240]
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # Anchor Box序号
            model = Yolo3Model(-1, anchors, anchor_mask)
            output = model.net(img)
            loss = self.get_losses(model, output, gt_box, gt_label)
        self._logger.info("build executor and init params")

        # 创建exe,加载增量模型
        exe = fluid.Executor(place)
        exe.run(start_program)
        train_fetch_list = [loss.name]
        # 加载增量模型
        self.load_pretrained_params(exe, train_program)
        stop_strategy = ""
        successive_limit = stop_strategy[""]
        sample_freq = stop_strategy[""]
        min_curr_map = stop_strategy[""]
        min_loss = stop_strategy[""]
        stop_train = True
        successive_count = 0
        total_batch_count = 0
        valid_thresh = ""
        nms_thresh = ""
        current_best_loss = 10000000000.0

        # 开始迭代训练
        for pass_id in range(1):
            self._logger.info(f"current pass:{pass_id},start read image")
            batch_id = 0
            total_loss = 0.0
            for batch_id, data in enumerate(reader()):
                t1 = time.time()
                loss = exe.run(train_program,
                               feed=feeder.feed(data),
                               fetch_list=train_fetch_list)
                period = time.time() - t1
                loss = np.mean(np.array(loss))
                total_loss += loss
                batch_id += 1
                total_batch_count += 1
                if batch_id % 20 == 0:  # 调整日志输出频率
                    self._logger.info(f"pass:{pass_id};trainbatch:{batch_id};loss:{loss},time:{period}")
            pass_mean_loss = total_loss / batch_id
            self._logger.info(f"pass:{pass_id} train result,current pass mean loss:{pass_mean_loss}")
            ###模型评估###

            # 采用每训练完一轮停止办法，可以调整为更精细的保存策略
            if pass_mean_loss < current_best_loss:
                self._logger.info(
                    "temp save {} epcho train result,current best pass loss{}".format(pass_id, pass_mean_loss))
                fluid.io.save_persistables(dirname="",
                                           main_program=train_program,
                                           executor=exe)
                current_best_loss = pass_mean_loss
        self._logger.info("training till last epcho,end training")
        fluid.io.save_persistables(dirname="",
                                   main_program=train_program,
                                   executor=exe)

    # 固化保存模型
    def freeze_model(self):
        """保存模型，用来预测"""
        exe = fluid.Executor(fluid.CPUPlace())
        path = ""
        model = Yolo3Model("class_dim", "anchors", "anchor_mask")
        image = fluid.layers.data(name="image", shape="input_size", dtype="float32")
        image_shape = fluid.layers.data(name="image_shape", shape=[2], dtype="int32")
        boxes = []
        scores = []
        outputs = model.net(image)
        downsample_ratio = ""
        for i, out in enumerate(outputs):
            box, score = fluid.layers.yolo_box(x=out,
                                               img_size=image_shape,
                                               anochors=model.get_class_num(),
                                               conf_tresh="valid_thresh",
                                               downsample_ratio=downsample_ratio,
                                               name="yolo_box_" + str(i))
            boxes.append(box)
            scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))
            downsample_ratio //= 2
        pred = fluid.layers.multiclass_nms(bboxes=fluid.layers.concat(boxes, axis=1),
                                           scores=fluid.layers.concat(scores, axis=2),
                                           score_threshold="valid_thresh",
                                           nms_top_k="nms_top_k",
                                           keep_top_k="nms_top_k",
                                           nms_threshold="nmn_thresh",
                                           background_label=-1,
                                           name="multiclass_nms")
        freeze_program = fluid.default_main_program()
        fluid.io.load_persistables(exe, path, freeze_program)
        freeze_program = freeze_program.clone(for_test=True)
        print("freeze out: {0}, pred layout: {1}".format(['freeze_dir'], pred))

        # 保存模型
        fluid.io.save_inference_model("",
                                      ["image", "image_shape"],
                                      pred,
                                      exe,
                                      freeze_program)
        print("freeze end")


class ModelPredict:
    def __init__(self):
        self.target_size = ""
        self.anchors = ""
        self.anchor_mask = ""
        self.label_dict = ""
        self.class_dim = ""

    def resize_img(self, img):
        """
        保持比例缩放图片
        :param img:
        :param target_size:
        :return:
        """
        img = img.resize(self.target_size[1:], Image.BILINEAR)
        return img

    def read_image(self, image_path):
        origin = Image.open(image_path)
        img = self.resize_img(origin)
        resized_img = img.copy()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = np.array(img).astype("float32").transpose((2, 0, 1))
        img -= 127.5
        img *= 0.007843
        img = img[np.newaxis, :]
        return origin, img, resized_img

    def draw_bbox_image(self, img, boxes, labels, save_name):
        img_width, img_height = img.size
        draw = ImageDraw.Draw(img)  # 图像绘制对象
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            draw.rectangle((xmin, ymin, xmax, ymax), None, "red")
            draw.text((xmin, ymin), self.label_dict[int(label)], (255, 255, 0))  # 绘制标签
        img.save(save_name)
        # display(img)

    def predict_model(self, image_path):
        """预测"""
        origin, tensor_img, resized_img = self.read_image(image_path)
        input_w, input_h = origin.size[0], origin.size[1]
        image_shape = np.array([input_h, input_w], dtype="int32")
        t1 = time.time()
        # 执行预测
        place = fluid.CUDAPlace(0) if "" else fluid.CPUPlace()
        exe = fluid.Executor(place)
        path = ""
        inference_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(dirname=path, executor=exe)
        batch_outputs = exe.run(inference_program,
                                feed={feed_target_names[0]: tensor_img,
                                      feed_target_names[1]: image_shape[np.newaxis, :]},
                                fetch_list=fetch_targets,
                                return_numpy=False)
        period = time.time() - t1
        print("predict cost time:{0}".format("%2.2f sec" % period))
        bboxes = np.array(batch_outputs[0])  # 预测结果
        if bboxes.shape[1] != 6:
            print("No object fund in {}".format(image_path))
            return
        labels = bboxes[:, 0].astype("int32")  # 类别
        scores = bboxes[:, 1].astype("float32")  # 概率
        boxes = bboxes[:, 2].astype("float32")  # 边框
        last_dot_index = image_path.rfind(".")
        out_path = image_path[:last_dot_index]
        out_path += "-result.jpg"
        draw_bbox_image = (origin, boxes, labels, out_path)
