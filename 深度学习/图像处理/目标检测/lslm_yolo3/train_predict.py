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

from lslm_deal_img_data import DealImgData
from yolo3_model import Yolo3Model
import paddle.fluid as fluid
from PIL import Image, ImageDraw


class TrainModel(DealImgData):
    # 读取器
    @staticmethod
    def paddle_data_reader():
        def reader():
            for img, ret_boxes, ret_lbls in DealImgData().read_img(mode="train"):
                yield img, ret_boxes, ret_lbls

        return reader

    # 定义优化器
    def optimizer_sgd_setting(self):
        batch_size = self.img_param["trainBatchSize"]
        iters = self.img_param["imgCount"] // batch_size
        iters = 1 if iters < 1 else iters
        # 学习率
        sgd_strategy = self.img_param["sgdStrategy"]
        lr = sgd_strategy["learningRate"]

        boundaries = [i * iters for i in sgd_strategy["lrEpochs"]]
        values = [i * lr for i in sgd_strategy["lrDecay"]]
        print(f"origin learning rate:{lr};boundaries:{boundaries};values:{values}")
        optimizer = fluid.optimizer.SGDOptimizer(
            learning_rate=fluid.layers.piecewise_decay(boundaries, values),  # 分段衰减学习率
            regularization=fluid.regularizer.L2Decay(0.00005)  # L2权重衰减正则化
        )
        return optimizer

    # 构造损失函数
    # @staticmethod
    def get_losses(self,model, outputs, gt_box, gt_label):
        losses = []
        down_sample_ratio = model.get_down_sample_ratio()
        with fluid.unique_name.guard("train"):
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
                    downsample_ratio=down_sample_ratio
                )
                losses.append(fluid.layers.reduce_mean(loss))
                down_sample_ratio //= 2
            loss = sum(losses)
            # 4、定义优化器
            optimizer = self.optimizer_sgd_setting()
            optimizer.minimize(loss)
            return loss

    def load_pretrained_params(self, exe, program):
        if os.path.exists(self.img_param["saveModelDir"]):
            fluid.io.load_persistables(
                executor=exe,
                dirname=self.img_param["saveModelDir"],
                main_program=program
            )
        elif os.path.exists(self.img_param["pretrainedModelDir"]):

            def if_exist(var):
                return os.path.exists(os.path.join(self.img_param["pretrainedModelDir"], var.name))

            fluid.io.load_vars(exe, self.img_param["pretrainedModelDir"], main_program=program, predicate=if_exist)

    def train_model(self):
        # 创建2个program
        # 5、定义执行器
        place = fluid.CUDAPlace(0) if self.img_param["useGPU"] else fluid.CPUPlace()
        train_program = fluid.Program()
        start_program = fluid.Program()
        with fluid.program_guard(train_program, start_program):
            # "max_box_num": 20,  # 一幅图上最多有多少个目标
            max_box_num = self.img_param["maxBoxNum"]
            # 1、定义变量
            img = fluid.layers.data(name="img", shape=self.img_param["inputSize"], dtype="float32")
            gt_box = fluid.layers.data(name="gt_box", shape=[max_box_num, 4], dtype="float32")
            gt_label = fluid.layers.data(name="gt_label", shape=[max_box_num], dtype="int32")
            # 2、构建训练模型
            anchors = self.img_param["anchors"]
            anchor_mask = self.img_param["anchorMask"]  # Anchor Box序号
            class_dim = self.img_param["classDim"]  # 初始化的时候设置
            with fluid.unique_name.guard():
                # 创建yolo模型
                model = Yolo3Model(class_dim, anchors, anchor_mask)
                output = model.net(img)
            # 3.计算损失函数，4、定义优化器
            loss = self.get_losses(model, output, gt_box, gt_label)

            # 6、定义数据喂入器
            feeder = fluid.DataFeeder(feed_list=[img, gt_box, gt_label], place=place, program=train_program)
            reader = self.paddle_data_reader()
            reader = paddle.reader.shuffle(reader, buf_size=self.img_param["trainBatchSize"])
            reader = paddle.batch(reader, batch_size=self.img_param["trainBatchSize"])

        # 创建exe,加载增量模型
        exe = fluid.Executor(place)
        exe.run(start_program)
        train_fetch_list = [loss.name]
        # 7、加载增量模型
        self.load_pretrained_params(exe, train_program)
        stop_strategy = ""
        # successive_limit = stop_strategy[""]
        # sample_freq = stop_strategy[""]
        # min_curr_map = stop_strategy[""]
        # min_loss = stop_strategy[""]
        # stop_train = True
        # successive_count = 0
        total_batch_count = 0
        # valid_thresh = ""
        # nms_thresh = ""
        current_best_loss = 10000000000.0

        # 8、开始迭代训练
        for pass_id in range(self.img_param["numEpochs"]):
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
                    print(f"pass:{pass_id};trainbatch:{batch_id};loss:{loss},time:{period}")
            pass_mean_loss = total_loss / batch_id
            print(f"pass:{pass_id} train result,current pass mean loss:{pass_mean_loss}")
            ###模型评估###

            # 采用每训练完一轮停止办法，可以调整为更精细的保存策略
            if pass_mean_loss < current_best_loss:
                print("temp save {} epoch train result,current best pass loss{}".format(pass_id, pass_mean_loss))
                fluid.io.save_persistables(dirname=self.img_param["pretrainedModelDir"],
                                           main_program=train_program,
                                           executor=exe)
                current_best_loss = pass_mean_loss
        print("training till last epoch,end training")
        # 9、保存增量模型
        fluid.io.save_persistables(dirname=self.img_param["saveModelDir"],
                                   main_program=train_program,
                                   executor=exe)
        # 10、保存预测模型
        self.freeze_model()

    # 固化保存模型
    def freeze_model(self):
        """保存模型，用来预测"""
        exe = fluid.Executor(fluid.CPUPlace())
        path = self.img_param["saveModelDir"]
        model = Yolo3Model("class_dim", "anchors", "anchor_mask")
        image = fluid.layers.data(name="image", shape=self.img_param["inputSize"], dtype="float32")
        image_shape = fluid.layers.data(name="image_shape", shape=[2], dtype="int32")
        boxes = []
        scores = []
        outputs = model.net(image)
        down_sample_ratio = model.get_down_sample_ratio()
        for i, out in enumerate(outputs):
            # 生成yolo3模型检测框
            box, score = fluid.layers.yolo_box(x=out,
                                               img_size=image_shape,
                                               anochors=model.get_class_num(),
                                               conf_tresh=self.img_param["validThresh"],
                                               downsample_ratio=down_sample_ratio,
                                               name="yolo_box_" + str(i))
            boxes.append(box)
            # 该OP根据perm对输入的多维Tensor进行数据重排。返回多维Tensor的第i维对应输入Tensor的perm[i]维。
            scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))
            down_sample_ratio //= 2
        # 该OP用于对边界框（bounding box）和评分（scores）执行多类非极大值抑制（NMS）
        pred = fluid.layers.multiclass_nms(bboxes=fluid.layers.concat(boxes, axis=1),
                                           scores=fluid.layers.concat(scores, axis=2),
                                           score_threshold=self.img_param["validThresh"],
                                           nms_top_k=self.img_param["nmsTopK"],
                                           keep_top_k=self.img_param["nmsTopK"],
                                           nms_threshold=self.img_param["nmnThresh"],
                                           background_label=-1,
                                           name="multiclass_nms")
        freeze_program = fluid.default_main_program()
        fluid.io.load_persistables(exe, path, freeze_program)
        freeze_program = freeze_program.clone(for_test=True)
        print("freeze out: {0}, pred layout: {1}".format(['freeze_dir'], pred))

        # 保存模型
        fluid.io.save_inference_model(self.img_param["inferenceModelDir"],
                                      ["image", "image_shape"],
                                      [pred],
                                      exe,
                                      freeze_program)
        print("freeze end")


class ModelPredict(DealImgData):
    def __init__(self):
        super().__init__()
        # self.anchors = ""
        # self.anchor_mask = ""
        # self.label_dict = ""
        # self.class_dim = ""

    def resize_img(self, img):
        """
        保持比例缩放图片
        :param img:
        :return:
        """
        img = img.resize(self.img_param[1:], Image.BILINEAR)
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
            draw.text((xmin, ymin), self.img_param["numDict"][int(label)], (255, 255, 0))  # 绘制标签
        img.save(save_name)
        # display(img)

    def predict_model(self, image_path):
        """预测"""
        origin, tensor_img, resized_img = self.read_image(image_path)
        input_w, input_h = origin.size[0], origin.size[1]
        image_shape = np.array([input_h, input_w], dtype="int32")
        t1 = time.time()
        # 执行预测
        place = fluid.CUDAPlace(0) if self.img_param["useGPU"] else fluid.CPUPlace()
        exe = fluid.Executor(place)
        path = self.img_param["inferenceModelDir"]
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
        self.draw_bbox_image(origin, boxes, labels, out_path)


if __name__ == '__main__':
    train_obj = TrainModel()
    train_obj.train_model()
