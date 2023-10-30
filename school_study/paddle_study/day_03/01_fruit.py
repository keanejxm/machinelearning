# 行为识别
# 使用cnn实现水果分类
# 如何判断卷积核的大小核数量
    #   首选小卷积（3*3），小卷积核参数规模更小
    # 卷积核数量：根据经验值设定（或实验），选择大小适中
# ，池化层的大小核数量，批次的选择，随机读取时多少次打乱一次如何选择

"""
如何优化图像分类模型
1.数据优化：
    数据量大，特征丰富（避免特征单一）
    主体图像特征要明显
    训练阶段：适当降低图像质量，预测阶段提高图像质量
    数据增强，增加样本数量，特征多样性（旋转、裁剪、灰度化、色彩变换）
    图像大小适中，不能过大、过小
    在图像中主动植入噪声
2.模型优化
    模型选择总原则：问题复杂度和模型复杂度匹配
            复杂的问题选择复杂模型，简单问题选择简单模型
            模型复杂读超过问题复杂性，容易造成过拟合
            问题复杂性超过模型复杂读，容易造成欠你和
    图像分类可选模型：AlexnET,VGG,GoogLeNet.ResNet
3.参数优化
    模型结构相关的层数：卷积层数、卷积核数量、神经元数量
    训练相关参数：学习率（一般采用先大后小）、丢弃率（适中）、训练轮次（轮次过多也会造成过拟合）、批次大小（影响稳定性）

4.外部优化
    减少干扰因素
        通过外部条件改善图像质量，减少遮挡、减少模糊

"""




# 增量训练模型
import matplotlib.pyplot as plt
import paddle
import os
import paddle.fluid as fluid
from multiprocessing import cpu_count
from PIL import Image

# 数据预处理
name_dict = {
    "apple": 0,
    "banana": 1,
    "grape": 2,
    "orange": 3,
    "pear": 4

}
# 模型的定义于训练
data_root_path = "dataset/fruits/"
# 测试
test_file = os.path.join(data_root_path, "test.txt")
train_file = os.path.join(data_root_path, "train.txt")

name_data_list = {}


def save_to_dict(path, name):
    if name not in name_data_list:
        img_list = list()
        img_list.append(path)
        name_data_list[name] = img_list
    else:
        name_data_list[name].append(path)


# 遍历路径
sub_dir = os.listdir(data_root_path)

# d:文件加名称：水果名称
for d in sub_dir:
    # 水果路径
    full_path = os.path.join(data_root_path, d)
    if os.path.isdir(full_path):
        # 图片名称fn
        for fn in os.listdir(full_path):
            # 图片路径
            img_full_path = os.path.join(full_path, fn)
            save_to_dict(img_full_path, d)
    else:
        pass

# 清空测试集，训练集
with open(train_file, "w") as f1:
    pass
with open(test_file, "w") as f2:
    pass

# 遍历数据集
for name, img_list in name_data_list.items():
    i = 0
    num = len(img_list)
    print("%s: %d张" % (name, num))
    for img in img_list:
        line = "%s\t%d\n" % (img, name_dict[name])
        if i % 10 == 0:
            with open(test_file, "a") as a1:
                a1.write(line)
        else:
            with open(train_file, "a") as a2:
                a2.write(line)
        i += 1
print("划分训练集测试集已结束")


def train_mapper(sample):
    img_path, label = sample
    # 加载图片
    img = paddle.dataset.image.load_image(img_path)
    # 缩放图片
    img = paddle.dataset.image.simple_transform(
        im=img,
        resize_size=128,  # 缩放成128
        crop_size=128,  # 裁剪
        is_color=True,
        is_train=True
    )
    img = img.astype("float32") / 255.0
    return img, label


# 读取文件路径
def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            for ln in lines:
                ln = ln.replace("\n", "")
                img_path, label = ln.split("\t")
                yield img_path, int(label)

    return paddle.reader.xmap_readers(
        train_mapper,
        reader,
        cpu_count(),
        buffered_size
    )


def create_cnn(image, type_size):
    """
    创建cnn
    :param image:输入的归一化后的图像
    :param type_size: 类别数量
    :return:
    """
    # 第一组卷积池化
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=image,
        filter_size=3,  # 卷积和大小
        num_filters=32,  # 卷积和数量
        pool_size=2,  # 池化大小
        pool_stride=2,  # 池化步长
        act='relu'
    )
    drop = fluid.layers.dropout(x=conv_pool_1, dropout_prob=0.5)

    # 第二组卷积池化
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=drop,
        filter_size=3,  # 卷积和大小
        num_filters=64,  # 卷积和数量
        pool_size=2,  # 池化大小
        pool_stride=2,  # 池化步长
        act='relu'
    )
    drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=0.5)

    # 第三组卷积池化
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=drop,
        filter_size=3,  # 卷积和大小
        num_filters=64,  # 卷积和数量
        pool_size=2,  # 池化大小
        pool_stride=2,  # 池化步长
        act='relu'
    )
    drop = fluid.layers.dropout(x=conv_pool_3, dropout_prob=0.5)

    # 全连接层
    fc = fluid.layers.fc(drop, size=512, act='relu')
    drop = fluid.layers.dropout(x=fc, dropout_prob=0.5)
    # 输出层
    predict = fluid.layers.fc(input=drop,
                              size=type_size,
                              act="softmax")
    return predict


#
BATCH_SIZE = 32

train_reader = train_r(train_list=train_file)
random_train_reader = paddle.reader.shuffle(reader=train_reader, buf_size=1300)

batch_reader = paddle.batch(random_train_reader, batch_size=BATCH_SIZE)

# 张量
image = fluid.layers.data(name="image", shape=[3, 128, 128], dtype="float32")

# 特征
label = fluid.layers.data(name="label", shape=[1], dtype="int64")

#
predict = create_cnn(image, type_size=5)

# 损失函数（交叉熵）、
cost = fluid.layers.cross_entropy(input=predict,
                                  label=label)
avg_cost = fluid.layers.mean(cost)

# 优化器
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)

# 正确率
accuracy = fluid.layers.accuracy(input=predict, label=label)
# 执行

place = fluid.CPUPlace()
# place = fluid.CUDAPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

feeder = fluid.DataFeeder(
    feed_list=[image, label],
    place=place)

costs = []
accs = []
times = 0
batchs = []
for epoch in range(20):
    for batch_id, data in enumerate(batch_reader()):
        times += 1
        c, a = exe.run(program=fluid.default_main_program(),
                       feed=feeder.feed(data),
                       fetch_list=[avg_cost, accuracy])
        if batch_id % 10 == 0:
            print("epoch:%d,batch:%d,cost:%.4f,acc:%.4f" % (epoch, batch_id, c[0], a[0]))
        accs.append(a[0])
        costs.append(c[0])
        batchs.append(times)

# 保存模型
model_save_path = "../model/fruits/"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
fluid.io.save_inference_model(dirname=model_save_path,
                              feeded_var_names=["image"],
                              target_vars=[predict],
                              executor=exe)

# 训练可视化
plt.title("training", fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("cost/acc", fontsize=20)
plt.plot(batchs, costs, color='red', label="training cost")
plt.plot(batchs, accs, color="green", label="training acc")
plt.legend()
plt.grid()
plt.savefig("train.png")
plt.show()


def load_img(path):
    """"""
    img = paddle.dataset.image.load_and_transform(
        path,
        128,
        128,
        False
    )
    img = img.astype("float32") / 255.0
    return img


# 定义执行器
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)

# 加载模型
# 模型地址
# 加载模型
infer_prog, feed_var, fetch_targets = \
    fluid.io.load_inference_model(model_save_path, infer_exe)

# 加载测试数据
test_img = "/home/tarena/keane_project/paddle_study/day_03/dataset/1.jpg"
infer_imgs = []
infer_imgs.append(load_img(test_img))
#
import numpy as np

infer_imgs = np.array(infer_imgs)

#
params = {feed_var[0]: infer_imgs}
results = infer_exe.run(
    program=infer_prog,
    feed=params,
    fetch_list=fetch_targets
)
# 加深模型，增加数据量

# 预测
# 获取结果概率最大的索引
r = np.argmax(results[0][0])
for k, v in name_dict.items():
    if v == r:
        print(f"预测结果为：{k}")
