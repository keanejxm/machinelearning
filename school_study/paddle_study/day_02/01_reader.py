import paddle


def reader_creator(file_path):
    def reader():
        with open(file_path, "r") as f:
            for line in f.readlines():
                yield line.strip()

    return reader


# 顺序
reader = reader_creator("test.txt")
# 随机
shuffle_reader = paddle.reader.shuffle(reader, 10)
# 批次
batch_reader = paddle.batch(shuffle_reader, 3)
# for data in reader():
# for data in shuffle_reader():
for data in batch_reader():
    print(data)
