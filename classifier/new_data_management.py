import os
import numpy as np
import pandas as pd
from PIL import Image


def cut_image(img, o, c):  # o指原图片的尺寸，正方形图片的边长；c指边长分为几份，所以输出应该为c*c张小图片
    images = []
    length = int(o/4)
    bits = 0
    while (1 << bits) < length:
        bits += 1
    for x in range(c):
        for y in range(c):
            temp = img.crop((x << bits, y << bits, (x << bits) + length, (y << bits) + length))
            Core = temp.getdata()  # 取数值
            img_data = np.array(Core, dtype='float32') / 255.0  # 标准化
            img_data = img_data.tolist()
            # images.extend(img_data)
            images.append(img_data)
    # images = np.array(images).reshape((c*c, length, length, 1))
    return images


# get label
def get_label():
    classes = {}
    current_directory = os.path.dirname(__file__)
    label_path = os.path.join(current_directory, "label.txt")
    df_label = pd.read_csv(label_path, sep=",")
    df_label['label_num'] = df_label['label_num'].astype(int)
    samples = df_label['sample'].values
    label_num = df_label['label_num'].values
    for i in range(len(samples)):
        classes[samples[i]] = label_num[i]
    # for sample, num in df_label[['sample', 'label_num']]:
    #     classes[sample] = num
    print(f'总共有{len(classes.keys())}个classes：{classes.keys()}')
    return classes


# load data
def load_data(o=2048, c=4):
    classes = get_label()
    current_directory = os.path.dirname(__file__)
    data_path = os.path.join(current_directory, "sample_1")
    X, Y = [], []
    count, detail_count = 0, 0
    length = int(o / c)
    bits = 0
    while (1 << bits) < length:
        bits += 1
    for root, dirs, files in os.walk(data_path):
        if count >= 10: break
        if len(files) != 0:
            for file in files:
                X_class = os.path.basename(root)
                # Y.append(classes[X_class])
                sample_path = os.path.join(root, file)
                sample = Image.open(sample_path)
                sample = sample.crop((0, 0, 2048, 2048))
                sample = sample.convert('L')
                for x in range(c):
                    for y in range(c):
                        temp = sample.crop((x << bits, y << bits, (x << bits) + length, (y << bits) + length))
                        Core = temp.getdata()  # 取数值
                        img_data = np.array(Core, dtype='float32') / 255.0  # 标准化
                        img_data = img_data.tolist()
                        X.extend(img_data)
                        Y.append(classes[X_class])
                        detail_count += 1
                count += 1
                print(count)
                if count == 10: break
    X = np.array(X).reshape((len(Y), length, length, 1))
    Y = np.array(Y)
    print(f'总共有{count}个samples')
    print(f'切割后总共有{detail_count}个samples')
    print(X.shape)
    return X, Y


def my_transform(X, Y, o=2048, c=4):
    # 1. 切成小patch
    new_X, new_Y = [], []
    for i in range(len(X)):
        sample = Image.open(X[i])
        sample = sample.crop((0, 0, 2048, 2048))
        sample = sample.convert('L')
        label = Y[i]
        new_X.extend(cut_image(sample, o, c))
        new_Y.extend([label for _ in range(c*c)])
        print(i)
    length = int(o/4)
    new_X = np.array(new_X).reshape((len(new_Y), length, length, 1))
    new_Y = np.array(new_Y)
    new_X = pd.DataFrame(new_X)
    new_Y = pd.DataFrame(new_Y)
    return new_X, new_Y


def load_data_path():
    classes = get_label()
    current_directory = os.path.dirname(__file__)
    data_path = os.path.join(current_directory, "sample")
    X, Y = [], []
    count = 0
    for root, dirs, files in os.walk(data_path):
        if len(files) != 0:
            for file in files:
                X_class = os.path.basename(root)
                Y.append(classes[X_class])
                sample_path = os.path.join(root, file)
                X.append(sample_path)
                count += 1
    print(f'总共有{count}个samples')
    return np.array(X), np.array(Y)
