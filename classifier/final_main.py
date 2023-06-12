import tensorflow as tf
from tensorflow.python.keras import models, layers, callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import new_data_management as mydata
# import data_management as mydata


# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 1)))  # 过滤器个数，卷积核尺寸，激活函数，输入形状
# model.add(layers.MaxPooling2D((2, 2)))  # 池化层
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())  # 降维
# model.add(layers.Dense(2401, activation='relu'))  # 全连接层
# model.add(layers.Dense(343, activation='relu'))  # 全连接层
# model.add(layers.Dense(7, activation='softmax'))  # 注意这里参数，我只有两类图片，所以是3.
# model.summary()  # 显示模型的架构
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model = models.Sequential()
# model.add(layers.Conv2D(8, (32, 32), activation='relu', input_shape=(512, 512, 1)))
# model.add(layers.MaxPooling2D((3, 3)))  # 池化层
# model.add(layers.Conv2D(16, (32, 32), activation='relu'))  # 过滤器个数，卷积核尺寸，激活函数，输入形状
# model.add(layers.MaxPooling2D((3, 3)))
# model.add(layers.Conv2D(16, (32, 32), activation='relu'))
# model.add(layers.Flatten())  # 降维
# model.add(layers.Dense(64, activation='relu'))  # 全连接层
# model.add(layers.Dense(7, activation='softmax'))  # 注意这里参数，我只有两类图片，所以是3.
# model.summary()  # 显示模型的架构
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

model = models.Sequential()
model.add(layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(512, 512, 1)))
model.add(layers.MaxPooling2D((3, 3), strides=2))  # 池化层
model.add(layers.Conv2D(256, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((3, 3), strides=2))
model.add(layers.Conv2D(384, (5, 5), activation='relu'))
model.add(layers.Conv2D(384, (5, 5), activation='relu'))
model.add(layers.Conv2D(256, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((3, 3), strides=2))
model.add(layers.Flatten())  # 降维
model.add(layers.Dense(6000, activation='relu'))  # 全连接层
model.add(layers.Dense(2000, activation='relu'))  # 全连接层
model.add(layers.Dense(400, activation='relu'))  # 全连接层
model.add(layers.Dense(7, activation='softmax'))  # 注意这里参数，我只有两类图片，所以是3.
model.summary()  # 显示模型的架构
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 绘图函数
def print_history(history):
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['val_loss'])
    plt.title('Train accuracy&loss')
    plt.xlabel('Epoch')
    plt.legend(['train_acc', 'train_loss', 'val_acc', 'val_loss'])
    plt.show()


def run(model, X, Y):
    # 0.1 用于检验 + 打乱
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.1, stratify=Y)
    # history = model.fit(X_train, Y_train, epochs=8)
    callback = callbacks.EarlyStopping(monitor='loss', min_delta=0.002, patience=5, mode='auto',
                                       restore_best_weights=False)
    # 0.2用于单次检验，由于上面以及打乱过，故此处不用打乱
    history = model.fit(x=X_train, y=Y_train, batch_size=16, epochs=10,
                        validation_split=0.2, callbacks=callback)
    print(history.history.keys())
    print_history(history)
    print("预测中。。。")
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
    print('误差', loss)
    print('准确率', accuracy)


# model1 = tf.keras.models.load_model('my_model.h5')
# X, Y = mydata.load_data()  # return as list
# run(model1, X, Y)
# model1.save('my_model.h5')  # 保存为h5模型
# print("模型保存成功！")

X, Y = mydata.load_data()  # return as list
run(model, X, Y)
model.save('my_model.h5')  # 保存为h5模型
print("模型保存成功！")
