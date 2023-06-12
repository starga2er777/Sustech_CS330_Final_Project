import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchsummary import summary
from PIL import Image
import matplotlib.pyplot as plt
from pyecharts.charts import Line
from pyecharts.options import VisualMapOpts, LabelOpts
from pyecharts import options as opts
import os

# 当前文件夹
current_directory = os.path.dirname(__file__)
samples_directory = os.path.join(current_directory, "samples/500X")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# label 标签对应整数，str->int
label_dict = {}


def cut_image(path):
    images = []
    labels = []
    # 获取所有图片
    samples = os.listdir(path)
    for sample in samples:
        sample_label = sample.split('-')[0]
        if sample_label not in label_dict.keys():
            label_dict[sample_label] = len(label_dict)
        sample_path = os.path.join(path, sample)
        # 图片选用500X，2048*2176
        img = Image.open(sample_path)
        # 切割图片，删去底部标签部分 2048*2176 -> 2048*2048
        img = img.crop((0, 0, 2048, 2048))
        # 原图是灰度图像，但灰度只有一个通道，用不了AlexNet，所以转成RGB图
        # img = img.convert('L')
        img = img.convert('RGB')
        # 切割成小图片并保存，切割成256*256，一个样本分成8*8=64个小样本
        for x in range(8):
            for y in range(8):
                temp = img.crop((x << 8, y << 8, (x << 8) + 256, (y << 8) + 256))
                images.append(temp)
                labels.append(label_dict[sample_label])
    print(len(images))
    # images[0].show()
    return images, labels


def random_cut_image(path, count=16):
    images = []
    labels = []
    # 获取所有图片
    samples = os.listdir(path)
    for sample in samples:
        sample_label = sample.split('-')[0]
        sample_path = os.path.join(path, sample)
        # 图片选用500X，2048*2176
        img = Image.open(sample_path)
        # 切割图片，删去底部标签部分 2048*2176 -> 2048*2048
        img = img.crop((0, 0, 2048, 2048))
        # 原图是灰度图像，但灰度只有一个通道，用不了AlexNet，所以转成RGB图
        # img = img.convert('L')
        img = img.convert('RGB')
        # 切割成小图片并保存，切割成256*256，随机切count个
        for i in range(count):
            temp = transforms.RandomCrop((256, 256))(img)
            images.append(temp)
            labels.append(label_dict[sample_label])
    return images, labels


def fit_alexnet_input(images: list):
    return [transforms.RandomCrop((227, 227))(image) for image in images]


def print_chart(x_data, y_data):
    line = Line()
    line.add_xaxis(x_data)
    line.add_yaxis("Loss", y_data, label_opts=LabelOpts(is_show=False), is_smooth=True)
    line.set_global_opts(
        visualmap_opts=VisualMapOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(
            type_="value",
            name="训练次数",
            is_show=True,
            is_scale=True,
            name_location='middle',
            min_=0,
            max_=1664,
        ))
    line.render()


class ImageDataSet(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        else:
            image.resize((256, 256))  # 使用resize
        return image, self.labels[index]


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 3)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    x_data = []
    y_data = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            x_data.append(current)
            y_data.append(loss)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    # print_chart(x_data, y_data)
    plt.plot(x_data, y_data)
    plt.show()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def run():
    model = AlexNet().to(device)
    summary(AlexNet(), (3, 227, 227), 4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # 训练和验证集采用transform做变换
    train_transform = transforms.Compose([
        # transforms.Resize([256, 256]),  # 图片resize
        transforms.ToTensor(),  # 将图像转为Tensor,img.float().div(255)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 测试集采用test_transform做变换
    # 原图片为2048*2048
    test_transform = transforms.Compose([
        # transforms.Resize([256, 256]),  # 把图片resize为256*256
        # transforms.RandomCrop(256),  # 随机裁剪256*256
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.ToTensor(),  # 将图像转为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    images, labels = cut_image(path=samples_directory)
    # 喂给AlexNet的需要227*227，所以每张要再随机截取256*256->227*227
    images = fit_alexnet_input(images)
    train_data = ImageDataSet(images=images, labels=labels, transform=train_transform)

    test_images, test_labels = random_cut_image(path=samples_directory, count=16)
    test_images = fit_alexnet_input(test_images)
    test_data = ImageDataSet(images=test_images, labels=test_labels, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=4, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=4, drop_last=True)

    epochs = 3
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), os.path.join(current_directory, "model.pth"))
    print("Done!")


# run()


if __name__ == '__main__':
    model = AlexNet()
    model.load_state_dict(torch.load('model.pth'))
    model.to(torch.device('cpu'))
    loss_fn = nn.CrossEntropyLoss()
    test_images, test_labels = random_cut_image(path=samples_directory, count=16)
    train_transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转为Tensor,img.float().div(255)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
    test_data = ImageDataSet(images=test_images, labels=test_labels, transform=train_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=4, drop_last=True)
    test(test_loader, model, loss_fn)
