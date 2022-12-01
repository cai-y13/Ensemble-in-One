import torch
import torch.nn as nn
from augment_op import RGB

class vgg16(nn.Module):
    def __init__(self, num_classes=10):
        super(vgg16, self).__init__()
        self.conv1 = RGB(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(64))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = RGB(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(64))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        self.conv3 = RGB(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(128))    
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = RGB(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(128))
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = RGB(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(256))
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = RGB(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(256))
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = RGB(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(256))
        self.relu7 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = RGB(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(512))
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = RGB(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(512))
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = RGB(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(512))
        self.relu10 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = RGB(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(512))
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = RGB(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(512))
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13 = RGB(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))#, nn.BatchNorm2d(512))
        self.relu13 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.maxpool1(x)

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.maxpool2(x)

        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        x = self.maxpool3(x)

        x = self.relu8(self.conv8(x))
        x = self.relu9(self.conv9(x))
        x = self.relu10(self.conv10(x))
        x = self.maxpool4(x)

        x = self.relu11(self.conv11(x))
        x = self.relu12(self.conv12(x))
        x = self.relu13(self.conv13(x))
        x = self.maxpool5(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_features(self, x, layer, before_relu=False):
        x = self.relu1(self.conv1(x))
        if layer == 1:
            return x
        x = self.relu2(self.conv2(x))
        if layer == 2:
            return x
        x = self.maxpool1(x)

        x = self.relu3(self.conv3(x))
        if layer == 3:
            return x
        x = self.relu4(self.conv4(x))
        if layer == 4:
            return x
        x = self.maxpool2(x)

        x = self.relu5(self.conv5(x))
        if layer == 5:
            return x
        x = self.relu6(self.conv6(x))
        if layer == 6:
            return x
        x = self.relu7(self.conv7(x))
        if layer == 7:
            return x
        x = self.maxpool3(x)

        x = self.relu8(self.conv8(x))
        if layer == 8:
            return x
        x = self.relu9(self.conv9(x))
        if layer == 9:
            return x
        x = self.relu10(self.conv10(x))
        if layer == 10:
            return x
        x = self.maxpool4(x)

        x = self.relu11(self.conv11(x))
        if layer == 11:
            return x
        x = self.relu12(self.conv12(x))
        if layer == 12:
            return x
        x = self.relu13(self.conv13(x))
        if layer == 13:
            return x
        x = self.maxpool5(x)
        if layer == 14:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        if layer > 14:
            raise ValueError('layer {:d} is out of index!'.format(layer))

