import torch
import torch.nn as nn
from utils.utils import Noise
batch_size = 100

class encode1(torch.nn.Module):
    def __init__(self):
        super(encode1, self).__init__()
        self.noise1 = Noise()
        self.conv1 = nn.Conv2d(1, 32, 3, padding='valid')
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.buffer_pre_z = None #保存于clear encode 阶段的输入
    def forward_clean(self, x):
        self.buffer_pre_z = x.detach().clone()
        z = self.conv1(x)
        z = self.maxpool(z)
        z = self.relu(z)
        return z
    def forward_noise(self, x):
        x1 = self.noise1(x)
        x = self.conv1(x1)
        x = self.maxpool(x)
        x = self.relu(x)
        return x
class encode2(torch.nn.Module):
    def __init__(self):
        super(encode2, self).__init__()
        self.noise2 = Noise()
        self.conv2 = nn.Conv2d(32, 64, 3, padding='valid')
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.buffer_pre_z = None #保存于clear encode 阶段的输入
    def forward_clean(self, x):
        # Store z_pre, z to be used in calculation of reconstruction cost
        self.buffer_pre_z = x.detach().clone()
        z = self.conv2(x)
        z = self.maxpool(z)
        z = self.relu(z)
        return z
    def forward_noise(self, x):
        x = self.noise2(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.relu(x)
        return x
class encode3(torch.nn.Module):
    def __init__(self):
        super(encode3, self).__init__()
        self.noise3 = Noise()
        self.conv3 = nn.Conv2d(64, 64, 3, padding='valid')
        self.buffer_pre_z = None #保存于clear encode 阶段的输入
    def forward_clean(self, x):
        # Store z_pre, z to be used in calculation of reconstruction cost
        self.buffer_pre_z = x.detach().clone()
        z = self.conv3(x)
        return z
    def forward_noise(self, x):
        x = self.noise3(x)
        x = self.conv3(x)
        return x

class encode4(torch.nn.Module):
    def __init__(self):
        super(encode4, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 3 * 3, 10)
        self.buffer_pre_z = None
    def forward_clean(self, x):
        # Store z_pre, z to be used in calculation of reconstruction cost
        x = self.relu(x)
        x = x.view(-1, 64 * 3 * 3)
        x = self.fc(x)
        return x
    def forward_noise(self, x):
        x = self.relu(x)
        x = x.view(-1, 64 * 3 * 3)
        x = self.fc(x)
        return x

class StackedEncoders(torch.nn.Module):
    def __init__(self):
        super(StackedEncoders, self).__init__()
        self.buffer_tilde_z_bottom = None
        self.decode_input = None
        self.encoders_ref = ['encode_1', 'encode_2', 'encode_3','encode_4']
        self.encoders = torch.nn.Sequential()

        encoder1 = encode1()
        encoder2 = encode2()
        encoder3 = encode3()
        encoder4 = encode4()

        self.encoders.add_module('encode_1', encoder1)
        self.encoders.add_module('encode_2', encoder2)
        self.encoders.add_module('encode_3', encoder3)
        self.encoders.add_module('encode_4', encoder4)

    def forward_clean(self, x):
        h = x
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward_clean(h)
        return h

    def forward_noise(self, x):  # 添加了噪声的向前传播过程
        h = x
        # pass through encoders
        for e_ref in self.encoders_ref:
            if(e_ref == 'encode_4'):
                self.decode_input = h
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward_noise(h)
        return h

    def get_encoders_z_pre(self):
        z_pre_layers = []
        for e_ref in self.encoders_ref:
            if(e_ref == 'encode_4'):
                break
            encoder = getattr(self.encoders, e_ref)
            z_pre = encoder.buffer_pre_z.clone()
            z_pre_layers.append(z_pre)
        z_pre_layers.reverse()
        return z_pre_layers