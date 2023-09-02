import torch
import torch.nn as nn


class Decode1(torch.nn.Module):
    def __init__(self):
        super(Decode1, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=0)
        # buffer for hat_z_l to be used for cost calculation
        self.buffer_hat_z_l = None
    def forward(self, x):
        hat_z_l = self.deconv1(x)
        self.buffer_hat_z_l = hat_z_l
        return hat_z_l

class Decode2(torch.nn.Module):
    def __init__(self):
        super(Decode2, self).__init__()
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=1, padding=0)
        self.upsample1 = nn.UpsamplingBilinear2d(size=(11, 11))
        # buffer for hat_z_l to be used for cost calculation
        self.buffer_hat_z_l = None
    def forward(self, x):
        x = self.upsample1(x)
        z_l = self.deconv2(x)

        self.buffer_hat_z_l = z_l
        return z_l

class Decode3(torch.nn.Module):
    def __init__(self):
        super(Decode3, self).__init__()
        self.deconv3 = nn.ConvTranspose2d(32, 1, 3, stride=1, padding=0)
        self.upsample2 = nn.UpsamplingBilinear2d(size=(26,26))
        # buffer for hat_z_l to be used for cost calculation
        self.buffer_hat_z_l = None

    def forward(self, x):
        x = self.upsample2(x)
        z_l = self.deconv3(x)

        self.buffer_hat_z_l = z_l
        return z_l

class StackedDecoders(torch.nn.Module):
    def __init__(self):
        super(StackedDecoders, self).__init__()
        self.decoders_ref = ['decoder_1', 'decoder_2', 'decoder_3']
        self.decoders = torch.nn.Sequential()

        decoder1 = Decode1()
        decoder2 = Decode2()
        decoder3 = Decode3()

        self.decoders.add_module('decoder_1', decoder1)
        self.decoders.add_module('decoder_2', decoder2)
        self.decoders.add_module('decoder_3', decoder3)

    def forward(self, x):
        # Note that tilde_z_layers should be in reversed order of encoders
        hat_z = []
        for i in range(len(self.decoders_ref)):
            d_ref = self.decoders_ref[i]
            decoder = getattr(self.decoders, d_ref)
            x = decoder.forward(x)
            hat_z.append(decoder.buffer_hat_z_l)
        return hat_z