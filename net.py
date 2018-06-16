import sys
import chainer
import chainer.functions as F
import chainer.links as L

class AutoEncoder(chainer.Chain):

    def __init__(self, n_hidden=100, n_linear_dim=36784):
        super(AutoEncoder, self).__init__()
        with self.init_scope():
            # encoder
            self.e_conv1 = L.Convolution2D(3, 8, (3, 3))
            self.e_conv2 = L.Convolution2D(8, 4, (3, 3))
            self.e_conv3 = L.Convolution2D(4, 2, (3, 3))
            self.e_conv4 = L.Convolution2D(2, 1, (3, 3))
            self.e_bn1 = L.BatchNormalization(8)
            self.e_bn2 = L.BatchNormalization(4)
            self.e_bn3 = L.BatchNormalization(2)
            self.e_l = L.Linear(n_linear_dim, n_hidden)
            # decoder
            self.d_dconv1 = L.Deconvolution2D(None, 2, (3, 3))
            self.d_dconv2 = L.Deconvolution2D(2, 4, (3, 3))
            self.d_dconv3 = L.Deconvolution2D(4, 8, (3, 3))
            self.d_dconv4 = L.Deconvolution2D(8, 3, (3, 3))
            self.d_bn1 = L.BatchNormalization(2)
            self.d_bn2 = L.BatchNormalization(4)
            self.d_bn3 = L.BatchNormalization(8)
            self.d_l = L.Linear(n_hidden, n_linear_dim)

            self.first_forward = True

    def __call__(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return y

    def encode(self, x):
        h = F.relu(self.e_bn1(self.e_conv1(x)))
        h = F.relu(self.e_bn2(self.e_conv2(h)))
        h = F.relu(self.e_bn3(self.e_conv3(h)))
        h = F.relu((self.e_conv4(h)))
        if self.first_forward:
            self.shape = h.shape
            self.first_forward = False
        h = self.e_l(h)
        return h

    def decode(self, z):
        h = F.relu(self.d_l(z))
        h = F.reshape(h, self.shape)
        h = F.relu(self.d_bn1(self.d_dconv1(h)))
        h = F.relu(self.d_bn2(self.d_dconv2(h)))
        h = F.relu(self.d_bn3(self.d_dconv3(h)))
        h = self.d_dconv4(h)
        return h
