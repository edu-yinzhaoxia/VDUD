import torch
from torch import nn

# Down sampling module
def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(dim_out),
        nn.LeakyReLU(0.2),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(dim_out),
        nn.LeakyReLU(0.2),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(dim_out),
        nn.ReLU()
    )
# Up sampling module
def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ch_fine),
        nn.Dropout(p=0.5, inplace=False),
        nn.ReLU()
    )
def recover(latent_dim,fc_dim):
    return nn.Sequential(
            nn.Linear(latent_dim, fc_dim),
            nn.ReLU()
        )

class MNIST_Net(nn.Module):
    def __init__(self, useBN=False):
        super(MNIST_Net, self).__init__()

        self.conv1 = add_conv_stage(1, 128)
        self.conv2 = add_conv_stage(128, 256)
        self.conv3 = add_conv_stage(256, 512)
        self.fc_mu = nn.Linear(512*7*7, 512)
        self.fc_log_var = nn.Linear(512*7*7, 512)
        self.fc = recover(512, 512*7*7)
        self.conv3m = add_conv_stage(512, 256)
        self.conv2m = add_conv_stage(256, 128)
        self.conv1m = add_conv_stage(128, 32)

        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample32 = upsample(512, 256)
        self.upsample21 = upsample(256, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.xavier_uniform(m.weight)

    def sample_z(self, mu, log_var):
        """sample z by reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std



    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv_out = torch.flatten(conv3_out, start_dim=1)
        mu = self.fc_mu(conv_out)
        log_var = self.fc_log_var(conv_out)
        z = self.sample_z(mu, log_var)
        gaussian_noise = torch.randn(z.size()).cuda()*0.01
        z1 = z + gaussian_noise
        z2 = self.fc(z1)
        conv_z = z2.reshape(z2.size(0), -1, 7, 7)

        conv3m_out_ = torch.cat((self.upsample32(conv_z), conv2_out), 1)
        conv2m_out = self.conv3m(conv3m_out_)
        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out_ = self.conv2m(conv2m_out_)
        conv1m_out = self.conv1m(conv1m_out_)
        conv0_out = self.conv0(conv1m_out)
        out = torch.clip(conv0_out, min=0, max=1)
        return out, z1, mu, log_var


class CIFAR_Net(nn.Module):
    def __init__(self, useBN=False):
        super(CIFAR_Net, self).__init__()

        self.conv1 = add_conv_stage(3, 32)
        self.conv2 = add_conv_stage(32, 64)
        self.conv3 = add_conv_stage(64, 128)
        self.conv4 = add_conv_stage(128, 256)
        self.conv5 = add_conv_stage(256, 512)
        self.fc_mu = nn.Linear(512*2*2, 1024)
        self.fc_log_var = nn.Linear(512*2*2, 1024)
        self.fc = recover(1024, 512 * 2 * 2)

        self.conv4m = add_conv_stage(512, 256)
        self.conv3m = add_conv_stage(256, 128)
        self.conv2m = add_conv_stage(128, 64)
        self.conv1m = add_conv_stage(64, 32)

        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                if m.bias is not None:
                    nn.init.xavier_uniform(m.weight)

    def sample_z(self, mu, log_var):
        """sample z by reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))
        conv5_out = self.conv5(self.max_pool(conv4_out))
        conv_out = torch.flatten(conv5_out, start_dim=1)
        mu = self.fc_mu(conv_out)
        log_var = self.fc_log_var(conv_out)
        z = self.sample_z(mu, log_var)
        gaussian_noise = torch.randn(z.size()).cuda()*0.1
        z1 = z + gaussian_noise
        z1 = self.fc(z1)
        conv_z = z1.reshape(z1.size(0), -1, 2, 2)
        conv5m_out_ = torch.cat((self.upsample54(conv_z), conv4_out), 1)
        conv4m_out = self.conv4m(conv5m_out_)
        conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
        conv3m_out = self.conv3m(conv4m_out_)
        conv2m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv2m_out_)
        conv1m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv1m_out_)
        out = self.conv0(conv1m_out)
        # out = torch.clip(conv0_out, min=0, max=1)
        return out, z1, mu, log_var