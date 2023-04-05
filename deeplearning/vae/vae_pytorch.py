# -*- coding: utf-8 -*-
'''
# Reference
[1] Kingma, Diederik P., and Max Welling, "Auto-Encoding Variational Bayes.", 
    https://arxiv.org/abs/1312.6114
[2] https://github.com/pytorch/examples/blob/main/vae/main.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim):
        super(VAE, self).__init__()
        # encoder
        self.dense1 = nn.Linear(x_dim, h_dim)
        self.dense2 = nn.Linear(h_dim, z_dim)
        self.dense3 = nn.Linear(h_dim, z_dim)
        # decoder
        self.dense4 = nn.Linear(z_dim, h_dim)
        self.dense5 = nn.Linear(h_dim, x_dim)

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z_sampled = self.reparameterize(z_mean, z_log_var)
        x_decoded = self.decode(z_sampled)
        loss, _, _ = self.loss(x, x_decoded, z_mean, z_log_var)
        return x_decoded, loss

    def encode(self, x):
        x = x.view(-1, x.size(-2)*x.size(-1))
        h = F.relu(self.dense1(x))
        z_mean = self.dense2(h)
        z_log_var = self.dense3(h)  # estimate log(sigma**2) actually
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.exp(0.5 * z_log_var)
        return z_mean + epsilon * torch.randn_like(epsilon)

    def decode(self, z):
        h = F.relu(self.dense4(z))
        x_decoded = F.sigmoid(self.dense5(h))
        return x_decoded

    def loss(self, inputs, outputs, z_mean, z_log_var):
        inputs = inputs.view(-1, inputs.size(-2)*inputs.size(-1))
        loss_recon = F.binary_cross_entropy(outputs, inputs, reduction='sum')
        loss_kl = 0.5 * torch.sum(torch.exp(z_log_var) + torch.pow(z_mean, 2)
                                  - 1. - z_log_var)
        loss_vae = loss_recon + loss_kl
        return loss_vae, loss_recon, loss_kl
    

class ConvVAE(nn.Module):

    def __init__(self, image_size, in_channels, h_dim, z_dim):
        super(ConvVAE, self).__init__()
        self.image_size = image_size
        # encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.recon_shape = (32, image_size[0] // 4, image_size[1] // 4)
        x_dim = torch.prod(torch.tensor(self.recon_shape))
        self.dense1 = nn.Linear(x_dim, h_dim)
        self.dense2 = nn.Linear(h_dim, z_dim)
        self.dense3 = nn.Linear(h_dim, z_dim)
        # decoder
        self.dense4 = nn.Linear(z_dim, h_dim)
        self.dense5 = nn.Linear(h_dim, x_dim)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid())

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z_sampled = self.reparameterize(z_mean, z_log_var)
        x_decoded = self.decode(z_sampled)
        loss, _, _ = self.loss(x, x_decoded, z_mean, z_log_var)
        return x_decoded, loss

    def encode(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        h = F.relu(self.dense1(x))
        z_mean = self.dense2(h)
        z_log_var = self.dense3(h)  # estimate log(sigma**2) actually
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.exp(0.5 * z_log_var)
        return z_mean + epsilon * torch.randn_like(epsilon)

    def decode(self, z):
        h = F.relu(self.dense4(z))
        x_conv = F.relu(self.dense5(h))
        x_conv = x_conv.view(x_conv.size(0), *self.recon_shape)
        x_decoded = self.deconv1(x_conv)
        return x_decoded

    def loss(self, inputs, outputs, z_mean, z_log_var):
        inputs = inputs.view(inputs.size(0), -1)
        outputs = outputs.view(outputs.size(0), -1)
        loss_recon = F.binary_cross_entropy(outputs, inputs, reduction='sum')
        loss_kl = 0.5 * torch.sum(torch.exp(z_log_var) + torch.pow(z_mean, 2)
                                  - 1. - z_log_var)
        loss_vae = loss_recon + loss_kl
        return loss_vae, loss_recon, loss_kl


if __name__ == '__main__':

    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    from engine import train, test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/Users/yuty2009/data/prmldata/mnist', train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/Users/yuty2009/data/prmldata/mnist', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=False)

    model = ConvVAE((28, 28), 1, 400, 20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(0, 10):
        train(train_loader, model, optimizer, epoch, device)
        test(test_loader, model, epoch, device)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       '.output/vae/sample_' + str(epoch) + '.png')
            