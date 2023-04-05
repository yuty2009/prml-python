import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock(nn.Module):
    def __init__(self, chan_in, hidden_size, chan_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan_in, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, chan_out, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class DiscreteVAE(nn.Module):
    def __init__(
        self,
        image_size = 256,
        in_channels = 3,
        hidden_dim = 64,
        num_tokens = 512,
        codebook_dim = 512,
        num_layers = 3,
        smooth_l1_loss = False,
        temperature = 0.9,
        straight_through = False,
        kl_div_loss_weight = 0.
    ):
        super().__init__()
        # assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        enc_layers = []
        dec_layers = []

        enc_in = in_channels
        dec_in = codebook_dim

        for layer_id in range(num_layers):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, hidden_dim, 4, stride=2, padding=1), nn.ReLU()))
            enc_layers.append(ResBlock(chan_in=hidden_dim, hidden_size=hidden_dim, chan_out=hidden_dim))
            enc_in = hidden_dim
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, hidden_dim, 4, stride=2, padding=1), nn.ReLU()))
            dec_layers.append(ResBlock(chan_in=hidden_dim, hidden_size=hidden_dim, chan_out=hidden_dim))
            dec_in = hidden_dim

        enc_layers.append(nn.Conv2d(hidden_dim, num_tokens, 1))
        dec_layers.append(nn.Conv2d(hidden_dim, in_channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

    def get_image_size(self):
        return self.image_size

    def get_image_tokens_size(self):
        return self.image_size // 8

    @torch.no_grad()
    def get_codebook_indices(self, images):
        logits = self.forward(images, return_logits = True)
        codebook_indices = logits.argmax(dim = 1)
        return codebook_indices

    @torch.no_grad()
    def get_codebook_probs(self, images):
        logits = self.forward(images, return_logits = True)
        return nn.Softmax(dim=1)(logits)

    def decode(self, img_seq):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(math.sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        images = self.decoder(image_embeds)
        return images

    def forward(self, img, temp = None):
        device, num_tokens, image_size, kl_div_loss_weight = \
            img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size[0] and img.shape[-2] == image_size[1], \
            f'input must have the correct image size {image_size}'

        logits = self.encoder(img)

        temp = temp if temp is not None else self.temperature
        soft_one_hot = F.gumbel_softmax(logits, tau = temp, dim = 1, hard = self.straight_through)
        sampled = torch.einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        # reconstruction loss
        loss_recon = self.loss_fn(img, out)

        # kl divergence
        logits = rearrange(logits, 'b n h w -> b (h w) n')
        qy = F.softmax(logits, dim = -1)

        log_qy = torch.log(qy + 1e-10)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device = device))
        loss_kl = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target = True)

        loss = loss_recon + (loss_kl * kl_div_loss_weight)

        return out, loss
    

if __name__ == '__main__':

    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    from engine import train, test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '/Users/yuty2009/data/prmldata/mnist', train=True, download=True,
            transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        ),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '/Users/yuty2009/data/prmldata/mnist', train=False, download=True,
            transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        ),
        batch_size=batch_size, shuffle=False)

    model = DiscreteVAE((32, 32), 1, hidden_dim=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(0, 10):
        train(train_loader, model, optimizer, epoch, device)
        test(test_loader, model, epoch, device)
        with torch.no_grad():
            sample = torch.randint(0, 512, 64).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 32, 32),
                       '.output/dvae/sample_' + str(epoch) + '.png')