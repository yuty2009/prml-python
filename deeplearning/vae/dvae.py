
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, out_channels, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class DiscreteVAE(nn.Module):
    def __init__(
        self,
        image_size = 256,
        in_channels = 3,
        hidden_dim = 64,
        vocab_size = 512,
        embed_dim = 384,
        num_layers = 3,
        smooth_l1_loss = False,
        temperature = 0.9,
        straight_through = False,
        loss_kl_weight = 0.
    ):
        super().__init__()
        # assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'

        self.image_size = image_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(vocab_size, embed_dim)

        enc_layers = []
        dec_layers = []

        enc_in = in_channels
        dec_in = embed_dim

        for layer_id in range(num_layers):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, hidden_dim, 4, stride=2, padding=1), nn.ReLU()))
            enc_layers.append(ResBlock(in_channels=hidden_dim, out_channels=hidden_dim, hidden_size=hidden_dim))
            enc_in = hidden_dim
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, hidden_dim, 4, stride=2, padding=1), nn.ReLU()))
            dec_layers.append(ResBlock(in_channels=hidden_dim, out_channels=hidden_dim, hidden_size=hidden_dim))
            dec_in = hidden_dim

        enc_layers.append(nn.Conv2d(hidden_dim, vocab_size, 1))
        dec_layers.append(nn.Conv2d(hidden_dim, in_channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.loss_kl_weight = loss_kl_weight

    def quantize(self, z, temp = None):
        # straight through estimator
        # z: (batch_size, vocab_size, h, w) one-hot encoded
        # zq: (batch_size, embed_dim, h, w)
        temp = temp if temp is not None else self.temperature
        soft_one_hot = F.gumbel_softmax(z, tau = temp, dim = 1, hard = self.straight_through)
        zq = torch.einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)

        # kl divergence
        z = z.permute(0, 2, 3, 1).contiguous().view(-1, self.vocab_size)
        qz = F.softmax(z, dim = -1)
        # calculate kl loss
        log_qz = torch.log(qz + 1e-10)
        log_uniform = torch.log(torch.tensor([1. / self.vocab_size], device = z.device))
        loss_kl = F.kl_div(log_uniform, log_qz, None, None, 'batchmean', log_target = True)
        return loss_kl, zq
    
    def forward(self, x, temp = None):
        z = self.encoder(x)
        loss_kl, zq = self.quantize(z, temp)
        x_recon = self.decoder(zq)
        # reconstruction loss
        loss_recon = self.loss_fn(x, x_recon)
        loss = loss_recon + self.loss_kl_weight * loss_kl
        return loss, x_recon
    
    def decode(self, img_seq):
        # img_seq: (batch_size, h, w)
        image_embeds = self.codebook(img_seq).permute(0, 3, 1, 2)
        images = self.decoder(image_embeds)
        return images
    
    def get_image_size(self):
        return self.image_size

    def get_image_tokens_size(self):
        return self.image_size // (2 ** self.num_layers)

    @torch.no_grad()
    def get_codebook_indices(self, images):
        logits = self.encoder(images)
        codebook_indices = logits.argmax(dim = 1)
        return codebook_indices

    @torch.no_grad()
    def get_codebook_probs(self, images):
        logits = self.encoder(images)
        return nn.Softmax(dim=1)(logits)
    

if __name__ == '__main__':

    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    from engine import train, test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = 32
    in_channels = 1
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '/Users/yuty2009/data/prmldata/mnist', train=True, download=True,
            transform=transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        ),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '/Users/yuty2009/data/prmldata/mnist', train=False, download=True,
            transform=transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        ),
        batch_size=batch_size, shuffle=False)

    num_layers = 3
    token_size = image_size // (2 ** num_layers)
    vocab_size = 512
    model = DiscreteVAE((image_size, image_size), in_channels, vocab_size=vocab_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(0, 10):
        train(train_loader, model, optimizer, epoch, device)
        test(test_loader, model, epoch, device)
        with torch.no_grad():
            sample = torch.randint(0, vocab_size, (64, token_size, token_size)).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, image_size, image_size),
                       '.output/dvae/sample_' + str(epoch) + '.png')
            