# [1] https://www.spaces.ac.cn/archives/6760
# [2] https://zhuanlan.zhihu.com/p/91434658
# [3] https://zhuanlan.zhihu.com/p/603508656
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class VectorQuantizer(nn.Module):
    def __init__(self, vocab_size, embed_dim, commitment_weight):
        super(VectorQuantizer, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.codebook = nn.Embedding(self.vocab_size, self.embed_dim)
        self.codebook.weight.data.uniform_(-1/self.vocab_size, 1/self.vocab_size)
        self.commitment_weight = commitment_weight

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embed_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.codebook.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.vocab_size, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_weight * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    
    @torch.no_grad()
    def decode(self, img_seq):
        # img_seq: (batch_size, h, w)
        image_embeds = self.codebook(img_seq)
        return image_embeds.permute(0, 3, 1, 2).contiguous()
    

class VectorQuantizerEMA(nn.Module):
    def __init__(self, vocab_size, embed_dim, commitment_weight, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.codebook = nn.Embedding(self.vocab_size, self.embed_dim)
        self.codebook.weight.data.normal_()
        self.commitment_weight = commitment_weight
        
        self.register_buffer('_ema_cluster_size', torch.zeros(vocab_size))
        self._ema_w = nn.Parameter(torch.Tensor(vocab_size, self.embed_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embed_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.codebook.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.vocab_size, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self.vocab_size * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self.codebook.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_weight * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    
    @torch.no_grad()
    def decode(self, img_seq):
        # img_seq: (batch_size, h, w)
        image_embeds = self.codebook(img_seq)
        return image_embeds.permute(0, 3, 1, 2).contiguous()
    

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
    

class VQVAE(nn.Module):
    def __init__(
        self,
        image_size = 256,
        in_channels = 3,
        hidden_dim = 64,
        vocab_size = 512,
        embed_dim = 384,
        num_layers = 3,
        smooth_l1_loss = False,
        loss_kl_weight = 1.,
        commitment_weight = 1.,
        decay = 0
    ):
        super(VQVAE, self).__init__()
        
        # assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'

        self.image_size = image_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

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

        # encoder output dimension is embed_dim, which is different from Discrete VAE (vocab_size)
        enc_layers.append(nn.Conv2d(hidden_dim, embed_dim, 1))
        dec_layers.append(nn.Conv2d(hidden_dim, in_channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        if decay > 0.0:
            self.quantizer = VectorQuantizerEMA(vocab_size, embed_dim, commitment_weight, decay)
        else:
            self.quantizer = VectorQuantizer(vocab_size, embed_dim, commitment_weight)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.loss_kl_weight = loss_kl_weight

    def forward(self, x):
        z = self.encoder(x)
        loss_latent, zq, _, _ = self.quantizer(z)
        x_recon = self.decoder(zq)
        # reconstruction loss
        loss_recon = self.loss_fn(x, x_recon)
        loss = loss_recon + self.loss_kl_weight * loss_latent
        return loss, x_recon
    
    def decode(self, img_seq):
        image_embeds = self.quantizer.decode(img_seq)
        images = self.decoder(image_embeds)
        return images
    
    def get_image_size(self):
        return self.image_size

    def get_image_tokens_size(self):
        return self.image_size // (2 ** self.num_layers)
    

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
    model = VQVAE((image_size, image_size), in_channels, vocab_size=vocab_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(0, 10):
        train(train_loader, model, optimizer, epoch, device)
        test(test_loader, model, epoch, device)
        with torch.no_grad():
            sample = torch.randint(0, vocab_size, (64, token_size, token_size)).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, image_size, image_size),
                       '.output/vqvae/sample_' + str(epoch) + '.png')
            