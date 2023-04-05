"""
Reference:
Oord, Aaron van den, Yazhe Li, and Oriol Vinyals.
"Representation Learning with Contrastive Predictive Coding."
arXiv preprint arXiv:1807.03748 (2018).

Code ref: https://github.com/davidtellez/contrastive-predictive-coding
"""
import torch
import torch.nn as nn


class CNNEncoder(nn.Sequential):
    def __init__(self, in_shape=(64, 64), in_channels=3, out_channels=64, out_dim=128):
        flatten_dim = out_channels * torch.prod(torch.tensor(in_shape)) // (16 * 16)
        super(CNNEncoder, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, out_dim),
        )


class MultiPrediction(nn.Module):
    def __init__(self, input_dim, embed_dim, out_seqlen):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.out_seqlen = out_seqlen
        self.embedding = nn.Linear(input_dim, embed_dim)
    def forward(self, x):
        outputs = []
        for i in range(self.out_seqlen):
            outputs.append(self.embedding(x))
        if len(outputs) == 1:
            output = torch.unsqueeze(x, dim=1)
        else:
            output = torch.stack(outputs, dim=1)
        return output


class CPCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = torch.mean(y_encoded * preds, axis=-1)
        dot_product = torch.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension
        # Keras loss functions take probabilities
        dot_product_probs = torch.sigmoid(dot_product)
        return dot_product_probs


class CPC(nn.Module):
    """
    Build a contrastive predictive coding model.
    """
    def __init__(self, encoder, encoder_dim=128, context_dim=256, in_seqlen=4, out_seqlen=4):
        """
        encoder: encoder you want to use to get feature representations (eg. resnet50)
        encoder_dim: dimension of the encoder output, your feature dimension (default: 2048 for resnets)
        feature_dim: dimension of the projector output (default: 2048)
        dim: hidden dimension of the predictor (default: 512)
        """
        super(CPC, self).__init__()

        self.in_seqlen = in_seqlen
        self.out_seqlen = out_seqlen

        self.encoder = encoder
        self.ar_context = nn.GRU(encoder_dim, context_dim, batch_first=True)
        self.prediction = MultiPrediction(context_dim, encoder_dim, out_seqlen)

    def forward(self, inputs):
        x_input, y_input = inputs

        x_encoded = []
        for i in range(self.in_seqlen):
            x_1 = x_input[:, i]
            x_encoded.append(self.encoder(x_1))
        x_encoded = torch.stack(x_encoded, dim=1)

        y_encoded = []
        for i in range(self.out_seqlen):
            y_1 = y_input[:, i]
            y_encoded.append(self.encoder(y_1))
        y_encoded = torch.stack(y_encoded, dim=1)

        context = self.ar_context(x_encoded)[0][:,-1,:]
        preds = self.prediction(context)

        return preds, y_encoded


if __name__ == "__main__":

    x = torch.rand((5, 4, 3, 64, 64))
    y = torch.rand((5, 4, 3, 64, 64))
    encoder = CNNEncoder()
    model = CPC(encoder)
    criterion = CPCLoss()
    preds, y_encoded = model((x, y))
    loss = criterion((preds, y_encoded))
