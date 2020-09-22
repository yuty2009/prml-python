# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utils.mnistreader import *
from deeplearning.gan.pytorch.dcgan import *

imsize = 28
datapath = 'e:/prmldata/mnist/'
mnist = MNISTReader(datapath=datapath)
trainset = mnist.get_train_dataset(onehot_label=False,
                                   normalize=True, mean_std=(0.5, 0.5),
                                   reshape=True, new_shape=(-1, imsize, imsize, 1),
                                   transpose=True, new_pos=(0, 3, 1, 2))
testset = mnist.get_test_dataset(onehot_label=False,
                                 normalize=True, mean_std=(0.5, 0.5),
                                 reshape=True, new_shape=(-1, imsize, imsize, 1),
                                 transpose=True, new_pos=(0, 3, 1, 2))

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.manual_seed(42)
if cuda:
    torch.cuda.manual_seed(42)

parser = argparse.ArgumentParser()
help_ = "Load model checkpoints"
parser.add_argument("-w", "--weights", help=help_)
args = parser.parse_args()

latent_dim = 100
model = DCGAN(image_shape=(1, imsize, imsize), latent_dim=latent_dim).to(device)
os.makedirs("images", exist_ok=True)

if args.weights:
    print('=> loading checkpoint %s' % args.weights)
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'])
    print('=> loaded checkpoint %s' % args.weights)
else:
    # train
    epochs = 20
    batch_size = 64
    sample_interval = 400
    batches_per_epoch = np.ceil(trainset.num_examples/batch_size).astype('int')
    for epoch in range(epochs):
        for batch in range(batches_per_epoch):
            X_batch, y_batch = trainset.next_batch(batch_size)
            # Adversarial ground truths
            output_real = Tensor(X_batch.shape[0], 1).fill_(1.0)
            output_fake = Tensor(X_batch.shape[0], 1).fill_(0.0)

            images_real = Tensor(X_batch)

            # -----------------
            #  Train Generator
            # -----------------
            z = Tensor(X_batch.shape[0], latent_dim).normal_(0, 1)
            images_fake = model.generator(z)
            predict_fake = model.discriminator(images_fake)
            # Loss measures generator's ability to fool the discriminator
            loss_G = model.loss_adversarial(predict_fake, output_real)

            model.optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            model.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            predict_real = model.discriminator(images_real)
            # Measure discriminator's ability to classify real from generated samples
            loss_real = model.loss_adversarial(predict_real, output_real)
            loss_fake = model.loss_adversarial(predict_fake, output_fake)
            loss_D = (loss_real + loss_fake) / 2

            model.optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            model.optimizer_D.step()

            # print statistics every 100 steps
            if (batch + 1) % 10 == 0:
                print("Epoch [{}/{}], Batch [{}/{}], D Loss {:.4f}, G Loss {:.4f}"
                      .format(epoch, epochs,
                              batch, batches_per_epoch,
                              loss_D.item(), loss_G.item()))

            batches_done = epoch * batches_per_epoch + batch
            if batches_done % sample_interval == 0:
                save_image(images_fake.data[:25], "images/%d.png" %
                           batches_done, nrow=5, normalize=True)
