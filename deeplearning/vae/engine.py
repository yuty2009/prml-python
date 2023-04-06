import torch
from torchvision.utils import save_image

def train(train_loader, model, optimizer, epoch, device):
    model.train()
    train_loss = 0
    for batch_idx, (data_batch, _) in enumerate(train_loader):
        data_batch = data_batch.to(device)
        optimizer.zero_grad()
        loss, data_recon = model(data_batch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_batch), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data_batch)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(test_loader, model, epoch, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data_batch, _) in enumerate(test_loader):
            batch_size = data_batch.size(0)
            data_batch = data_batch.to(device)
            loss, data_recon = model(data_batch)
            test_loss += loss.item()
            if i == 0:
                n = min(data_batch.size(0), 8)
                comparison = torch.cat([data_batch[:n],
                                      data_recon.view(data_batch.size())[:n]])
                save_image(comparison.cpu(),
                         '.output/vae/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))