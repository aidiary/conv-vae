import argparse
import torch
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

import os
import shutil
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model import ConvVAE


cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = torch.device("cuda" if cuda else "cpu")


def loss_function(recon_x, x, mu, logvar):
    # reconstruction loss
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch, model, train_loader, optimizer, args):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
        data = data.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        loss = loss_function(recon_batch, data, mu, logvar)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)

    return train_loss


def test(epoch, model, test_loader, writer, args):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, _) in tqdm(enumerate(test_loader), total=len(test_loader), desc='test'):
            data = data.to(device)

            recon_batch, mu, logvar = model(data)

            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            if batch_idx == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 28, 28)[:n]]).cpu()
                img = make_grid(comparison)
                writer.add_image('reconstruction', img, epoch)
                save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)

    return test_loss


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


def main():
    parser = argparse.ArgumentParser(description='Convolutional VAE MNIST Example')
    parser.add_argument('--result_dir', type=str, default='results', metavar='DIR',
                        help='output directory')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None')

    # model options
    parser.add_argument('--latent_size', type=int, default=32, metavar='N',
                        help='latent vector size of encoder')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = ConvVAE(args.latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 1
    best_test_loss = np.finfo('f').max

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s (epoch: %d)' % (args.resume, checkpoint['epoch']))
        else:
            print('=> no checkpoint found at %s' % args.resume)

    writer = SummaryWriter()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train(epoch, model, train_loader, optimizer, args)
        test_loss = test(epoch, model, test_loader, writer, args)

        # logging
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('test/loss', test_loss, epoch)

        print('Epoch [%d/%d] loss: %.3f val_loss: %.3f' % (epoch, args.epochs, train_loss, test_loss))

        is_best = test_loss < best_test_loss
        best_test_loss = min(test_loss, best_test_loss)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.results_dir, 'checkpoint.pth'))

        with torch.no_grad():
            sample = torch.randn(64, 32).to(device)
            sample = model.decode(sample).cpu()
            img = make_grid(sample)
            writer.add_image('sampling', img, epoch)
            save_image(sample.view(64, 1, 28, 28), 'results/sample_' + str(epoch) + '.png')


if __name__ == '__main__':
    main()
