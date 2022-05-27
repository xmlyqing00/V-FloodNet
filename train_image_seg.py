import os
import argparse
import time
import torch
from torch.utils import model_zoo, data

from WaterSeg import WaterNetV1, WaterDataset_RGB, myutils


def train_WaterNet():
    # Dataset

    dataset = WaterDataset_RGB(
        mode='train_offline',
        dataset_path=args.dataset,
        input_size=(480, 480)
    )
    train_loader = data.DataLoader(dataset, batch_size=10, shuffle=True, pin_memory=True, num_workers=4)

    # Model
    water_net_v1 = WaterNetV1().to(device)

    # Optimizor
    optimizer = torch.optim.SGD(
        params=water_net_v1.parameters(),
        lr=args.lr,
        momentum=0.9,
        dampening=1e-4
    )

    # Load pretrained model
    start_epoch = 0
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print('Load checkpoint \'{}\''.format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            water_net_v1.load_state_dict(checkpoint['water_net_v1'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Loaded checkpoint \'{}\' (epoch {})'
                  .format(args.checkpoint, checkpoint['epoch']))
        else:
            print('No checkpoint found at \'{}\''.format(args.checkpoint))
    else:
        print('Load pretrained ResNet 34.')
        resnet34_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        # resnet50_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        pretrained_model = model_zoo.load_url(resnet34_url)
        water_net_v1.load_pretrained_model(pretrained_model)

    # Set train mode
    water_net_v1.train()

    # Criterion
    criterion = torch.nn.BCELoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    epoch_endtime = time.time()
    saved_models = 'WaterSegModels'
    if not os.path.exists(saved_models):
        os.mkdir(saved_models)

    epoch_time = myutils.AvgMeter()

    training_mode = 'Offline'
    best_loss = 100000

    for epoch in range(start_epoch, args.epochs):

        losses = myutils.AvgMeter()
        batch_time = myutils.AvgMeter()
        batch_endtime = time.time()

        lr = scheduler.get_last_lr()[0]

        print('\n=== {0} Training Epoch: [{1:4}/{2:4}]\tlr: {3:.8f} ==='.format(
            training_mode, epoch, args.epochs - 1, lr
        ))

        for i, sample in enumerate(train_loader):

            img, label = sample['img'].to(device), sample['label'].to(device)

            output = water_net_v1(img)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                batch_time.update(time.time() - batch_endtime)
                batch_endtime = time.time()

                print('Batch: [{0:4}/{1:4}]\t'
                      'Time: {batch_time.val:.0f}s ({batch_time.sum:.0f}s)  \t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                    i, len(train_loader) - 1,
                    batch_time=batch_time, loss=losses))

        scheduler.step()
        epoch_time.update(time.time() - epoch_endtime)
        epoch_endtime = time.time()

        print('Time: {epoch_time.val:.0f}s ({epoch_time.sum:.0f}s)  \t'
              'Avg loss: {loss.avg:.4f}'.format(
            epoch_time=epoch_time, loss=losses))

        model_path = os.path.join(saved_models, 'cp_WaterNet_final.pth')
        checkpoint = {
            'epoch': epoch,
            'water_net_v1': water_net_v1.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': losses.avg,
        }
        torch.save(obj=checkpoint, f=model_path)
        print(f'Epoch {epoch}. Save latest model to {model_path}.')

        if losses.avg < best_loss:
            best_loss = losses.avg
            model_path = os.path.join(saved_models, 'cp_WaterNet_best.pth')
            torch.save(obj=checkpoint, f=model_path)
            print(f'Epoch {epoch}. Save best model to {model_path}.')


if __name__ == '__main__':
    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch WaterNet Training')
    parser.add_argument('--epochs', default=200, type=int, help='Number of total epochs to run (default 200).')
    parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate.')
    parser.add_argument('--dataset', type=str, default='/Ship01/Dataset/VOS/water', help='Dataset folder.')
    parser.add_argument(
        '-c', '--checkpoint', default=None, type=str, metavar='PATH',
        help='Path to latest checkpoint (default: none).')
    args = parser.parse_args()

    print('Args:', args)

    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    train_WaterNet()
