import os
import traceback
import sys
import argparse
import time
import gc
import warnings

from pathlib import Path

import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

import torch
from torch.utils import data

ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

from image_module.dataset_water import WaterDataset_RGB


time_str = time.strftime("%Y-%m-%d %H-%M-%S")
DEFAULT_CHKPT_DIR = os.path.join(ROOT_DIR, 'output', 'checkpoint_' + time_str)

warnings.filterwarnings("ignore", category=UserWarning)

# Device
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

# Input size must be a multiple of 32 as the image will be subsampled 5 times


def train(args):
    """
    Executes train script given arguments
    :param args: Training parameters
    :return:
    """
    try:
        torch.cuda.empty_cache()
    except:
        print("Error clearing cache.")
        print(traceback.format_exc())

    dataset_path = args.dataset_path
    input_shape = args.input_shape
    batch_size = args.batch_size
    init_lr = args.init_lr
    epochs = args.epochs
    out_path = args.out_path
    encoder_name = args.encoder

    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')

    # Input size must be a multiple of 32 as the image will be subsampled 5 times
    train_dataset = WaterDataset_RGB(
        mode='train_offline',
        dataset_path=train_dir,
        input_size=(416, 416)
    )

    val_dataset = WaterDataset_RGB(
        mode='train_offline',
        dataset_path=val_dir,
        input_size=(input_shape, input_shape)
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    linknet_model = smp.Linknet(
        encoder_name=encoder_name,
        encoder_depth=5,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )

    # Train LinkNet Model with given backbone

    try:
        train_model(
            linknet_model,
            init_lr=init_lr,
            num_epochs=epochs,
            out_path=out_path,
            train_loader=train_loader,
            val_loader=val_loader,
            encoder_name=encoder_name
        )
    except:
        print(traceback.format_exc())
    try:
        linknet_model = None
        gc.collect()
    except:
        print(traceback.format_exc())


def train_model(model, init_lr, num_epochs, out_path, train_loader, val_loader, encoder_name):
    """
    Trains a single image given model and further arguments
    :param model: Model from SMP library
    :param init_lr: Initial learning rate
    :param num_epochs: Number of epochs to train
    :param out_path: Folder to output checkpoints and model
    :param train_loader: Dataloader for train dataset
    :param val_loader: Dataloader for validation dataset
    :return:
    """
    plots_dir = os.path.join(out_path, 'graphs')
    checkpoints_dir = os.path.join(out_path, 'checkpoints')
    models_dir = os.path.join(out_path, 'model')

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=init_lr),
    ])

    # Create training epoch
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True
    )

    # Create validation epoch
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True
    )

    max_score = 0
    train_iou_score_ls = []
    train_dice_loss_ls = []

    val_iou_score_ls = []
    val_dice_loss_ls = []

    # Go through each epoch
    for epoch in range(0, num_epochs):
        title = 'Epoch: {}'.format(epoch)
        print('\nEpoch: {}'.format(epoch))

        # Epoch logs
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        # Checkpoint to resume training
        checkpoint = {
            'epoch': epoch,
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss.state_dict()
        }

        # Get IOU score
        score = float(valid_logs['iou_score'])

        checkpoint_savepth = os.path.join(checkpoints_dir, 'epoch_' + str(epoch).zfill(3) + '_score' + str(score) + '.pth')
        torch.save(checkpoint, checkpoint_savepth)

        # Check score on valid dataset
        if score > max_score:
            max_score = score
            model_savepth = os.path.join(models_dir, 'linknet_' + encoder_name + '_epoch_' + str(epoch).zfill(3) + '_score' + str(score) + '.pth')
            torch.save(model, model_savepth)
            print('New best model detected.')

        # Adjust learning rate halfway through training.
        if epoch == int(num_epochs / 2):
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


        train_iou_score_ls.append(train_logs['iou_score'])
        train_dice_loss_ls.append(train_logs['dice_loss'])

        val_iou_score_ls.append(valid_logs['iou_score'])
        val_dice_loss_ls.append(valid_logs['dice_loss'])

        plot_train_filepth = os.path.join(plots_dir, 'epoch_' + str(epoch).zfill(3) + '_train.png')
        plot_val_filepth = os.path.join(plots_dir, 'epoch_' + str(epoch).zfill(3) + '_val.png')
        plt.plot(train_iou_score_ls, label='train iou_score')
        plt.plot(train_dice_loss_ls, label='train dice_loss')
        plt.legend(loc="upper left")
        plt.title(title)
        plt.savefig(plot_train_filepth)
        plt.close()

        plt.plot(val_iou_score_ls, label='val iou_score')
        plt.plot(val_dice_loss_ls, label='val dice_loss')
        plt.legend(loc="upper left")
        plt.title(title)
        plt.savefig(plot_val_filepth)
        plt.close()


"""
    python train_segmodel.py --dataset_path 
"""
if __name__ == '__main__':
    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch WaterNet Model Testing')
    # Required: Path to the .pth file.
    parser.add_argument('--dataset_path',
                        type=str,
                        metavar='PATH',
                        help='Path to the dataset. Expects format shown in the header comments.')
    # Required: Model name. Can be efficient
    parser.add_argument('--encoder',
                        type=str,
                        metavar='PATH',
                        help='Encoder name, as used by segmentation_model.pytorch library')
    # Optional: Image input size that the model should be designed to accept. In LinkNet, image will be
    #           subsampled 5 times, and thus must be a factor of 32.
    parser.add_argument('--input_shape',
                        default=416,
                        type=int,
                        help='(OPTIONAL) Input size for model. Single integer, should be a factor of 32.')
    # Optional: Batch size for mini-batch gradient descent. Defaults to 4, depends on GPU and your input shape.
    parser.add_argument('--batch_size',
                        default=4,
                        type=int,
                        help='(OPTIONAL)  Batch size for mini-batch gradient descent.')
    # Initial Learning Rate: Initial learning rate. Learning gets set to 1e-5 halfway through training.
    parser.add_argument('--init_lr',
                        default=1e-4,
                        type=float,
                        help='(OPTIONAL)  Batch size for mini-batch gradient descent.')
    # Optional: Number of epochs for training
    parser.add_argument('--epochs',
                        default=300,
                        type=int,
                        help='(OPTIONAL) Number of epochs for training')
    # Optional: Which folder the checkpoints will be saved. Defaults to a new checkpoint folder in output.
    parser.add_argument('--out_path',
                        default=DEFAULT_CHKPT_DIR,
                        type=str,
                        metavar='PATH',
                        help='(OPTIONAL) Path to output folder, defaults to project root/output')
    _args = parser.parse_args()

    print("== System Details ==")
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print("== System Details ==")
    print()

    train(_args)
    print("Done.")