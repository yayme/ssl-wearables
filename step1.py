""""

Differences: ( can be changed/revised later)
- Removed DDP/distributed training (setup, cleanup, main_worker with rank)
- Removed Hydra config system (hardcoded parameters instead)
- Removed activity_dataset, subject_collate, worker_init_fn (using TensorDataset)
- Removed data augmentation (RandomSwitchAxisTimeSeries, RotationAxisTimeSeries)
- Removed multi-GPU support (nn.DataParallel)
- Removed linear learning rate scaling (set_linear_scale_lr)
- Removed check_file_list and file-based data loading
- Added create_dummy_dataset function
- Simplified set_up_data4train (removed cfg and rank parameters)
- Simplified evaluate_model (removed cfg and rank parameters)
- Simplified compute_loss (removed cfg parameter, all 4 tasks always active)
- Removed torchsummary, signal handling and multi-processing imports
"""
#imports

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sslearning.pytorchtools import EarlyStopping
import warnings

from sslearning.models.accNet import SSLNET, Resnet

cuda = torch.cuda.is_available()
now = datetime.now()
#heck: is set_seed() enough for reproducibility?
def set_seed(my_seed=0):
    random_seed = my_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if cuda:
        torch.cuda.manual_seed_all(random_seed)

def set_up_data4train(my_X, aot_y, scale_y, permute_y, time_w_y, my_device):
    aot_y, scale_y, permute_y, time_w_y = (
        Variable(aot_y),
        Variable(scale_y),
        Variable(permute_y),
        Variable(time_w_y),
    )
    my_X = Variable(my_X)
    my_X = my_X.to(my_device, dtype=torch.float)
    aot_y = aot_y.to(my_device, dtype=torch.long)
    scale_y = scale_y.to(my_device, dtype=torch.long)
    permute_y = permute_y.to(my_device, dtype=torch.long)
    time_w_y = time_w_y.to(my_device, dtype=torch.long)
    return my_X, aot_y, scale_y, permute_y, time_w_y

def evaluate_model(model, data_loader, my_device):
    model.eval()
    losses = []
    acces = []
    task_losses = []

    for i, (my_X, aot_y, scale_y, permute_y, time_w_y) in enumerate(data_loader):
        with torch.no_grad():
            my_X, aot_y, scale_y, permute_y, time_w_y = set_up_data4train(
                my_X, aot_y, scale_y, permute_y, time_w_y, my_device
            )
            aot_y_pred, scale_y_pred, permute_y_pred, time_w_h_pred = model(my_X)

            loss, acc, task_loss = compute_loss(
                aot_y,
                scale_y,
                permute_y,
                time_w_y,
                aot_y_pred,
                scale_y_pred,
                permute_y_pred,
                time_w_h_pred,
            )
            losses.append(loss.item())
            acces.append(acc.item())
            task_losses.append(task_loss)
    losses = np.array(losses)
    acces = np.array(acces)
    task_losses = np.array(task_losses)
    return losses, acces, task_losses

def log_performance(current_loss, current_acces, writer, mode, epoch, task_name, task_loss=[]):
    loss = np.mean(current_loss)
    acc = np.mean(current_acces)

    writer.add_scalar(mode + "/" + task_name + "_loss", loss, epoch)
    writer.add_scalar(mode + "/" + task_name + "_acc", acc, epoch)

    if len(task_loss) > 0:
        aot_loss = np.mean(task_loss[:, 0])
        permute_loss = np.mean(task_loss[:, 1])
        scale_loss = np.mean(task_loss[:, 2])
        time_w_loss = np.mean(task_loss[:, 3])
        writer.add_scalar(mode + "/aot_loss", aot_loss, epoch)
        writer.add_scalar(mode + "/permute_loss", permute_loss, epoch)
        writer.add_scalar(mode + "/scale_loss", scale_loss, epoch)
        writer.add_scalar(mode + "/time_w_loss", time_w_loss, epoch)

    return loss

def compute_acc(logits, true_y):
    pred_y = torch.argmax(logits, dim=1)
    acc = torch.sum(pred_y == true_y)
    acc = 1.0 * acc / (pred_y.size()[0])
    return acc

def compute_loss(
    aot_y,
    scale_y,
    permute_y,
    time_w_y,
    aot_y_pred,
    scale_y_pred,
    permute_y_pred,
    time_w_h_pred,
):
    entropy_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    total_task = 0
    total_acc = 0
    aot_loss = 0
    permute_loss = 0
    scale_loss = 0
    time_w_loss = 0

    aot_loss = entropy_loss_fn(aot_y_pred, aot_y)
    permute_loss = entropy_loss_fn(permute_y_pred, permute_y)
    scale_loss = entropy_loss_fn(scale_y_pred, scale_y)
    time_w_loss = entropy_loss_fn(time_w_h_pred, time_w_y)

    total_loss = aot_loss + permute_loss + scale_loss + time_w_loss
    total_acc = compute_acc(aot_y_pred, aot_y) + compute_acc(permute_y_pred, permute_y) + compute_acc(scale_y_pred, scale_y) + compute_acc(time_w_h_pred, time_w_y)
    total_task = 4

    return (
        total_loss / total_task,
        total_acc / total_task,
        [aot_loss.item(), permute_loss.item(), scale_loss.item(), time_w_loss.item()],
    )
# we will replace this with our data.
def create_dummy_dataset(num_samples, num_channels, sequence_length):
    my_X = torch.randn(num_samples, num_channels, sequence_length)
    aot_y = torch.randint(0, 2, (num_samples,))
    scale_y = torch.randint(0, 2, (num_samples,))
    permute_y = torch.randint(0, 2, (num_samples,))
    time_w_y = torch.randint(0, 2, (num_samples,))
    return TensorDataset(my_X, aot_y, scale_y, permute_y, time_w_y)

def main():
    set_seed()
# this parameters can possibly be set up in a more automatic way via python libraries.
    num_epochs = 10
    lr = 0.0001
    batch_size = 16
    GPU = 0
    log_interval = 10
    epoch_len = 10
    sample_rate = 30
    num_train_samples = 100
    num_test_samples = 20

    main_log_dir = "logs"
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    log_dir = os.path.join(main_log_dir, "step1_" + dt_string)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    # name needs to be revised for better results checking.
    model_path = os.path.join(main_log_dir, "models", "step1_" + dt_string + ".mdl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("Learning rate: %f" % lr)
    print("Number of epoches: %d" % num_epochs)
    print("GPU usage: %d" % GPU)
    print("Batch size : %d" % batch_size)
    print("Tensor log dir: %s" % log_dir)

    if GPU >= 0 and cuda:
        my_device = "cuda:" + str(GPU)
    else:
        my_device = "cpu"
#is_mtl means multitask learning, resnet_version refers to version defined. Notable difference in number of layers.
    model = Resnet(
        output_size=2,
        resnet_version=18,
        epoch_len=epoch_len,
        is_mtl=True,
    )
    model = model.float()
    print(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num of paras %d " % pytorch_total_params)

    print("Training using device %s" % my_device)
    model.to(my_device, dtype=torch.float)

    train_dataset = create_dummy_dataset(num_train_samples, 3, sample_rate * epoch_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    test_dataset = create_dummy_dataset(num_test_samples, 3, sample_rate * epoch_len)
    #pin_memory mihgt need adjustment.
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    total_step = len(train_loader)

    print("Start training")
    #early stopping needs to be adjusted or removed if necessary
    early_stopping = EarlyStopping(patience=5, path=model_path, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_acces = []
        task_losses = []

        for i, (my_X, aot_y, scale_y, permute_y, time_w_y) in enumerate(train_loader):
            my_X, aot_y, scale_y, permute_y, time_w_y = set_up_data4train(
                my_X, aot_y, scale_y, permute_y, time_w_y, my_device
            )
            aot_y_pred, scale_y_pred, permute_y_pred, time_w_h_pred = model(my_X)

            loss, acc, task_loss = compute_loss(
                aot_y,
                scale_y,
                permute_y,
                time_w_y,
                aot_y_pred,
                scale_y_pred,
                permute_y_pred,
                time_w_h_pred,
            )
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            if i % log_interval == 0:
                msg = "Train: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, ACC : {:.4f}".format(
                    epoch + 1,
                    num_epochs,
                    i,
                    total_step,
                    loss.item(),
                    acc.item(),
                )
                print(msg)
            train_losses.append(loss.cpu().detach().numpy())
            train_acces.append(acc.cpu().detach().numpy())
            task_losses.append(task_loss)

        train_task_losses = np.array(task_losses)

        train_losses = np.array(train_losses)
        train_acces = np.array(train_acces)
        test_losses, test_acces, task_losses = evaluate_model(
            model, test_loader, my_device
        )

        log_performance(
            train_losses,
            train_acces,
            writer,
            "train",
            epoch,
            "ssl_task",
            task_loss=train_task_losses,
        )
        test_loss = log_performance(
            test_losses,
            test_acces,
            writer,
            "test",
            epoch,
            "ssl_task",
            task_loss=task_losses,
        )

        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":
    main()

#next step: change into real dataset. find what needs to be fixed.