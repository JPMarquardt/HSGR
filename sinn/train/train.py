import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sinn.train.utils import gen_to_func
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler



def test_model(model, dataset, device):
    """Runs one epoch of training or evaluation."""

    pred_list = []
    model.eval()

    graph_to = gen_to_func(dataset[0], device)

    with torch.no_grad():
        for step, (g, y) in enumerate(tqdm(dataset)):

            g = graph_to(g)

            pred = model(g)
            pred_list.append(pred)

    return pred_list

def run_epoch(model, loader, loss_func, optimizer, device, epoch, scheduler = None, train=True, debug=False, swa=False):
    """Runs one epoch of training or evaluation."""

    ave_loss = 0

    if train:
        model.train()
        grad = torch.enable_grad()
        train_or_test = 'Train'
        def bw_closure():
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    else:
        model.eval()
        grad = torch.no_grad()
        train_or_test = 'Validation'
        def bw_closure():
            pass

    if debug:
        grad = (torch.autograd.set_detect_anomaly(True), grad)

    graph_to = gen_to_func(next(iter(loader))[0], device)
    y_to = gen_to_func(next(iter(loader))[1], device)

    print(next(iter(loader)))
    with grad:
        for step, (g, y) in enumerate(tqdm(loader)):
            g = graph_to(g)
            y = y_to(y)

            pred = model(g)
            loss = loss_func(pred, y)
            bw_closure()

            inv_step = 1/(step + 1)
            inv_step_comp = 1 - inv_step
            ave_loss = ave_loss * inv_step_comp + loss.item() * inv_step

            torch.cuda.empty_cache()

    if swa:
        swa_model = swa
        swa_model.update_parameters(model)
    
    if scheduler:
        if type(scheduler) == tuple:
            for index in range(len(scheduler)):
                scheduler[index].step()
        else:
            scheduler.step()

    print(f'Epoch {epoch}-- {train_or_test} Loss: {ave_loss}')

    return ave_loss

def train_model(model,
                dataset,
                n_epochs,
                model_name,
                device = 'cpu',
                loss_func = nn.MSELoss(),
                optimizer = None,
                save_path = '',
                batch_size = 4,
                loss_graph = True,
                scheduler = None,
                swa = False
                ):
    
    if swa:
        model_name = model_name + '_swa'
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        if scheduler is None:
            scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.0005)
    else:
        swa_model = False

    if optimizer == None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    t_device = torch.device(device)
    model = model.to(t_device)

    ave_training_loss = []
    ave_test_loss = []
    epoch_saved = []

    train_loader = dataset[dataset.split["train"]]
    val_loader = dataset[dataset.split["val"]]

    for epoch in range(n_epochs):
        ave_loss = run_epoch(model=model,
                            loader=train_loader,
                            loss_func=loss_func,
                            optimizer=optimizer,
                            device=t_device,
                            epoch=epoch,
                            scheduler=scheduler,
                            train=True,
                            swa=swa_model)

        ave_training_loss.append(ave_loss)

        ave_loss = run_epoch(model=model,
                            loader=val_loader,
                            loss_func=loss_func,
                            optimizer=optimizer, 
                            device=t_device,
                            epoch=epoch,
                            scheduler=None,
                            train=False,
                            swa=False)

        ave_test_loss.append(ave_loss)

        if ave_loss <= min(ave_test_loss):
            output_dir = f'{save_path}{model_name}.pkl'
            with open(output_dir, 'wb') as output_file:
                torch.save(model, output_file)
            epoch_saved.append(epoch)

        if loss_graph:  
            plt.figure()
            plt.plot(ave_training_loss, label = 'train')
            plt.plot(ave_test_loss, label = 'test')
            plt.xlabel("training epoch")
            plt.ylabel("loss")
            plt.semilogy()
            plt.legend(loc='upper right')
            plt.savefig(f'{save_path}{model_name}_loss.png')
            plt.close()



