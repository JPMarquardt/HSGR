import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler



def test_model(model, dataset, device):
    """Runs one epoch of training or evaluation."""

    pred_list = []
    with torch.no_grad():
        for step, (g, y) in enumerate(tqdm(dataset)):
            if isinstance(g, tuple):
                g = tuple(graph_part.to(device) for graph_part in g)
            else:
                g = g.to(device)

            pred = model(g)
            pred_list.append(pred.item())

    return pred_list

def run_epoch(model, loader, loss_func, optimizer, device, epoch, scheduler = None, train=True, swa=False):
    """Runs one epoch of training or evaluation."""

    ave_mae = 0
    ave_loss = 0

    if train:
        model.train()
        grad = torch.enable_grad()
        train_or_test = 'Train'
    else:
        model.eval()
        grad = torch.no_grad()
        train_or_test = 'Validation'

    with grad:
        for step, (g, y) in enumerate(tqdm(loader)):
            if isinstance(g, tuple):
                g = tuple(graph_part.to(device) for graph_part in g)
            else:
                g = g.to(device)

            y = y.to(device)
            pred = model(g)
            loss = loss_func(pred, y)
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            mae = torch.mean(torch.abs(y - pred))

            inv_step = 1/(step + 1)
            inv_step_comp = 1 - inv_step
            ave_loss = ave_loss * inv_step_comp + loss.item() * inv_step
            ave_mae = ave_mae * inv_step_comp + mae.item() * inv_step

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

    print(f'Epoch {epoch}-- {train_or_test} Loss: {ave_loss} {train_or_test} mae: {ave_mae}')

    return ave_loss, ave_mae

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
                mae_graph = True,
                scheduler = None,
                swa = False
                ):
    
    if swa:
        model_name = model_name + '_swa'

    if optimizer == None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    t_device = torch.device(device)
    model = model.to(t_device)

    ave_training_mae = []
    ave_training_loss = []
    ave_test_loss = []
    ave_test_mae = []
    final_average_mae = []
    epoch_saved = []

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate,
        sampler=SubsetRandomSampler(dataset.split["train"]),
        drop_last=True
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate,
        sampler=SubsetRandomSampler(dataset.split["val"]),
        drop_last=True
    )

    for epoch in range(n_epochs):

        ave_loss, ave_mae = run_epoch(model=model,
                                      loader=train_loader,
                                      loss_func=loss_func,
                                      optimizer=optimizer,
                                      device=t_device,
                                      epoch=epoch,
                                      scheduler=scheduler,
                                      train=True,
                                      swa=swa)

        ave_training_loss.append(ave_loss)
        ave_training_mae.append(ave_mae)

        ave_loss, ave_mae = run_epoch(model=model,
                                      loader=val_loader,
                                      loss_func=loss_func,
                                      optimizer=optimizer, 
                                      device=t_device,
                                      epoch=epoch,
                                      scheduler=None,
                                      train=False,
                                      swa=False)

        ave_test_loss.append(ave_loss)
        ave_test_mae.append(ave_mae)

        if ave_loss <= min(ave_test_loss):
            output_dir = f'{save_path}{model_name}.pkl'
            with open(output_dir, 'wb') as output_file:
                torch.save(model, output_file)
            final_average_mae.append(ave_mae)
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

        if mae_graph:
            plt.figure()
            plt.plot(ave_training_mae, label = 'train')
            plt.plot(ave_test_mae, label = 'test')
            plt.plot(epoch_saved, final_average_mae, 'r.', label = 'saved')
            plt.xlabel("training epoch")
            plt.ylabel("loss")
            plt.semilogy()
            plt.legend(loc='upper right')
            plt.savefig(f'{save_path}{model_name}_mae.png')
            plt.close()


