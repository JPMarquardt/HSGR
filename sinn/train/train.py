import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import nfflr
import sys

from tqdm import tqdm
from typing import Callable
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sinn.graph.graph import create_supercell, create_knn_graph, lattice_plane_slicer
from sinn.dataset.dataset import collate_noise
from sinn.noise.gaussian_noise import noise_regression


def run_epoch(model, loader, loss_func, optimizer, device, epoch, scheduler = None, train=True, swa=False):
    """Runs one epoch of training or evaluation."""

    ave_MAE = 0
    ave_loss = 0

    if train:
        model.train()
        grad = torch.enable_grad()
        train_or_test = 'Train'
    else:
        model.eval()
        grad = torch.no_grad()
        train_or_test = 'Test'
    torch.autograd.set_detect_anomaly(True)
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

            MAE = torch.sum(torch.abs(y - torch.where(y == 1, pred, 0)))/y.shape[0]

            inv_step = 1/(step + 1)
            inv_step_comp = 1 - inv_step
            ave_loss = ave_loss * inv_step_comp + loss.item() * inv_step
            ave_MAE = ave_MAE * inv_step_comp + MAE.item() * inv_step

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

    print(f'Epoch {epoch}-- {train_or_test} Loss: {ave_loss} {train_or_test} MAE: {ave_MAE}')

    return ave_loss, ave_MAE

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
                MAE_graph = True,
                scheduler = None,
                pre_eval_func = None,
                swa = False
                ):
    
    t_device = torch.device(device)

    if optimizer == None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    model = model.to(t_device)

    ave_training_MAE = []
    ave_training_loss = []
    ave_test_loss = []
    ave_test_MAE = []
    final_average_MAE = []
    epoch_saved = []

    base_dataset = dataset

    for epoch in range(n_epochs):

        dataset = deepcopy(base_dataset)
        if pre_eval_func:
            dataset = pre_eval_func(dataset)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate,
            sampler=SubsetRandomSampler(dataset.split["train"]),
            drop_last=True
        )
        ave_loss, ave_MAE = run_epoch(model=model,
                                      loader=train_loader,
                                      loss_func=loss_func,
                                      optimizer=optimizer,
                                      device=t_device,
                                      epoch=epoch,
                                      scheduler=scheduler,
                                      train=True,
                                      swa=swa)

        ave_training_loss.append(ave_loss)
        ave_training_MAE.append(ave_MAE)

        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate,
            sampler=SubsetRandomSampler(dataset.split["test"]),
            drop_last=True
        )

        ave_loss, ave_MAE = run_epoch(model=model,
                                      loader=test_loader,
                                      loss_func=loss_func,
                                      optimizer=optimizer, 
                                      device=t_device,
                                      epoch=epoch,
                                      scheduler=None,
                                      train=False,
                                      swa=False)

        ave_test_loss.append(ave_loss)
        ave_test_MAE.append(ave_MAE)

        if ave_loss <= min(ave_test_loss):
            output_dir = f'{save_path}{model_name}.pkl'
            with open(output_dir, 'wb') as output_file:
                torch.save(model, output_file)
            final_average_MAE.append(ave_MAE)
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

        if MAE_graph:
            plt.figure()
            plt.plot(ave_training_MAE, label = 'train')
            plt.plot(ave_test_MAE, label = 'test')
            plt.plot(epoch_saved, final_average_MAE, 'r.', label = 'saved')
            plt.xlabel("training epoch")
            plt.ylabel("loss")
            plt.semilogy()
            plt.legend(loc='upper right')
            plt.savefig(f'{save_path}{model_name}_MAE.png')
            plt.close()

def noise_regression_prep(a: nfflr.Atoms, n_target_atoms: int, noise: Callable = None, k: int = 9):
    coords = a.positions
    lattice = a.cell
    numbers = a.numbers

    data = torch.matmul(coords, torch.inverse(lattice))

    replicates = (n_target_atoms / data.size()[0]) ** (1/3)
    replicates = int(replicates)

    miller_index = torch.randint(0, 4, (3,))

    if noise is None:
        noise = lambda x: x

    supercell = create_supercell(data, replicates)
    sample_noise, supercell = noise_regression(supercell, noise)
    supercell = lattice_plane_slicer(supercell, miller_index, replicates)
    supercell = supercell @ lattice

    g = create_knn_graph(supercell, k=k, line_graph=False)
    numbers = numbers.repeat(replicates**3)
    g.ndata['z'] = numbers

    return g, sample_noise

class NoiseRegressionEval(nn.Module):
    def __init__(self, noise, k):
        super(NoiseRegressionEval, self).__init__()
        self.noise = noise
        self.k = k

    def forward(self, datapoint):
        n_atoms = torch.randint(1, 5, (1,)) * 1000
        return noise_regression_prep(datapoint, n_atoms, self.noise, self.k)
