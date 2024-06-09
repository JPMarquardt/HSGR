import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import nfflr
import sys
import datetime

from tqdm import tqdm
from typing import Callable
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sinn.graph.graph import create_supercell, create_labeled_supercell, create_knn_graph, lattice_plane_slicer, create_periodic_graph
from sinn.dataset.dataset import collate_noise
from sinn.noise.gaussian_noise import noise_regression
from sinn.simulation.utils import find_max_k_dist
from sinn.simulation.simulation import box_filter


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
        train_or_test = 'Test'

    with grad:
        for step, (g, y) in enumerate(tqdm(loader)):
            if isinstance(g, tuple):
                g = tuple(graph_part.to(device) for graph_part in g)
            else:
                g# = g.to(device)

            y #= y.to(device)
            pred = model(g)
            loss = loss_func(pred, y)
            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            mae = torch.sum(torch.abs(y - pred))

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
                pre_eval_func = None,
                swa = False
                ):
    
    t_device = torch.device(device)

    if optimizer == None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    model = model.to(t_device)

    ave_training_mae = []
    ave_training_loss = []
    ave_test_loss = []
    ave_test_mae = []
    final_average_mae = []
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

        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate,
            sampler=SubsetRandomSampler(dataset.split["test"]),
            drop_last=True
        )

        ave_loss, ave_mae = run_epoch(model=model,
                                      loader=test_loader,
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

def noise_regression_sim_prep(a: nfflr.Atoms, k: int = 9):
    data = a.positions
    lattice = a.cell
    numbers = a.numbers
    replicates = 3

    dx = 0.1 * torch.min(torch.norm(lattice, dim=1))
    supercell, atom_id, cell_id = create_labeled_supercell(data, n=replicates, lattice=lattice)
    numbers = numbers.repeat(replicates**3)
    filt = box_filter(supercell, lattice, dx)

    supercell = supercell[filt]
    atom_id = atom_id[filt]
    cell_id = cell_id[filt]
    numbers = numbers[filt]

    g = create_knn_graph(supercell, k=k, line_graph=False)

    g.ndata['z'] = numbers
    g.ndata['atom_id'] = atom_id
    g.ndata['cell_id'] = cell_id

    g = create_periodic_graph(g)

    return g

class NoiseRegressionEval(nn.Module):
    def __init__(self, noise, k):
        super(NoiseRegressionEval, self).__init__()
        self.noise = noise
        self.k = k

    def forward(self, datapoint):
        n_atoms = torch.randint(1, 5, (1,)) * 1000
        return noise_regression_prep(datapoint, n_atoms, self.noise, self.k)
    
class SimulatedNoiseRegressionEval(nn.Module):
    def __init__(self, k):
        super(SimulatedNoiseRegressionEval, self).__init__()
        self.k = k

    def forward(self, datapoint):
        return noise_regression_sim_prep(datapoint, self.k)
