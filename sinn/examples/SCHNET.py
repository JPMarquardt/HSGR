import torch.nn as nn
import torch
from datetime import datetime
import os

from sinn.dataset.dataset import FilteredAtomsDataset, collate_multihead_noise
from sinn.model.schnet import SchNet_Multihead
from sinn.train.train import train_model
from sinn.train.transforms import NoiseRegressionTrain
from sinn.train.loss import RegressionClassificationLoss, find_class_weights

date = datetime.now().strftime("%y-%m-%d")

for alpha in [0.5, 0.25, 0.1, 0.01]:
    n_atoms = 2
    spg = list(range(215,231))
    categorical_filter = ([True],['spg_number'],[spg])

    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    k = 17
    noise = lambda: 1 - torch.sqrt(1 - torch.rand(1)**2)
    crystal_size = lambda: torch.randint(1, 5, (1,)) * 1000

    pre_eval_func = NoiseRegressionTrain(noise = noise, crystal_size=crystal_size, k = k)
    target = 'international_number'

    dataset = FilteredAtomsDataset(source = "dft_3d",
                            n_unique_atoms = (True,n_atoms),
                            categorical_filter = categorical_filter,
                            target = target,
                            transform=pre_eval_func,
                            collate = collate_multihead_noise,
                            ).dataset

    class_weights = find_class_weights(dataset, target, exponential=False)
    print(class_weights)

    num_classes = class_weights.size(0)
    num_layers = 4

    model = SchNet_Multihead(num_classes = num_classes, num_layers = num_layers, hidden_features = 64, radial_features = 256)
    model_type_name = type(model).__name__

    model_path = f'models/{date}/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    loss_func = RegressionClassificationLoss(num_classes=num_classes, class_weights=class_weights, alpha=alpha, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model_num = 7
    model_name = f'{model_type_name}-k{k}-L{num_layers}-Spg{num_classes}-a{alpha}-n{model_num}'
    print(model_name)

    scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=20)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0.0001)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[30])

    train_model(model = model,
                dataset = dataset,
                loss_func = loss_func,
                optimizer = optimizer,
                scheduler=scheduler,
                n_epochs = 60,
                batch_size = batch_size,
                model_name=model_name,
                save_path = model_path,
                device = device)

