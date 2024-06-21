import torch.nn as nn
import torch

from sinn.dataset.dataset import FilteredAtomsDataset, collate_multihead_noise
from sinn.model.model import SchNet, Alignn
from sinn.train.train import train_model
from sinn.train.transforms import NoiseRegressionEval

class SchNet_Multihead(nn.Module):
    def __init__(self, num_classes, num_layers, hidden_features, radial_features):
        super(SchNet_Multihead, self).__init__()
        self.model = SchNet(num_classes=num_classes+1, num_layers=num_layers, hidden_features=hidden_features, radial_features=radial_features)
        self.classifier = nn.Linear(num_classes+1, num_classes)
        self.regression = nn.Linear(num_classes+1, 1)

        self.sm = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.model(x)

        reg_pred = self.regression(x)

        class_pred = self.classifier(x)
        class_pred = self.sm(class_pred)
        
        return class_pred, reg_pred
    
model_name = 'SchNet-AtomNoise-Spg225-2L'
model_path = 'models/24-06-21/'
model = torch.load(f'{model_path}{model_name}.pkl')

n_atoms = 2
spg = (225, 220)
categorical_filter = ([True],['spg_number'],[spg])

batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 17
noise = lambda x: 1 - torch.sqrt(1 - x**2)
pre_eval_func = NoiseRegressionEval(noise = noise, k = k)

def custom_loss_func(output, target):
    classification_pred = output[0]
    classification_target = target[0]

    class_weights = torch.tensor([1, 0.1], device=device).unsqueeze(0)
    weight = torch.sum(class_weights * classification_target, dim = 1)

    regression_pred = output[1].squeeze()
    regression_target = target[1]

    dataset_loss = nn.BCELoss()(classification_pred, classification_target)
    noise_loss = nn.MSELoss()(regression_pred, regression_target)

    penalty = 1 - regression_target
    output = dataset_loss * penalty + noise_loss
    return torch.mean(weight * output)

dataset = FilteredAtomsDataset(source = "dft_3d",
                        n_unique_atoms = (True,n_atoms),
                        categorical_filter = categorical_filter,
                        target = 'spg_number',
                        transform=pre_eval_func,
                        collate = collate_multihead_noise,
                        ).dataset


loss_func = custom_loss_func
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model = model,
            dataset = dataset,
            loss_func = loss_func,
            optimizer = optimizer,
            n_epochs = 20,
            batch_size = batch_size,
            model_name=model_name,
            save_path = model_path,
            device = device)

train_model(model = model,
            dataset = dataset,
            loss_func = loss_func,
            optimizer = optimizer,
            n_epochs = 5,
            batch_size = batch_size,
            model_name=model_name,
            save_path = model_path,
            device = device,
            swa=True)
        