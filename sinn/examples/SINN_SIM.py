import torch.nn as nn
import torch
import importlib

from MDAnalysis.coordinates.GSD import GSDReader
from MDAnalysis import Universe

from sinn.dataset.dataset import FilteredAtomsDataset, collate_noise
from sinn.model.model import SchNet, Alignn
from sinn.train.train import test_model
from sinn.train.transforms import SimulatedNoiseRegressionEval

n_atoms = 2
spg = ('225',)
categorical_filter = ([True],['spg_number'],[spg])

batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

k = 17
pre_eval_func = SimulatedNoiseRegressionEval(k = k)

dataset = Universe('./test_traj/trajectory_LS4.5_FP0.5_RN105_BL10_DL5.35_aggregate.gsd')

dataset = FilteredAtomsDataset(source = dataset,
                               transform=pre_eval_func,
                               target = 'target').dataset

model_name = 'SchNet-AtomNoise-Spg225-1L'
model_path = 'models/24-06-16/'

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

model = torch.load(model_path + model_name + '.pkl')
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def hook_fn(module, input, output):
    fc2.append(output)
model.model.fc.register_forward_hook(hook_fn)

fc2 = []

preds = test_model(model = model, 
                   dataset=dataset,
                   device=device,)

preds = list(map(lambda x: torch.cat(x, dim=1), preds))

fc_save = torch.stack(fc2, dim=0)
preds_save = torch.stack(preds)

torch.save(fc_save, model_path + model_name + '_fc2.pkl')
torch.save(preds_save, model_path + model_name + '_preds.pkl')