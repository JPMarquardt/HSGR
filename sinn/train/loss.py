import torch
import torch.nn as nn
import pandas

class RegressionClassificationLoss(nn.Module):
    """
    Custom loss function for combined regression and classification tasks
    """
    def __init__(self, num_classes, class_weights = None, device = None, alpha = 0.5):
        super(RegressionClassificationLoss, self).__init__()
        self.num_classes = num_classes

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        if class_weights is None:
            self.class_weights = torch.ones(num_classes).to(self.device)
        else:
            self.class_weights = class_weights
        self.alpha = alpha
        self.beta = 1 - alpha

        self.classification_loss = nn.BCELoss()
        self.regression_loss = nn.MSELoss()

    def forward(self, output, target):
        classification_pred = output[0]
        classification_target = target[0]

        class_weights = self.class_weights.unsqueeze(0).to(self.device)
        weight = torch.sum(class_weights * classification_target, dim=1)

        regression_pred = output[1].squeeze()
        regression_target = target[1]

        classification_loss = self.classification_loss(classification_pred, classification_target)
        regression_loss = self.regression_loss(regression_pred, regression_target)

        penalty = 1 - regression_target
        output = classification_loss * penalty * self.beta + regression_loss * self.alpha
        return torch.mean(weight * output)

def find_class_weights(dataset, target: str):
    """
    takes AtomsDataset with a target of one hot encoded classes 
    and returns the normalized inverse proportions of each class
    """
    dataset = dataset.df
    target = dataset[target]
    num_classes = target.nunique()
    class_weights = torch.zeros(num_classes)
    for val in target:
        class_weights += val
    class_weights = 1 / class_weights
    class_weights /= torch.sum(class_weights)
    return class_weights