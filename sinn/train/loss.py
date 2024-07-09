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
            class_weights = torch.ones(num_classes).to(self.device)
        else:
            class_weights = class_weights.squeeze().to(self.device)
        
        if num_classes == 2:
            classification_loss = nn.BCEWithLogitsLoss(weight=class_weights)
        else:
            classification_loss = nn.CrossEntropyLoss(weight=class_weights)

        regression_loss = nn.MSELoss()
        
        if alpha > 1 or alpha < 0:
            raise ValueError("Alpha must be between 0 and 1")

        self.classification_loss = lambda pred, y: classification_loss(pred, y) * alpha
        self.regression_loss = lambda pred, y: regression_loss(pred, y) * (1 - alpha)

    def forward(self, output, target):
        classification_pred = output[0]
        classification_target = target[0]

        regression_pred = output[1].squeeze()
        regression_target = target[1]

        classification_loss = self.classification_loss(classification_pred, classification_target)
        regression_loss = self.regression_loss(regression_pred, regression_target)

        penalty = 1 - regression_target
        output = classification_loss * penalty + regression_loss
        return torch.mean(output)

def find_class_weights(dataset, target: str, exponential: bool = False):
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
    
    if exponential:
        class_weights = -torch.pow(exponential, class_weights) + 1
    
    class_weights = 1 / class_weights
    class_weights /= torch.sum(class_weights)
    return class_weights