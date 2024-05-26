from sinn.dataset.dataset import FilteredAtomsDataset, collate_spg
from sinn.noise.gaussian_noise import gaussian_noise
from sinn.model.model import SchNet

"""
Scale invariant neural network (SINN) for predicting space group of a crystal structure.
data = load_data(path)
data = filter_data(data)

either
    data = melt/crystalize/gas(data)

    data = manual_onehot_target(data)
or
    positives = add_small_noise(data)
    negatives = add_large_noise(data)

    data = merge(positives, negatives)

data = graph_construction(data)
train_data, test_data = split_data(data)

model = create_model()
model = train_model(model, train_data)
model = test_model(model, test_data)
"""

dataset = FilteredAtomsDataset(
    source='dft_3d',
    n_unique_atoms=(True, 3),
    categorical_filter=(('search', True, ['-']),)
)