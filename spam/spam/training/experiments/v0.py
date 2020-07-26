from spam.spam_classifier.datasets.dataset import Dataset
from spam.spam_classifier.models.STModel import STModel
from spam.spam_classifier.networks.Networks import frozen_networks

input_size = (256, 256, 3)
classes = ['normal', 'monotone', 'screenshot', 'unknown']
config = {
    'model': STModel,
    'fit_kwargs': {
        'batch_size': 64,
        'epochs_finetune': 10,
        'epochs_full': 10,
        'debug': False
    },
    'model_kwargs': {
        'network_fn': frozen_networks,
        'network_kwargs': {
            'input_size': input_size,
            'n_classes': len(classes)
        },
        'dataset_cls': Dataset,
        'dataset_kwargs': {
            'classes': classes,
            'input_size': input_size,
            'base_dir': False
        },
    },
}
