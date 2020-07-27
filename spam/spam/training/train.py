from importlib import import_module

import nsml

from spam.spam_classifier.models import BasicModel
from spam.spam_classifier.models import STModel


def train(experiment_name: str = 'v1', pause: bool = False, mode: str = 'train', ST_name: str = 'v0'):

    config = import_module(f'spam.training.experiments.{ST_name}').config
    model = config['model'](**config[
        'model_kwargs'])  # model: STModel(network_fn = frozen_networks, network_kwargs = [input_size, len(classes)])
    STModel.bind_model(model)
    if pause:
        nsml.paused(scope=locals())
    if mode == 'train':
        base_dir = model.fit(**config['fit_kwargs'])
    #############

    config = import_module(f'spam.training.experiments.{experiment_name}').config
    config['model_kwargs']['dataset_kwargs']['base_dir'] = base_dir  # self training add
    model = config['model'](**config['model_kwargs'])#model: BasicModel(network_fn = frozen_networks, network_kwargs = [input_size, len(classes)])
    BasicModel.bind_model(model)
    if pause:
        nsml.paused(scope=locals())
    if mode == 'train':
        model.fit(**config['fit_kwargs'])
