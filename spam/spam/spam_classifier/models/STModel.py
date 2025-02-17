from typing import Callable, List

import keras
from keras.optimizers import SGD, Adam
import nsml
import numpy as np
import pandas as pd

from spam.spam_classifier.datasets.dataset import Dataset
from spam.spam_classifier.models.utils import Metrics, NSMLReportCallback, evaluate

class STModel:
    """
    A basic model that first finetunes the last layer of a pre-trained network, and then unfreezes all layers and
    train them.
    """

    def __init__(self, network_fn: Callable, dataset_cls: Dataset, dataset_kwargs, network_kwargs):
        self.data: Dataset = dataset_cls(**kwargs_or_empty_dict(dataset_kwargs))# (classes, input_size)
        self.network: keras.Model = network_fn(**kwargs_or_empty_dict(network_kwargs))
        self.debug = False

    def fit(self, epochs_finetune, epochs_full, batch_size, debug=False):
        sessionName = 'qkek984/spam-3/59'
        nsml.load(checkpoint='best', session=sessionName)
        print(sessionName, "model load!")
        #nsml.save(checkpoint='saved')
        #exit()
        self.debug = debug
        self.data.prepare(unlabeledset=True)
        print("lenunlabeled : ", self.data.lenUnlabeled('unlabeled'))  # check unlabeldata

        self.network.compile(
            loss=self.loss(),
            optimizer=self.optimizer('full'),
            metrics=self.fit_metrics()
        )

        val_gen = self.data.ST_val_gen(batch_size)

        self.myMetrics(val_gen=val_gen, batch_size=batch_size)  # do self training

        return self.data.base_dir

    def loss(self) -> str:
        loss = keras.losses.CategoricalCrossentropy()
        return loss

    def optimizer(self, stage: str) -> keras.optimizers.Optimizer:
        return {
            'finetune': SGD(lr=1e-4, momentum=0.9),
            'full': Adam(lr=1e-4)
        }[stage]

    def fit_metrics(self) -> List[str]:
        return ['accuracy']


    def evaluate(self, test_dir: str) -> pd.DataFrame:
        """

        Args:
            test_dir: Path to the test dataset.

        Returns:
            ret: A dataframe with the columns filename and y_pred. One row is the prediction (y_pred)
                for that file (filename). It is important that this format is used for NSML to be able to evaluate
                the model for the leaderboard.

        """
        gen, filenames = self.data.test_gen(test_dir=test_dir, batch_size=64)
        y_pred = self.network.predict_generator(gen)
        ret = pd.DataFrame({'filename': filenames, 'y_pred': np.argmax(y_pred, axis=1)})
        return ret

    def myMetrics(self, val_gen, batch_size) -> None:#self supervised learning

        class_Unlabeled = [[], [], [], []]
        y_true, y_prob = evaluate(data_gen=val_gen, model=self.network)
        y_true, y_pred = [np.argmax(y, axis=1) for y in [y_true, y_prob]]

        for metadata in zip(y_true, y_pred, y_prob):
            if metadata[0] != metadata[1]:
                class_Unlabeled[metadata[1]].append(max(metadata[2]))

        class_threshold = []
        for i in range(0,len(class_Unlabeled)):
            class_prob = max(class_Unlabeled[i])

            class_threshold.append(max(class_prob,0.99))
            print("class:",i,", max_prob:",class_prob)
            print(class_Unlabeled[i][:10])

        print("each error: ",len(class_Unlabeled[0]),len(class_Unlabeled[1]),len(class_Unlabeled[2]),len(class_Unlabeled[3]))
        print("each class_thresh : ",class_threshold)


        ##########################################################################
        unlabeled_gen = self.data.test_unlabeled_gen(batch_size = batch_size)
        class_Unlabeled = [[], [], [], []]
        output = self.network.predict_generator(unlabeled_gen)
        pred = np.argmax(output, axis=1)
        for metadata in zip(unlabeled_gen.filenames, pred, output):
            pred_prob = max(metadata[2])
            if class_threshold[metadata[1]] < pred_prob:
                class_Unlabeled[metadata[1]].append(metadata[0])
        class_Unlabeled[0] = class_Unlabeled[0][:1500]# suppress too much data
        print("class_Unlabeled: ", len(class_Unlabeled[0]), len(class_Unlabeled[1]), len(class_Unlabeled[2]),
              len(class_Unlabeled[3]))
        self.data.insertUnlabeledData(class_Unlabeled)

        print("sucesss !")

def bind_model(model: STModel):
    """
    Utility function to make the model work with leaderboard submission.
    """

    def load(dirname, **kwargs):
        model.network.load_weights(f'{dirname}/model')

    def save(dirname, **kwargs):
        filename = f'{dirname}/model'
        print(f'Trying to save to {filename}')
        model.network.save_weights(filename)

    def infer(test_dir, **kwargs):
        return model.evaluate(test_dir)

    nsml.bind(load=load, save=save, infer=infer)


def kwargs_or_empty_dict(kwargs):
    if kwargs is None:
        kwargs = {}
    return kwargs
