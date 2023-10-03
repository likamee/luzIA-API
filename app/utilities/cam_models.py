# -*- coding: utf-8 -*-
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.models import load_model

metrics = [
      BinaryAccuracy(name='accuracy'),
      AUC(name='auc')
    ]

def load_saved_model(model_path):
    """Load the saved model from the specified path."""
    return load_model(model_path)


def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]
