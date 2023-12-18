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
    custom_objects = {'global_average_pooling': global_average_pooling}
    return load_model(model_path, custom_objects=custom_objects, compile=True, safe_mode=False)



def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]
