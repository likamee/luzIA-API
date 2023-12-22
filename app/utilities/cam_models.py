# -*- coding: utf-8 -*-
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

class GlobalAveragePooling(Layer):
    def __init__(self, **kwargs):
        super(GlobalAveragePooling, self).__init__(**kwargs)

    def call(self, inputs):
        return K.mean(inputs, axis=(2, 3))

    def compute_output_shape(self, input_shape):
        return input_shape[0:2]

    def get_config(self):
        base_config = super(GlobalAveragePooling, self).get_config()
        return base_config

metrics = [
      BinaryAccuracy(name='accuracy'),
      AUC(name='auc')
    ]

def load_saved_model(model_path):
    """Load the saved model from the specified path."""
    custom_objects = {'Custom>GlobalAveragePooling': GlobalAveragePooling}
    return load_model(model_path, custom_objects=custom_objects)

