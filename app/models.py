import base64
import io

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from google.cloud import storage
from IPython.display import Image
from PIL import Image
from tensorflow import keras

from app.utilities import cam_models


def save_heatmap(heatmap, output_path="heatmap.png"):
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((299, 299))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def load_model_from_gcs(bucket_name: str, model_path: str):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = storage.Blob(model_path, bucket)
    local_model_path = f"/tmp/{model_path}"
    blob.download_to_filename(local_model_path)
    return cam_models.load_saved_model(local_model_path)

def evaluate_models(models, image_array: np.ndarray):
    predictions = [model.predict(image_array) for model in models]
    return predictions

def superimpose_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    preprocess_input = keras.applications.xception.preprocess_input
    img = preprocess_input(img)

    heatmap = heatmap / np.max(heatmap)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * cv2.resize(heatmap, (img.shape[2], img.shape[1])))

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    img_processed = (img[0] + 1) * 127.5

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    # squeeze() removes the batch dimension
    superimposed_img = (jet_heatmap * alpha) + (img_processed * (1 - alpha))
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Save the superimposed image
    with io.BytesIO() as buffer:
        img_pil = keras.preprocessing.image.array_to_img(superimposed_img)
        img_pil.save(buffer, format='JPEG')
        buffer.seek(0)
        superimposed_image = buffer.getvalue()

    return superimposed_image


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)


    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def get_cam(model, image):

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(image, model, "block5_conv3")

    img_bytes = superimpose_gradcam(image, heatmap)

    # Save the image to a BytesIO buffer and encode it as a base64 string
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return img_base64
