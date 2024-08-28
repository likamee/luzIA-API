import base64
import io
import os

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from google.cloud import storage
from IPython.display import Image
from PIL import Image
from tensorflow import keras
from utilities import cam_models


def save_heatmap(heatmap, output_path="heatmap.png"):
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((299, 299))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def load_model_from_gcs(bucket_name: str, model_path: str):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials/credentials.json"
    print(f"Downloading model from {bucket_name}/{model_path}")
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = storage.Blob(model_path, bucket)
    local_model_path = f"/tmp/{model_path}"
    blob.download_to_filename(local_model_path)
    return cam_models.load_saved_model(local_model_path)


def evaluate_model(model, image_array: np.ndarray):
    prediction = model.predict(image_array)
    return prediction


def superimpose_gradcam(image_path, heatmap, alpha=0.4, cam_path="cam.jpg"):
    # Load the original image
    original_img = cv2.imread(image_path)
    # resize the original image to the size of the heatmap
    original_img = cv2.resize(original_img, (299, 299))
    original_img = np.clip(original_img, 0, 255).astype(
        np.uint8
    )  # Ensure values are still in the range [0, 255]

    heatmap1 = heatmap - np.min(heatmap)
    heatmap1 = heatmap1 / np.ptp(heatmap1)

    # Rescale heatmap to a range 0-255
    heatmap1 = np.uint8(
        255 * cv2.resize(heatmap1, (original_img.shape[0], original_img.shape[1]))
    )

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap1]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + original_img
    # superimposed_img = (jet_heatmap * alpha) + (original_img * (1 - alpha))

    # Ensure the resulting image is still in the range [0, 255]
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Save the superimposed image in JPEG format
    with io.BytesIO() as buffer:
        img_pil = keras.preprocessing.image.array_to_img(superimposed_img_rgb)
        img_pil.save(buffer, format="JPEG")
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

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    return heatmap.numpy()


def get_cam(model, image, img_path):

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(image, model, "block5_conv3")

    img_bytes = superimpose_gradcam(img_path, heatmap)

    # Save the image to a BytesIO buffer and encode it as a base64 string
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return img_base64
