from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import cv2

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# import h1

try:
    import shutil

    shutil.rmtree("uploaded / image")
    print()
except:
    pass

# Prediction function
def classify_brain(model, img_path, labels=["Normal", "Tumor"]):
    # Load image and convert to array
    img = imread(img_path)
    img = resize(img, (256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction using the model
    prediction = model.predict(img_array)

    if prediction > 0.5:
        pred = labels[1]
    else:
        pred = labels[0]

    # plt.imshow(img_array[0])
    # plt.title("Predicted: "+ pred)

    # Return predicted label (0 or 1)
    return pred


def predict_breast(breast_classifier, image_path):
    img = imread(image_path)
    img = resize(img, (256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    pred = breast_classifier.predict(img_array)
    labels = ["Benign", "Malignant", "Normal"]

    return labels[np.argmax(pred)]


def f1score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return K.mean(f1)


model1 = tf.keras.models.load_model(
    r"brain_breast_model.h5", custom_objects={"f1score": f1score}, compile=False
)

brain_classifier = tf.keras.models.load_model(
    "brain_classifier.h5", custom_objects={"Adam": Adam}, compile=False
)

breast_classifier = tf.keras.models.load_model(
    "breast_classifier.h5", custom_objects={"Adam": Adam}, compile=False
)

# model = tf.keras.models.load_model("brain_breast_model.h5")
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploaded\\image"


@app.route("/")
def upload_f():
    return render_template("upload.html")


import keras

# keras.preprocessing
def finds():

    f = request.files["file"]
    img = tf.keras.preprocessing.image.load_img(
        os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename)),
        target_size=(256, 256),
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    images = np.vstack([x])

    pred = model1.predict_generator(images)
    print(pred)
    if pred[0][0] < 0.5:
        p = "Brain"
    else:
        p = "Breast"

    if p == "Brain":
        pred1 = classify_brain(
            brain_classifier,
            os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename)),
        )
        if pred1 == "Normal":
            filename = os.path.join(
                "static\images", "predections", secure_filename(f.filename)
            )

            cv2.imwrite(filename, img)

            return "Normal Brain", filename
        else:
            filename = os.path.join(
                "static\images", "predections", secure_filename(f.filename)
            )

            cv2.imwrite(filename, img)

            return "Brain Tumor", filename
        #     masked= h1.segment_brain(os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename)))
        #     filename = os.path.join(
        #         "static\images", "predections", secure_filename(f.filename)
        #     )

        #     cv2.imwrite(filename, masked)

        #     return "Brain Tumor", filename
    else:
        pred2 = predict_breast(
            breast_classifier,
            os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename)),
        )
        if pred2 == "Normal":
            filename = os.path.join(
                "static\images", "predections", secure_filename(f.filename)
            )

            cv2.imwrite(filename, img)

            return "Normal Breast", filename

        elif pred2 == "Benign":
            filename = os.path.join(
                "static\images", "predections", secure_filename(f.filename)
            )

            cv2.imwrite(filename, img)

            return "Benign Breast", filename
        else:
            filename = os.path.join(
                "static\images", "predections", secure_filename(f.filename)
            )

            cv2.imwrite(filename, img)

            return "Malignant Breast", filename


@app.route("/uploader", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        f = request.files["file"]
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename)))
        val, img = finds()
        os.remove(
            os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
        )
        return render_template("pred.html", ss=val, img_path=img)


if __name__ == "__main__":
    app.run()
