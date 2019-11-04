from flask import Flask, request, jsonify
import json
from keras.models import Model, load_model
import urllib.request
import io
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image


app = Flask(__name__)


model = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
model_new = Model(model.input, model.layers[-2].output)


def preprocess_img(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def encode_image(f):
    img = preprocess_img(f)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector


word_to_idx = {}
idx_to_word = {}

with open("word_to_idx.txt", "r") as f:
    word_to_idx = eval(f.read())
with open("idx_to_word.txt", "r") as f:
    idx_to_word = eval(f.read())


def predict_caption(model1, photo):
    in_text = "startseq"
    for i in range(80):
        sequence = [word_to_idx[w]
                    for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=80, padding="post")
        ypred = model1.predict([photo, sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text += " " + word
        if word == "endseq":
            break

    final_caption = in_text.split()[1:-1]
    final_caption = " ".join(final_caption)
    return final_caption


@app.route("/", methods=["POST"])
def index():
    with urllib.request.urlopen(request.json["image"]) as url:
        asdw = io.BytesIO(url.read())
    model1 = load_model("./model.h5")
    return jsonify(
        captions=predict_caption(model1, encode_image(asdw).reshape((1, 2048)))
    )
