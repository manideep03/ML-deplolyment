from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
import os
import urllib.request

if os.psth.exists("model_V1.h5"):
    continue
else:
    urllib.request.urlretrieve("https://firebasestorage.googleapis.com/v0/b/books-68839.appspot.com/o/model_V1.h5?alt=media&token=00176109-597e-4335-808e-f5b632a07f87", "model_V1.h5")


model = None

app = flask.Flask(__name__)


def loadmodel():
    global model
    model = load_model('model_V1.h5')


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


@app.route("/predict", methods=["POST"])
def predict():
    c = ['Link Down', 'Link Up']

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            image = prepare_image(image, target=(224, 224))

            preds = model.predict(image)

            ans = np.argmax(preds)

    return str(c[ans])


if __name__ == "__main__":
    print(("* Loading your model and Flask server..."
           "please wait let the server start"))
    loadmodel()
    app.run()
