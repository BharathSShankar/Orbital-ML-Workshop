from PIL import Image
import io
import numpy as np
import flask
import tensorflow as tf
from flask import Flask, request, jsonify

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((32, 32))
    img = np.array(img)
    return img

model = tf.keras.models.load_model('Final_model.h5')

def predict_result(img):
    return "Cat" if model.predict(img) > 0.5 else "Dog"

app = Flask(__name__)

@app.route("/predict", method = "POST")
def predict_img():
    if "image" not in request.files:
        return "ERROR: NO IMAGE PROVIDED :<"
    
    image = request.files.get("image")

    if not image:
        return 
    
    image = prepare_image(image)
    
    return jsonify(prediction=predict_result(image))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')