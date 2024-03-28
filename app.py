from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)

model = load_model('skin1.h5') 

target_size = (128, 128)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run("0.0.0.0", debug=False)
