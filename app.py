from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

model = load_model('skin1.h5') 

target_size = (128, 128)

@app.route("/", methods=["POST"])
def main():
    try:
        if 'image' in request.files:
            image_file = request.files['image']

            image_content = image_file.read()


            img = image.load_img(io.BytesIO(image_content), target_size=target_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            preds = model.predict(img_array)

            i = np.argmax(preds[0])
            label_to_class = {'Actinic keratosis': 0,
                      'Basal cell carcinoma': 1,
                      'Benign keratosis': 2,
                      'Dermatofibroma': 3,
                      'Melanocytic nevus':4,
                      'Melanoma':5,
                      'Squamous cell carcinoma':6,
                      'Vascular lesion':7}

            class_to_label = {v: k for k, v in label_to_class.items()}

            label = class_to_label[i]


            response = {
                'class': label,
                'Probability': max(preds[0]) * 100
            }

            return jsonify(response)

        else:
            return {"error": "No 'image' key found in request files"}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)
