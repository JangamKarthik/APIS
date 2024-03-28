from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

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
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  

            predictions = model.predict(img_array)

            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index]

            class_names = ['Actinic keratosis','Basal cell carcinoma','Benign keratosis', 'Dermatofibroma','Melanocytic nevus', 'Melanoma' ,'Squamous cell carcinoma','Vascular lesion']
            predicted_class = class_names[class_index]


            response = {
                'class': predicted_class,
                'confidence': float(confidence)
            }

            return jsonify(response)

        else:
            return {"error": "No 'image' key found in request files"}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)
