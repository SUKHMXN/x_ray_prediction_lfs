# from __future__ import division, print_function
# import os
# import numpy as np
# from flask import Flask, render_template, request, flash
# from werkzeug.utils import secure_filename
# from keras.models import load_model
# from keras.preprocessing import image
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import sys
# import locale

# # Set the default encoding to UTF-8
# os.environ["PYTHONIOENCODING"] = "utf-8"
# locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

# def preprocess_image(img, target_size=(224, 224)):
#     img_array = image.img_to_array(img)
#     img_array = img_array / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array, img  

# img_path = None



# # Initialize Flask app
# app = Flask(__name__, static_url_path='/static')
# app.config["SECRET_KEY"] = "secret_key"

# # Load the Keras model
# pipeline_model = load_model("chest_xray_model.h5")

# # Ensure an upload folder exists
# UPLOAD_FOLDER = "uploads"
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# @app.route("/", methods=["GET", "POST"])
# def Home():
#     prediction = None  # Initialize prediction variable
#     if request.method == "POST":
#         try:
#             # Check if a file is uploaded
#             file = request.files["file"]
#             if file:
#                 # Save uploaded file
#                 filename = secure_filename(file.filename)
#                 filepath = os.path.join(UPLOAD_FOLDER, filename)
#                 file.save(filepath)
#                 img = image.load_img(filepath, target_size=(224, 224))  # Update size if required by the model
#                 processed_image, resized_image = preprocess_image(img)
#                 preds = pipeline_model.predict(processed_image)[0][0]
#                 if preds>=0.5:
#                     prediction="Pnemonia"
#                 else:
#                     prediction="Not Pnemonia"
#                 print(preds)
#             else:
#                 flash("No file uploaded. Please upload an image file.", "danger")

#         except Exception as e:
#             flash(f"An error occurred during prediction: {e}", "danger")
#             prediction = None

#     return render_template("index.html", prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("chest_xray_model.h5")

# Preprocess image for prediction
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match the input size of the model
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    img = image.load_img(file, target_size=(224, 224))
    processed_img = preprocess_image(img)
    
    # Get prediction
    prediction = model.predict(processed_img)[0][0]
    result = "Pneumonia" if prediction >= 0.5 else "Not Pneumonia"
    
    return jsonify({"prediction": result, "confidence": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
