import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model_path = 'C:\\Users\\shree sai\\Downloads\\asl_alphabet_test\\asl_alphabet_model_5.h5'
model = tf.keras.models.load_model(model_path)

# Map the predicted index to corresponding ASL alphabet letter
# ASL_ALPHABET = 'abcdefghijklmnopqrstuvwxyz '
ASL_ALPHABET = {
    0: 'A',
    1: 'B',
    2: 'C ',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N ',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
    26: 'del',
    27: 'nothing',
    28: 'space'
}

   

# Function to preprocess the image
# def preprocess_image(image):
#     img = image.resize((64, 64))  # Assuming your model accepts 64x64 images
#     img_array = np.array(img) / 255.0  # Normalize the pixel values
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array
# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match model input shape
    img_array = np.array(img) / 255.0  # Normalize the pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Function to make predictions
def predict_image(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index]
    return ASL_ALPHABET[predicted_index], confidence

# Streamlit app
def main():
    st.title("ASL Alphabet Gesture Recognition")

    st.write("""
    Upload an image of an ASL alphabet gesture, and the model will predict the corresponding letter.
    """)

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Make prediction
        gesture, confidence = predict_image(image)

        st.write(f"Prediction: {gesture}, Confidence: {confidence:.2f}")

if __name__ == '__main__':
    main()
