#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

#Loading the Model
model = load_model('dog_breed.h5')

#Name of Classes
CLASS_NAMES = ['Scottish Deerhound', 'Maltese Dog', 'Bernese Mountain Dog']

st.title("Dog Breed Prediction (Limited to Specific Breeds)")
st.markdown("This app predicts the breed of a dog. **Currently, it can only distinguish between the following breeds:**")
st.markdown("- Scottish Deerhound")
st.markdown("- Maltese Dog")
st.markdown("- Bernese Mountain Dog")
st.markdown("Upload an image of the dog to proceed.")

#Uploading the dog image
dog_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])
submit = st.button('Predict')

#On predict button click
if submit:
    if dog_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Check if the file is a valid image
        if opencv_image is None:
            st.error("Invalid image file. Please upload a valid image.")
        else:
            # Displaying the image
            st.image(opencv_image, channels="BGR")

            # Resizing the image
            opencv_image = cv2.resize(opencv_image, (224, 224))

            # Convert image to 4 Dimension
            opencv_image = opencv_image.reshape((1, 224, 224, 3))

            # Make Prediction
            Y_pred = model.predict(opencv_image)

            # Display the prediction
            st.title(f"The Dog Breed is {CLASS_NAMES[np.argmax(Y_pred)]}")
