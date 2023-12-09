
import streamlit as st
import pandas as pd
from predict_number import predict_number, preprocess_image, load_model, LeNet


st.title('Handwritten Digit Recognition')
st.header('This is a simple web app to recognize hand written digits. The ML model is trained on the MNIST dataset and the model architecture is Lenet.')
st.divider()
st.subheader('Please upload an image of a handwritten digit or upload an image from camera.')
st.divider()
file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
#st.write(type(file))
st.divider()
image = st.camera_input("Take a picture from my camera.")
st.divider()
button = st.button('PREDICT')

if button:
    st.subheader('You pressed the button')
    number = predict_number(file)
    st.write('The predicted number is:', number)
