# AI Image Colorizer

A Python web application that uses a pretrained deep learning model
to add plausible colors to grayscale photographs.

## Features

- Upload JPG, JPEG, and PNG images
- Generate predicted colors using a pretrained model
- Compare the original and colorized images
- Download the result as a PNG image

## Technologies

- Python
- OpenCV
- NumPy
- Streamlit
- Pillow
- Pretrained image colorization neural network

## Run locally

Install the required packages:

    pip install -r requirements.txt

Start the application:

    streamlit run app.py

The application downloads the pretrained model files when required.

## Limitations

The predicted colors are estimates and may not match the original
photograph. Results depend on image content and model performance.

## Model reference

Research project: https://richzhang.github.io/colorization/

This project uses a pretrained colorization model for educational
purposes.
