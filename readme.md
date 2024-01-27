# Streamlit Image Search App

This is a Streamlit web application for searching and displaying similar images using the CLIP model.

## Overview

The application utilizes the CLIP model from Hugging Face's Transformers library to calculate image and text embeddings. Users can input text, and the app will find and display the top three most similar images from a predefined set of URLs.

## Features

- **Text-Based Image Search:** Enter text to find images similar to the provided description.
- **Display Similar Images:** View the top three most similar images in the same row.
- **Show All Images:** Explore all images used in the model.

## Getting Started

1. Install the required dependencies:

    ```bash
    pip install -r requirements
    ```

2. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

3. Open your web browser and go to `http://localhost:8501` to interact with the app.

## Dependencies

- Streamlit
- Transformers (Hugging Face)
- Pillow
- NumPy
- scikit-learn

## Usage

- Choose between the "Model" and "Show all images" options in the sidebar.
- For the "Model" option, enter text and see the top three most similar images.
- For the "Show all images" option, view all images used in the model.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)

