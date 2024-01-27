import streamlit as st
from PIL import Image
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
import logging

# Set up custom logging format
log_format = "%(asctime)s [%(levelname)s]%(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)

# URLs of images
URLS = [
    "https://lh5.googleusercontent.com/p/AF1QipMIXre2FHhrB9k-bxyzhV0rXTsBuzgcrkjZi16d=w426-h240-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipOmu-WoAyUegSOwpkKLi25Lusu940Brh7o6G-I_=w426-h240-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipPmXNqC5XS8_ofbJBq4rE0h4HDRqAUZCBYWbhpa=w408-h306-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipM5nd0dj_qWSYMy2H0jxQxmY2KE3bXQ9nXwopN0=w408-h272-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipNcVzkZt5RpmBOSaujU_fXwvE9hyQJo_CRPF4l6=w408-h544-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipPRMQbhytsVGDJGMP_MOLCXM4rqWKlCCoq4QUX3=w408-h306-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipNpoaU6r1nXhHyXg4uSAVDPXGF8DtHW2JO9IIB_=w408-h408-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipMY_eqnfA5y-NfmqtfKjIv2Yu5R87Y2BKNMqcSh=w408-h544-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipNZfDZIgf5LFPku_3-sHEp319rnshTH43WAyA1-=w408-h272-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipOvptlyfXBZYx3Oj_Ecmc6v0VMQhTjvWOrwH4Lq=w408-h306-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipNNoh8JVj5AlLxq1cQqwFT16qWhp7GIyUoZ5yw8=w408-h272-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipPLGQIte6NLIdFbmboCkcroWThW67DX6e2N99L3=w408-h271-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipPJOJkpIX0rK5EQhYi7lrTfSgTDq6Nhuh6eEwsM=w408-h306-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipN_zoePbGbIbANDYYpEXHxlxPPwlamvZKqZSzMc=w408-h544-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipO6TLBLdU4YUhKc4GqB0-mNTa1CubFoFGnSZyom=w426-h240-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipPEPbvyumkfEG9vZeM6DsrYS0WGyMeK_SgOwkPz=w408-h304-k-no",
    "https://lh5.googleusercontent.com/p/AF1QipOZ1IENK7UHnLPCh0-msycG_-iBux0q30uxPfBQ=w426-h240-k-no"
]



@st.cache_resource
def get_models():
    """Load CLIP model and processor."""
    logging.info("Loading models")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_resource
def get_image_embeddings(_model, _processor):
    """
    Calculate image embeddings for the given URLs.

    Parameters:
    - _model (CLIPModel): CLIP model.
    - _processor (CLIPProcessor): CLIP processor.

    Returns:
    - numpy.ndarray: Image embeddings.
    """
    logging.info("Calculating image embeddings")
    images = [Image.open(requests.get(u, stream=True).raw) for u in URLS]
    inputs = _processor(text=["test"], images=images, return_tensors="pt", padding=True)
    outputs = _model(**inputs)
    image_embeds = outputs["image_embeds"].detach().numpy()
    return image_embeds

@st.cache_data
def read_image(url):
    """
    Read and return an image from the specified URL.

    Parameters:
    - url (str): URL of the image.

    Returns:
    - PIL.Image.Image: Image object.
    """
    logging.info(f"Reading image from {url}")
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

@st.cache_resource
def get_text_embeddings(string, _model, _processor):
    """
    Calculate text embeddings for the given input text.

    Parameters:
    - string (str): Input text.
    - _model (CLIPModel): CLIP model.
    - _processor (CLIPProcessor): CLIP processor.

    Returns:
    - numpy.ndarray: Text embeddings.
    """
    logging.info("Calculating input text embeddings")
    default_image_url = "https://fastly.picsum.photos/id/188/200/200.jpg?hmac=TipFoTVq-8WOmIswCmTNEcphuYngcdkCBi4YR7Hv6Cw"
    image = Image.open(requests.get(default_image_url, stream=True).raw)
    inputs = _processor(text=[string], images=image, return_tensors="pt", padding=True)
    outputs = _model(**inputs)
    text_embeds = outputs["text_embeds"].detach().numpy()
    return text_embeds

@st.cache_data
def app_info():
    """
    Display information and usage instructions for the app.
    """
    st.title("Streamlit Image Search App")

    st.markdown("""
        This is a Streamlit web application for searching and displaying similar images using the CLIP model.

        ## Overview

        The application utilizes the [CLIP model](https://openai.com/research/clip) from [Hugging Face](https://huggingface.co/)'s Transformers library to calculate image and text embeddings. Users can input text, and the app will find and display the top three most similar images from a predefined set of URLs.

        ## Usage

        ### **Text-Based Image Search:** (Model Tab)
        *Enter text to find images similar to the provided description.*

        The "Model" tab is like a smart photo searcher. You can tell it what you're looking for, and it will find the top three pictures that are most like your description.

        **Example:** If you type "alcoholic drinks," the model will show you three pictures that are most related to the idea of alcoholic drinks.

        Feel free to type in different things and see what pictures the model finds for you!

        ### **Show All Images:** (Show All Images Tab)
        *Explore all images used in the model*

        The "All Photos" tab is like a big photo album. It shows you all the pictures that the model knows about. You can use this tab to check out and enjoy all the images the model has seen.
    """)

def main():
    """
    Main function for Streamlit app.
    """
    urls = URLS
    model, processor = get_models()
    image_embeds = get_image_embeddings(model, processor)
    st.title("Demo image search")

    # Add a sidebar
    st.sidebar.title("Sidebar")

    # Add options to the sidebar
    option = st.sidebar.radio("Choose an option", ["Introduction", "Model", "Show all images"])

    # Show content based on selected options
    if option == "Introduction":
        app_info()
    elif option == "Model":
        text_input = st.text_input("Enter text to find similarity:")
        text_embeds = get_text_embeddings(text_input, model, processor)
        cosine_similarities = cosine_similarity(image_embeds, text_embeds)
        cosine_similarities_array = cosine_similarities.flatten()
        image_indices = np.argsort(cosine_similarities_array)[-3:][::-1]

        images = []
        for image_index in image_indices:
            image = read_image(urls[image_index])
            st.image(image, caption=f"Most similar image {image_index}", use_column_width=False)

    else:
        st.write("All images used in the model")
        for url in urls:
            image = read_image(url)
            st.image(image, caption=f"Image from {url}", use_column_width=True)

if __name__ == "__main__":
    main()


