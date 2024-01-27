import streamlit as st
from PIL import Image
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
import logging


# Crear un formato personalizado
log_format = "%(asctime)s %(levelname)s:%(name)s:%(message)s"
date_format = "%Y-%m-%d %H:%M:%S"
# formatter = logging.Formatter(log_format, datefmt=date_format)
logging.basicConfig(level=logging.INFO, format=log_format)

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


# Cargar modelo y procesador CLIP
@st.cache_resource
def get_models():
    logging.info("Loading models")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_resource
def get_image_embeddings(_model, _processor):
    logging.info("Calculating image embeddings")
    images = [Image.open(requests.get(u, stream=True).raw) for u in URLS]

    inputs = _processor(text=["test"], images=images, return_tensors="pt", padding=True)

    outputs = _model(**inputs)
    image_embeds = outputs["image_embeds"].detach().numpy()
    return image_embeds

@st.cache_data
def read_image(url):
    logging.info(f"Reading image from {url}")
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

@st.cache_resource
def get_text_embeddings(string, _model, _processor):
    logging.info("Calculating input text embeddings")
    # default image. The modelo doesn't work without an image input
    url = "https://fastly.picsum.photos/id/188/200/200.jpg?hmac=TipFoTVq-8WOmIswCmTNEcphuYngcdkCBi4YR7Hv6Cw"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = _processor(text=[string], images=image, return_tensors="pt", padding=True)
    outputs = _model(**inputs)
    text_embeds = outputs["text_embeds"].detach().numpy()
    return text_embeds

def main():

    urls = URLS
    model, processor = get_models()
    image_embeds = get_image_embeddings(model, processor)
    st.title("Demo image search")

    # Añadir una barra lateral
    st.sidebar.title("sidebar")

    # Agregar opciones a la barra lateral
    option = st.sidebar.radio("Choose an option", ["Model", "Show all images"])

    # Mostrar el contenido dependiendo de las opciones seleccionadas
    if option=="Model":
        # Entrada de texto y botón para procesar
        text_input = st.text_input("Ingrese un texto para buscar similitud:")

        text_embeds = get_text_embeddings(text_input, model, processor)

        cosine_similarities = cosine_similarity(image_embeds, text_embeds)
        cosine_similarities_array = cosine_similarities.flatten()

        # image_index = np.argmax(cosine_similarities)
        image_indexs = np.argsort(cosine_similarities_array)
        print(np.argsort(cosine_similarities_array))
        print(cosine_similarities_array)

        for image_index in image_indexs[-3:][::-1]:

            image = read_image(urls[image_index])
            st.image(image, caption="Most similar image {image_index}", use_column_width=True)
        st.write("Similitud del coseno entre texto e imágenes:")
        st.table(cosine_similarities)

    else:
        st.write("All images used in model")
        for url in urls:
            image = read_image(url)

            # Mostrar la imagen en Streamlit
            st.image(image, caption=f"Image from {url}", use_column_width=True)



if __name__ == "__main__":
    main()