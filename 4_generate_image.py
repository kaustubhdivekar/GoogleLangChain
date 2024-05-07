from dotenv import load_dotenv
load_dotenv() ## loading all the environment variables

import streamlit as st
import os
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load the OpenJourney model
model_id="prompthero/openjourney"
pipe=StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe=pipe.to("cpu")

##Function to generate an image using OpenJourney API
def generate_image(prompt):
    if prompt:
        with torch.no_grad(): #Ensure that no gradients are calculated
            image = pipe(prompt).images[0]
        return image
    else:
        return None

##initialize StreamLit app
st.set_page_config(page_title="OpenJourney Image Generation Demo")

st.header("OpenJourney Image Generation Application")
prompt=st.text_input("Enter a prompt for the image: ", key="prompt")

# uploaded_file=st.file_uploader("Choose an image ", type=["jpg","jpeg","png"])
# image=""
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

submit=st.button("Generate Image")

## TO GENERATE THE IMAGE

##If submit is clicked
if submit:
    generated_image=generate_image(prompt)
    if generated_image:    
        st.header("Generated Image")
        # Display the generated image
        st.image(generated_image, caption="Generated Image", use_column_width=True)
    else:
        st.write("Please enter a prompt to generate an image")