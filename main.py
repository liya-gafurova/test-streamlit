import json
import time

import scipy.io.wavfile as scipy_wavfile
import torch
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import streamlit as st
from streamlit_jina import jina

from Speech.sst import quartznet
from Speech.tts import text2speech, vocoder, fs

app_options = {
    "sst": "Speech to Text",
    'tts': "Text to Speech",
    'ns': 'Resumes Semantic Search',
    'drawable_canvas': "Drawable Canvas",
}

add_selectbox = st.sidebar.selectbox(
    "Speech Processing:",
    tuple(app_options.values())
)


@st.cache
def transcribe_audio(server_file_path: str):
    transcribed_text = quartznet.transcribe(paths2audio_files=[server_file_path])
    return transcribed_text


def text_to_speech(text: str, export_wav_filepath):

    with torch.no_grad():
        start = time.time()
        wav, c, *_ = text2speech(text)
        wav = vocoder.inference(c)

    scipy_wavfile.write(export_wav_filepath, fs, wav.view(-1).cpu().numpy())
    return export_wav_filepath


def display_sst():
    st.write("""# Speech to Text""")

    uploaded_file = st.file_uploader("Upload Files", type=['wav'])
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        local_file_name = f"./current_{uploaded_file.name}"
        with open(local_file_name, 'wb') as new_wav:
            new_wav.write(uploaded_file.getvalue())

        st.write(file_details)

        transcription = transcribe_audio(local_file_name)[0]

        st.write(f"""
           For file *{uploaded_file.name}* recognized text: *{transcription}*
           """)


def display_tts():
    st.write("""# Text to Speech """)

    st.write("Enter the text that you would like to be voiced:")
    text = st.text_area(label='Text', height=50)
    if text is not '':
        result_wav = text_to_speech(text, './current_tts.wav')

        audio_file = open(result_wav, 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format='audio/ogg')


def display_drawable_canvas():
    st.write("""Canvas is defined once at the begging. Therefore, when loading a new background image, the canvas size does not change and the picture changes according to the shape of the canvas.""")

    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ( "rect", "freedraw", "line", "circle", "transform")
    )
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    image = Image.open(bg_image) if bg_image else None
    height  = image.height if image else 1200
    width = image.width if image else 900

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image= image,
        update_streamlit=realtime_update,
        # height= height,
        # width= width,
        drawing_mode=drawing_mode,
        key="canvas",

    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))

    with open('canvas_result.json', 'a') as json_result:
        json.dump(canvas_result.json_data, json_result)


def display_neural_search():
    s = jina.text_search(endpoint="http://localhost:12345/search")



if add_selectbox == app_options['sst']:
    display_sst()
elif add_selectbox == app_options['tts']:
    display_tts()
elif add_selectbox == app_options['ns']:
    display_neural_search()
elif add_selectbox == app_options['drawable_canvas']:
    display_drawable_canvas()
