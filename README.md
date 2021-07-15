# test-streamlit

Project works on Python3.7.9  
Project has 4 main functionalities. Each functionality has its own requirements. Some of them are described in Speech/requirement files. Additional requirements are defined in final_requirements.txt.

## Speech to Text

NeMo model used for STT task. 
See [ColabNotebook](https://colab.research.google.com/drive/14YRN708RFDOxbT67YoIvfGLtMvKkzUdU?usp=sharing) for more information.
Takes wav file as an input, gives text transcription as an output.   
For downloading requirements see Speech/requirements.txt

## Text to Speech

ESPNet framework with FastSpeech2 and MalGan models is used. See [ColabNotebook](https://colab.research.google.com/drive/1k-01TfI60J8EVL_JF7SPwoWdutPg2L4q?usp=sharing) for more experiments.  
Takes text as input and gives wav file with voiced text as an output.  
For downloading requirements see Speech/requirements2.txt

## Text Neural Search

Jina search engine is used for neural search.  
*streamlit-jina* component used for wrapping Jina engine and 
convenient access.

For usage:
1. start ns.py -- launches Jina Engine with prepared data.
2. streamlit run main.py

## Drawable Canvas

Testing of *streamlit-drawable-canvas* component.   
This component helps you select certain areas on a photo or document and save the coordinates of the selected areas. The disadvantage of this component is that the canvas size is determined only at the start of the project and does not change in accordance with the newly uploaded image.