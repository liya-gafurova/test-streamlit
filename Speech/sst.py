# NeMo's "core" package
import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr

# This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

files = ['/home/lia/Desktop/wavs/audio_dir/speaker1/uttr1.wav',
         "/home/lia/PycharmProjects/IPR/IPR2/streamlit_project/Speech/karplus.wav"]

#
#
# for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
#   print(f"Audio in {fname} was recognized as: {transcription}")