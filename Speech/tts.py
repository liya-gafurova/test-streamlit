import time
import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model

lang = 'English'
fs = 22050 #@param {type:"integer"}
tag = 'kan-bayashi/ljspeech_fastspeech2' #@param ["kan-bayashi/ljspeech_tacotron2", "kan-bayashi/ljspeech_fastspeech", "kan-bayashi/ljspeech_fastspeech2", "kan-bayashi/ljspeech_conformer_fastspeech2"] {type:"string"}
vocoder_tag = "ljspeech_multi_band_melgan.v2" #@param ["ljspeech_parallel_wavegan.v1", "ljspeech_full_band_melgan.v2", "ljspeech_multi_band_melgan.v2"] {type:"string"}



d = ModelDownloader()
text2speech = Text2Speech(
    **d.download_and_unpack(tag),
    device="cuda",
    # Only for Tacotron 2
    threshold=0.5,
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2
    speed_control_alpha=1.0,
)
text2speech.spc2wav = None  # Disable griffin-lim
# NOTE: Sometimes download is failed due to "Permission denied". That is 
#   the limitation of google drive. Please retry after serveral hours.
vocoder = load_model(download_pretrained_model(vocoder_tag)).to("cuda").eval()
vocoder.remove_weight_norm()

# decide the input sentence by yourself
# print(f"Input your favorite sentence in {lang}.")
# x = input()

# # synthesis
# with torch.no_grad():
#     start = time.time()
#     wav, c, *_ = text2speech(x)
#     wav = vocoder.inference(c)
# rtf = (time.time() - start) / (len(wav) / fs)
# print(f"RTF = {rtf:5f}")
#
# # let us listen to generated samples
# from IPython.display import display, Audio
# f = display(Audio(wav.view(-1).cpu().numpy(), rate=fs))
#
#
# import scipy.io.wavfile
# scipy.io.wavfile.write("karplus.wav", fs, wav.view(-1).cpu().numpy())
