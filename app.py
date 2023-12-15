from flask import Flask, request, render_template, send_file
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
#from utils.prompt_making import make_prompt
import langid
langid.set_languages(['en', 'zh', 'ja'])
import logging
import os
import nltk
nltk.data.path = nltk.data.path + [os.path.join(os.getcwd(), "nltk_data")]
import soundfile as sf
import torch
import torchaudio
import numpy as np

from data.collation import get_text_token_collater
from utils.g2p import PhonemeBpeTokenizer
from descriptions import *
from macros import *
from examples import *
import re
import whisper
from vocos import Vocos
import multiprocessing
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
thread_count = multiprocessing.cpu_count()

print("Use",thread_count,"cpu cores for computing")

torch.set_num_threads(thread_count)
torch.set_num_interop_threads(thread_count)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)
if torch.backends.mps.is_available():
    device = torch.device("mps")
codec = AudioTokenizer(device)
#model
preload_models()

if not os.path.exists("./whisper/"): os.mkdir("./whisper/")
try:
    whisper_model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper")).cpu()
except Exception as e:
    logging.info(e)
    raise Exception(
        "\n Whisper download failed or damaged, please go to "
        "'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt'"
        "\n manually download model and put it to {} .".format(os.getcwd() + "\whisper"))

def transcribe_one(model, audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False if device == torch.device("cpu") else True, sample_len=150)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)

    text_pr = result.text
    if text_pr.strip(" ")[-1] not in "?!.,。，？！。、":
        text_pr += "."
    return lang, text_pr

def make_prompt(name, audio_prompt_path, transcript=None):
    global model, text_collater, text_tokenizer, codec
    wav_pr, sr = torchaudio.load(audio_prompt_path)
    # check length
    if wav_pr.size(-1) / sr > 15:
        raise ValueError(f"Prompt too long, expect length below 15 seconds, got {wav_pr / sr} seconds.")
    if wav_pr.size(0) == 2:
        wav_pr = wav_pr.mean(0, keepdim=True)
    text_pr, lang_pr = make_transcript(name, wav_pr, sr, transcript)

    # tokenize audio
    encoded_frames = tokenize_audio(codec, (wav_pr, sr))
    audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

    # tokenize text
    phonemes, langs = text_tokenizer.tokenize(text=f"{text_pr}".strip())
    text_tokens, enroll_x_lens = text_collater(
        [
            phonemes
        ]
    )

    message = f"Detected language: {lang_pr}\n Detected text {text_pr}\n"

    # save as npz file
    save_path = os.path.join("./customs/", f"{name}.npz")
    np.savez(save_path, audio_tokens=audio_tokens, text_tokens=text_tokens, lang_code=lang2code[lang_pr])
    logging.info(f"Successful. Prompt saved to {save_path}")


def make_transcript(name, wav, sr, transcript=None):

    if not isinstance(wav, torch.FloatTensor):
        wav = torch.tensor(wav)
    if wav.abs().max() > 1:
        wav /= wav.abs().max()
    if wav.size(-1) == 2:
        wav = wav.mean(-1, keepdim=False)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    assert wav.ndim and wav.size(0) == 1
    if transcript is None or transcript == "":
        logging.info("Transcript not given, using Whisper...")
        global whisper_model
        if whisper_model is None:
            whisper_model = whisper.load_model("medium", download_root=os.path.join(os.getcwd(), "whisper"))
        whisper_model.to(device)
        torchaudio.save(f"./prompts/{name}.wav", wav, sr)
        lang, text = transcribe_one(whisper_model, f"./prompts/{name}.wav")
        lang_token = lang2token[lang]
        text = lang_token + text + lang_token
        os.remove(f"./prompts/{name}.wav")
        whisper_model.cpu()
    else:
        text = transcript
        lang, _ = langid.classify(text)
        lang_token = lang2token[lang]
        text = lang_token + text + lang_token

    torch.cuda.empty_cache()
    return text, lang

def textsplit(text):
    raw_text = text
    new_text = re.sub(r'，', ',', raw_text)
    new_text = re.sub(r'。', '.', new_text)
    # 找出所有中文
    chinese_matches = re.findall(r'[\u4e00-\u9fff]+', new_text)

    # 分離中英混合的字串
    splitted_text = re.split(r'[\u4e00-\u9fff]+', new_text)

    # 印出分開的結果
    result = []
    for i in range(len(splitted_text)):
        result.append(splitted_text[i])
        if i < len(chinese_matches):
            result.append(chinese_matches[i])

    # 移除空字串
    result = [seg.strip() for seg in result if seg.strip()]

    # 移除開頭和結尾的標點符號
    for i, seg in enumerate(result):
        if seg.startswith(",") or seg.startswith("."):
            result[i] = seg[1:]
        if seg.endswith(",") or seg.endswith("."):
            result[i] = seg[:-1]

    return result

app = Flask(__name__)
@app.route('/json2', methods=['POST'])
def JSON():
    if request.method == "POST":
        json_data = request.json
        text = json_data.get('text')
        split_text = textsplit(text)
        if json_data:
            print("generating audio...")
            audio_array = generate_audio(split_text, prompt='test')
            print('saving file...')
            write_wav("text.wav", SAMPLE_RATE, audio_array)
            filename = "text.wav"
            print("sending...")
            return send_file(filename, as_attachment=True)
        else:
            print('No received')
            return "No received"

@app.route('/upload', methods=['POST'])
def upload():
    print("start upload")
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No selected file"
        file = request.files['file']
        if 'file':
            print('received file')
            file.save('received_file.wav')
            make_prompt("test", "received_file.wav")
        return "File successfully uploaded"
        
@app.route('/text', methods=['POST'])
def test():
    if request.method=='POST':
        textinput = request.form['text']
        print(textinput)
        split_text = textsplit(textinput)
        audio_array = generate_audio(split_text, prompt='test')
        write_wav("text.wav", SAMPLE_RATE, audio_array)
        filename = "text.wav"
        print("sending...")
    return send_file(filename, as_attachment=True)

@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)