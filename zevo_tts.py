#!/usr/bin/env python3

import asyncio
import websockets
import numpy as np
import sounddevice as sd
import json
import wave
import ssl

class TTSRequestParams:
    def __init__(self, key, text='Acesta este un test.', output_filename="output_synth.wav",
                 voice='gia', audio_format="WAV_PCM", sample_rate=22050,
                 pace=1.0, pitch=0, bits_per_sample=16):
        self.key = key
        self.text = text
        self.output_filename = output_filename
        self.voice = voice
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.pace = pace
        self.pitch = pitch
        self.bits_per_sample = bits_per_sample



def construct_message(params):
    return json.dumps({
        "task": [
            {"text": params.text},
            {"voice": params.voice},
            {"key": params.key},
            {"pace": str(params.pace)},
            {"pitch": str(params.pitch)},
            {"audio_format": params.audio_format},
            {"bits_per_sample": str(params.bits_per_sample)},
            {"sample_rate": str(params.sample_rate)}
        ]
    })



async def text2speech(api_uri, params):
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    async with websockets.connect(api_uri, max_size=10000000, ssl=ssl_context) as websocket:
        message = construct_message(params)
        await websocket.send(message)
        result = await websocket.recv()
        if isinstance(result, str):
            print(result)
        else:
            return result



def play_audio(audio_data, sample_rate):
    audio = np.frombuffer(audio_data, dtype=np.int16)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()



def save_audio(audio_data, filename, sample_rate, bits_per_sample):
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(bits_per_sample // 8)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
    except Exception as e:
        print(f"File save error: {e}")



def perform_text_to_speech(api_key, text, voice):
    params = TTSRequestParams(key=api_key, text=text, voice=voice)
    audio_data = asyncio.run(text2speech('wss://api-tts.zevo-tech.com:2083', params))
    if audio_data:
        play_audio(audio_data, params.sample_rate)
        save_audio(audio_data, params.output_filename, params.sample_rate, params.bits_per_sample)



if __name__ == '__main__':
    api_key = 'icvsilab2026'
    voice = 'gia'
    perform_text_to_speech(api_key, "Acesta este un test", voice)

