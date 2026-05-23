#!/usr/bin/env python3

import asyncio
import websockets
import json
import sys
import ssl

if len(sys.argv) != 4 and len(sys.argv) != 5 and len(sys.argv) != 6:
    print("Usage 1: STT.py <filename> <license> <sample_rate> <(optional) transcription_domain>")
    print("<filename> audio file path")
    print("<license> Zevo Live STT API key")
    print("<sample_rate> audio data sample rate")
    print("(optional) <transcription_domain>")
    print(" ")
    print("Usage 2: STT.py <filename> <license> <sample_rate> <transcription_domain> <phrases>")
    print('Example usage with "phrases": \nSTT.py audio-file.wav <license> 16000 ro-RO_phrases \'["zero", "unu", "doi", "trei", "patru", "cinci", "şase", "şapte", "opt", "nouă",  "<UNK>"]\'')
    print('Example usage with "phrases_v2": \nSTT.py audio-file.wav <license> 16000 ro-RO_phrases \'[{"phrase": "<UNK>", "weight": 10}, {"phrase": "unu", "weight": 5}, {"phrase": "doi trei", "weight": 5}]\'')

    exit(0)

filename = sys.argv[1]
key = sys.argv[2]
sample_rate = sys.argv[3]
transcription_domain = None

server_address = "wss://live-transcriber.zevo-tech.com:2053"

def check_phrases_format(phrases):
    try:
        data = json.loads(phrases)
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            return 'v1'
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return 'v2'
        else:
            return None
    except json.JSONDecodeError:
        return None


if len(sys.argv) == 5 or len(sys.argv) == 6:
    transcription_domain = sys.argv[4]

if transcription_domain == "ro-RO_phrases":
    phrases = sys.argv[5]
    phrases_mode = check_phrases_format(phrases)

async def speechtotext(uri):
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    
    async with websockets.connect(uri, ssl=ssl_ctx) as websocket:
        if len(sys.argv) == 4:
            message = '{"config": {"key": "' + str(key) + '", "sample_rate": "' + str(sample_rate) + '"}}'
        else:
            if len(sys.argv) == 5:
                message = '{"config": {"key": "' + str(key) + '", "sample_rate": "' + str(sample_rate) + '", "domain": "' + str(transcription_domain) +'"}}'
            else:
                if len(sys.argv) == 6:
                    if phrases_mode == 'v1':
                            message = '{"config": {"key": "' + str(key) + '", "sample_rate": "' + str(sample_rate) + '", "domain": "' + str(transcription_domain) + '", "phrases": ' + str(phrases) +'}}'
                    if phrases_mode == 'v2':
                        message = '{"config": {"key": "' + str(key) + '", "sample_rate": "' + str(sample_rate) + '", "domain": "' + str(transcription_domain) + '", "phrases_v2": ' + str(phrases) +'}}'
        await websocket.send(message)
        print(await asyncio.wait_for(websocket.recv(), timeout=120.0))
        wf = open(filename, "rb")
        while True:
            data = wf.read(4096)

            if len(data) == 0:
                break

            await websocket.send(data)
            result_text = await asyncio.wait_for(websocket.recv(), timeout=120.0)
            print(result_text)
            if "message" in result_text:
                exit()

        await websocket.send('{"eof" : 1}')
        print (await asyncio.wait_for(websocket.recv(), timeout=120.0))


asyncio.run(speechtotext(server_address))

