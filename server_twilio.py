"""
Server Quart pentru integrarea Twilio cu robotul telefonic.
Twilio trimite audio prin WebSocket, serverul proceseaza cu Zevo STT + GPT + Zevo TTS.

Rulare:
    python3 server_twilio.py
"""

import os
import json
import asyncio
import audioop
import base64
import ssl
import time
import websockets

from quart import Quart, request, Response

from openai import OpenAI
from date_robot import (
    INTENTII_DOMENII, REPLICI_IDENTIFICARE, REPLICA_INCHEIERE,
    REPLICA_OPERATOR, NUME_DOMENII, TIP_REZUMAT,
    NIVELURI_DIFICULTATE, DETALII_NECESARE, SALUT_DOMENII
)
from solutii_predefinite import SOLUTII_PREDEFINITE
from robot_telefonic_v2 import (
    normalizeaza, extrage_intentie, extrage_satisfactie,
    client_vrea_sa_incheie, alege_dificultate,
    extrage_intentie_llm, genereaza_replica_robot,
    verifica_detalii_complete, analiza_finala, call_llm
)

# ─── CONFIGURARE ──────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
NGROK_DOMENIU  = os.environ.get("NGROK_URL", "").replace("https://", "").replace("http://", "")

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN", "")

ZEVO_STT_KEY    = "icvsilab2026"
ZEVO_STT_DOMAIN = "ro-RO_general-2026.1"
ZEVO_STT_SERVER = "wss://live-transcriber.zevo-tech.com:2053"
ZEVO_TTS_SERVER = "wss://api-tts.zevo-tech.com:2083"
ZEVO_TTS_VOICE  = "gia"

DOMENIU_DEFAULT = os.environ.get("DOMENIU", "banking")

app = Quart(__name__)
client_gpt = OpenAI(api_key=OPENAI_API_KEY)
conversatii_active = {}


# ─── ZEVO TTS ─────────────────────────────────────────────────────────────────

async def zevo_tts_bytes(text):
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    params = json.dumps({"task": [
        {"text": text}, {"voice": ZEVO_TTS_VOICE}, {"key": ZEVO_STT_KEY},
        {"pace": "1.0"}, {"pitch": "0"}, {"audio_format": "WAV_PCM"},
        {"bits_per_sample": "16"}, {"sample_rate": "8000"}
    ]})
    try:
        async with websockets.connect(ZEVO_TTS_SERVER, ssl=ssl_ctx, max_size=10_000_000) as ws:
            await ws.send(params)
            result = await ws.recv()
            return result if isinstance(result, bytes) else b""
    except Exception as e:
        print(f"[TTS EROARE]: {e}")
        return b""


async def tts_to_mulaw(text):
    wav_bytes = await zevo_tts_bytes(text)
    if not wav_bytes:
        return b""
    pcm_data = wav_bytes[44:]
    return audioop.lin2ulaw(pcm_data, 2)


# ─── ZEVO STT ─────────────────────────────────────────────────────────────────

async def zevo_stt_transcrie(audio_chunks_mulaw):
    if not audio_chunks_mulaw:
        return ""

    print(f"  [STT] Incepe transcrierea: {len(audio_chunks_mulaw)} chunks")

    mulaw_concat = b"".join(audio_chunks_mulaw)
    pcm_8k = audioop.ulaw2lin(mulaw_concat, 2)
    pcm_16k = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)[0]

    config_msg = json.dumps({"config": {
        "key": ZEVO_STT_KEY, "sample_rate": "16000", "domain": ZEVO_STT_DOMAIN
    }})
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    chunk_size = 4096
    sleep_per_chunk = chunk_size / (16000 * 2)
    rezultate = []
    partial_curent = ""

    try:
        async with websockets.connect(ZEVO_STT_SERVER, ssl=ssl_ctx) as ws:
            await ws.send(config_msg)
            await asyncio.wait_for(ws.recv(), timeout=10)

            offset = 0
            while offset < len(pcm_16k):
                chunk = pcm_16k[offset:offset + chunk_size]
                await ws.send(chunk)
                result = await asyncio.wait_for(ws.recv(), timeout=30)
                if "message" in result:
                    break
                try:
                    parsed = json.loads(result)
                    if "text_pp" in parsed and parsed["text_pp"].strip():
                        rezultate.append(parsed["text_pp"].strip())
                        partial_curent = ""
                    elif "text" in parsed and parsed["text"].strip() and "partial" not in parsed:
                        rezultate.append(parsed["text"].strip())
                        partial_curent = ""
                    elif "partial" in parsed and parsed["partial"].strip():
                        partial_curent = parsed["partial"].strip()
                except Exception:
                    pass
                offset += chunk_size
                await asyncio.sleep(sleep_per_chunk)

            if partial_curent and partial_curent not in " ".join(rezultate):
                rezultate.append(partial_curent)

            await ws.send('{"eof" : 1}')
            eof_resp = await asyncio.wait_for(ws.recv(), timeout=30)
            try:
                parsed = json.loads(eof_resp)
                if "text_pp" in parsed and parsed["text_pp"].strip():
                    rezultate.append(parsed["text_pp"].strip())
                elif "partial" in parsed and parsed["partial"].strip():
                    p = parsed["partial"].strip()
                    if p not in " ".join(rezultate):
                        rezultate.append(p)
            except Exception:
                pass
    except Exception as e:
        print(f"[STT EROARE]: {e}")

    return " ".join(rezultate)


# ─── STAREA CONVERSATIEI ─────────────────────────────────────────────────────

class StareConversatie:
    def __init__(self, domeniu=DOMENIU_DEFAULT):
        self.domeniu = domeniu
        self.dificultate = alege_dificultate()
        self.replici = []
        self.identificare_colectata = {}
        self.intentie = None
        self.faza = "identificare_client"
        self.tururi_fara_progres = 0
        self.stream_sid = None

    def adauga(self, rol, text):
        self.replici.append({"rol": rol, "text": text})

    def get_dialog(self):
        return "\n".join(f"{r['rol'].upper()}: {r['text']}" for r in self.replici)


# ─── LOGICA CONVERSATIE ───────────────────────────────────────────────────────

def proceseaza_replica_client(stare, text_client):
    if not text_client or not text_client.strip():
        return None, False

    stare.adauga("client", text_client)

    if stare.faza == "identificare_client":
        replici_id = REPLICI_IDENTIFICARE.get(stare.domeniu,
                     ["Va rog sa imi spuneti numele complet.", "Cu ce va pot ajuta?"])
        nr_replici_id = len(replici_id) - 1

        if len(stare.identificare_colectata) < nr_replici_id:
            idx = len(stare.identificare_colectata)
            stare.identificare_colectata[f"detaliu_{idx}"] = text_client

        nr_raspunsuri = len(stare.identificare_colectata)
        if nr_raspunsuri < nr_replici_id:
            replica = replici_id[nr_raspunsuri]
        else:
            replica = replici_id[-1]
            stare.faza = "identificare_intentie"

        stare.adauga("operator", replica)
        return replica, False

    if stare.faza == "identificare_intentie":
        stare.intentie = extrage_intentie_llm(stare.get_dialog(), stare.domeniu)

        if stare.intentie == "alta_solicitare":
            stare.tururi_fara_progres += 1
            if stare.tururi_fara_progres >= 2:
                stare.adauga("operator", REPLICA_OPERATOR)
                return REPLICA_OPERATOR, True
            replica = genereaza_replica_robot(
                stare.get_dialog(), stare.domeniu,
                stare.intentie, stare.dificultate, "identificare_intentie"
            )
        else:
            stare.tururi_fara_progres = 0
            detalii_nec = DETALII_NECESARE.get(stare.intentie, [])
            stare.faza = "colectare_detalii" if detalii_nec else "solutie"
            replica = genereaza_replica_robot(
                stare.get_dialog(), stare.domeniu,
                stare.intentie, stare.dificultate, stare.faza
            )
            if stare.faza == "solutie":
                stare.faza = "confirmare"

        stare.adauga("operator", replica)
        return replica, False

    if stare.faza == "colectare_detalii":
        if verifica_detalii_complete(stare.get_dialog(), stare.intentie):
            stare.faza = "solutie"
        else:
            replica = genereaza_replica_robot(
                stare.get_dialog(), stare.domeniu,
                stare.intentie, stare.dificultate, "colectare_detalii"
            )
            stare.adauga("operator", replica)
            return replica, False

    if stare.faza == "solutie":
        if SOLUTII_PREDEFINITE.get(stare.intentie) == "OPERATOR":
            stare.adauga("operator", REPLICA_OPERATOR)
            return REPLICA_OPERATOR, True

        replica = genereaza_replica_robot(
            stare.get_dialog(), stare.domeniu,
            stare.intentie, stare.dificultate, "solutie"
        )
        stare.adauga("operator", replica)
        stare.faza = "confirmare"
        return replica, False

    if stare.faza == "confirmare":
        if client_vrea_sa_incheie(text_client):
            stare.adauga("operator", REPLICA_INCHEIERE)
            return REPLICA_INCHEIERE, True

        replica = genereaza_replica_robot(
            stare.get_dialog(), stare.domeniu,
            stare.intentie, stare.dificultate, "confirmare"
        )
        stare.adauga("operator", replica)
        return replica, False

    return None, False


# ─── WEBHOOK TWILIO ───────────────────────────────────────────────────────────

@app.route("/incoming-call", methods=["POST"])
@app.route("/incoming-call/", methods=["POST"])
async def incoming_call():
    form_data = await request.form
    call_sid = form_data.get("CallSid", "unknown")
    domeniu  = request.args.get("domeniu", DOMENIU_DEFAULT)

    print(f"\n[APEL NOU] {call_sid} | Domeniu: {domeniu}")

    stare = StareConversatie(domeniu=domeniu)
    conversatii_active[call_sid] = stare

    salut = SALUT_DOMENII[domeniu]
    prima_intrebare = REPLICI_IDENTIFICARE[domeniu][0]
    intro = salut + " " + prima_intrebare
    stare.adauga("operator", intro)

    twiml_body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Response><Connect>'
        f'<Stream url="wss://{NGROK_DOMENIU}/media-stream"></Stream>'
        '</Connect></Response>'
    )
    return Response(twiml_body, status=200, headers={
        "Content-Type": "text/xml; charset=utf-8",
        "Cache-Control": "no-cache"
    })


# ─── WEBSOCKET HANDLER ────────────────────────────────────────────────────────

@app.websocket("/media-stream")
@app.websocket("/media-stream/")
async def handle_twilio_ws():
    from quart import websocket

    call_sid = None
    stare = None
    audio_buffer = []
    robot_vorbeste = True
    processing = False
    stream_sid = None
    silence_timeout = 1.3

    print("[WS] Conexiune noua primita.")

    async def trimite_audio_robot(text_replica):
        nonlocal robot_vorbeste, audio_buffer
        robot_vorbeste = True
        audio_buffer.clear()
        
        print(f"  [DEBUG] Generez TTS pentru: '{text_replica[:20]}...'")
        audio_payload = await tts_to_mulaw(text_replica)
        
        if audio_payload and stream_sid:
            try:
                # 1. Trimitem audio propriu-zis
                await websocket.send(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(audio_payload).decode()}
                }))
                
                # 2. Trimitem MARK-ul corect (fără cheia "media" înăuntru!)
                await websocket.send(json.dumps({
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "robot_done"}
                }))
                print("  [DEBUG] Pachetul MARK a fost trimis către Twilio.")
                
            except Exception as e:
                print(f"[WS SEND EROARE]: {e}")
        else:
            # Dacă TTS-ul a eșuat sau nu avem pachete, deschidem microfonul forțat ca să nu blocăm apelul
            print("  [DEBUG] TTS gol sau lipsă stream_sid. Deschid microfonul forțat.")
            robot_vorbeste = False

    async def proceseaza_silence():
        nonlocal processing, audio_buffer, last_audio_time
        processing = True
        chunks = audio_buffer.copy()
        audio_buffer.clear()

        print(f"  [STT] Pauza detectata. {len(chunks)} chunks la Zevo...")
        text_client = await zevo_stt_transcrie(chunks)
        print(f"  [CLIENT] {text_client}")

        if text_client and text_client.strip() and stare:
            replica, incheie = proceseaza_replica_client(stare, text_client)

            if replica:
                print(f"  [ROBOT] {replica}")
                await trimite_audio_robot(replica)

            if incheie:
                if stare.intentie:
                    try:
                        satisfactie, rezumat = analiza_finala(
                            stare.get_dialog(), stare.domeniu
                        )
                        print(f"\n  [ANALIZA] {stare.intentie} | {satisfactie}")
                        print(f"  [REZUMAT] {rezumat[:100]}...")
                    except Exception as e:
                        print(f"  [ANALIZA EROARE] {e}")
                return True

        processing = False
        last_audio_time[0] = time.time()
        return False

    stop_event = asyncio.Event() # CORECTAT: E mare obligatoriu
    last_audio_time = [time.time()]

    async def citeste_mesaje():
        nonlocal call_sid, stare, stream_sid, robot_vorbeste, processing
        try:
            while not stop_event.is_set():
                message = await websocket.receive()
                if message is None:
                    break
                data = json.loads(message)
                event = data.get("event")

                if event == "start":
                    meta = data.get("start", {})
                    call_sid = meta.get("callSid")
                    stream_sid = meta.get("streamSid")
                    stare = conversatii_active.get(call_sid)
                    if stare:
                        stare.stream_sid = stream_sid
                    print(f"[WS] Stream pornit: {call_sid}")

                    if stare and stare.replici:
                        intro_text = stare.replici[0]["text"]
                        print(f"  [ROBOT] {intro_text}")
                        await trimite_audio_robot(intro_text)

                elif event == "media":
                    if not robot_vorbeste and not processing:
                        payload = data.get("media", {}).get("payload", "")
                        if payload:
                            audio_buffer.append(base64.b64decode(payload))
                            # Print discret la fiecare ~1 secunda de stream
                            if len(audio_buffer) % 50 == 0:
                                print(f"  [DEBUG] Colectez audio... Buffer curent: {len(audio_buffer)} chunks")
                            last_audio_time[0] = time.time()

                elif event == "mark":
                    mark_name = data.get("mark", {}).get("name")
                    if mark_name == "robot_done":
                        print("  [SYSTEM] Robot terminat. Microfon deschis.")
                        robot_vorbeste = False
                        # CORECTAT: Nu mai stergem bufferul aici ca sa nu pierdem frame-urile clientului
                        last_audio_time[0] = time.time()

                elif event == "stop":
                    print(f"[WS] Stream oprit: {call_sid}")
                    stop_event.set()
                    break
        except Exception as e:
            print(f"[WS CITIRE EROARE]: {e}")
            stop_event.set()

    async def detecteaza_silence():
        nonlocal robot_vorbeste, processing
        print("  [DEBUG] Task-ul de detectare silence a pornit.")
        
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
            
            # 1. Siguranță: Dacă robotul pare blocat de mai mult de 8 secunde, deschidem microfonul
            timp_de_la_ultimul_pachet = time.time() - last_audio_time[0]
            if robot_vorbeste and timp_de_la_ultimul_pachet > 8.0:
                print("  [DEBUG] Siguranță activată: Robotul a terminat. Deschid microfonul.")
                robot_vorbeste = False
                audio_buffer.clear()

            # 2. Logica de trimitere la STT
            if audio_buffer and not robot_vorbeste and not processing:
                timp_scurs = time.time() - last_audio_time[0]
                
                # REPARATIE: Mărim silence_timeout la 1.5 secunde (să apuci să respiri între cuvinte)
                # REPARATIE: Mărim limita de chunks de la 250 la 750 (~15 secunde de vorbire continuă maximă)
                if timp_scurs > 1.5 or len(audio_buffer) > 750:
                    print(f"  [DEBUG] Declanșez STT! Motiv: timp_scurs={timp_scurs:.2f}s, chunks={len(audio_buffer)}")
                    incheie = await proceseaza_silence()
                    if incheie:
                        stop_event.set()
                        break

    try:
        await asyncio.gather(citeste_mesaje(), detecteaza_silence())
    except Exception as e:
        print(f"[WS EROARE]: {e}")
    finally:
        if call_sid and call_sid in conversatii_active:
            del conversatii_active[call_sid]


# ─── PORNIRE ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SERVER TWILIO (QUART) — ROBOT TELEFONIC")
    print("=" * 60)
    print(f"HTTP:    http://127.0.0.1:8080/incoming-call")
    print(f"Ngrok:   https://{NGROK_DOMENIU}")
    print(f"Webhook: https://{NGROK_DOMENIU}/incoming-call")
    print("=" * 60 + "\n")

    app.run(host="127.0.0.1", port=8080, debug=False)