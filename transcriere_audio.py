"""
Transcriere audio cu zevo STT — salveaza fiecare conversatie intr-un fisier separat.

Utilizare:
    python3 transcrie_audio.py --folder conversatii_subset_audio/
    python3 transcrie_audio.py --fisier conversatie_BNK_006.mp3
"""

import os
import sys
import json
import asyncio
import websockets
import ssl
import argparse
import tempfile
import time
import speech_recognition as sr
from pathlib import Path
from datetime import datetime

CONFIG = {
    "STT_API_KEY": "icvsilab2026",
    "STT_DOMAIN":  "ro-RO_general-2026.1",
    "STT_SERVER":  "wss://live-transcriber.zevo-tech.com:2053",
    "OUTPUT_DIR":  "./transcrieri_audio",
}

os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# ─── STT ─────────────────────────────────────────────────────────────────────

async def speech_to_text_ws(audio_data, api_key, domain,
                             sample_rate=16000, chunk_size=4096,
                             server_uri="wss://live-transcriber.zevo-tech.com:2053"):
    """
    Serverul zevo trimite JSON cu cheia 'partial' sau 'text'/'text_pp' direct la root.
    Exemplu partial: {"partial": "buna ziua ce doriti"}
    Exemplu result:  {"text": "buna ziua", "text_pp": "Buna ziua"}
    """
    config_msg = json.dumps({
        "config": {"key": api_key, "sample_rate": str(sample_rate), "domain": domain}
    })

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    sleep_per_chunk = chunk_size / (sample_rate * 2)

    rezultate_finale = []
    partial_curent = ""

    async with websockets.connect(server_uri, ssl=ssl_ctx) as ws:
        await ws.send(config_msg)
        await asyncio.wait_for(ws.recv(), timeout=30)

        offset = 0
        while offset < len(audio_data):
            chunk = audio_data[offset:offset + chunk_size]
            await ws.send(chunk)
            result = await asyncio.wait_for(ws.recv(), timeout=60)
            if "message" in result:
                break
            try:
                parsed = json.loads(result)
                if "text_pp" in parsed and parsed["text_pp"].strip():
                    # Rezultat final al unui fragment
                    rezultate_finale.append(parsed["text_pp"].strip())
                    partial_curent = ""
                elif "text" in parsed and parsed["text"].strip() and "partial" not in parsed:
                    # Rezultat final fara text_pp
                    rezultate_finale.append(parsed["text"].strip())
                    partial_curent = ""
                elif "partial" in parsed and parsed["partial"].strip():
                    # Transcriere partiala — retine cel mai recent
                    partial_curent = parsed["partial"].strip()
            except Exception:
                pass
            offset += chunk_size
            await asyncio.sleep(sleep_per_chunk)

        # Daca mai e un partial nefinalizat, adauga-l
        if partial_curent and partial_curent not in " ".join(rezultate_finale):
            rezultate_finale.append(partial_curent)

        await ws.send('{"eof" : 1}')
        eof_resp = await asyncio.wait_for(ws.recv(), timeout=60)
        try:
            parsed = json.loads(eof_resp)
            if "text_pp" in parsed and parsed["text_pp"].strip():
                rezultate_finale.append(parsed["text_pp"].strip())
            elif "text" in parsed and parsed["text"].strip():
                rezultate_finale.append(parsed["text"].strip())
            elif "partial" in parsed and parsed["partial"].strip():
                p = parsed["partial"].strip()
                if p not in " ".join(rezultate_finale):
                    rezultate_finale.append(p)
        except Exception:
            pass

    text_final = " ".join(rezultate_finale)
    return json.dumps({"text_pp": text_final, "text": text_final})

def transcrie_fisier(cale_fisier):
    """Transcrie un fisier MP3 sau WAV si returneaza textul."""
    cale_str = str(cale_fisier)
    fisier_tmp = None

    if cale_str.lower().endswith(".mp3"):
        fisier_tmp = tempfile.mktemp(suffix=".wav")
        ret = os.system(
            f'ffmpeg -i "{cale_str}" -ar 16000 -ac 1 -sample_fmt s16 '
            f'"{fisier_tmp}" -y -loglevel quiet'
        )
        if ret != 0:
            raise RuntimeError("Conversie MP3->WAV esuata. Verifica ca ffmpeg e instalat: brew install ffmpeg")
        cale_de_citit = fisier_tmp
    else:
        cale_de_citit = cale_str

    try:
        with sr.AudioFile(cale_de_citit) as source:
            audio = sr.Recognizer().record(source)
        wav_data = audio.get_wav_data()

        start = time.time()
        result_json = asyncio.run(
            speech_to_text_ws(
                wav_data,
                CONFIG["STT_API_KEY"],
                CONFIG["STT_DOMAIN"],
                server_uri=CONFIG["STT_SERVER"]
            )
        )
        latenta = round(time.time() - start, 2)
        result = json.loads(result_json)
        text = result.get("text_pp", result.get("text", ""))
        return text, latenta

    finally:
        if fisier_tmp and os.path.exists(fisier_tmp):
            os.remove(fisier_tmp)


# ─── SALVARE ─────────────────────────────────────────────────────────────────

def salveaza_transcriere(nume_fisier, text, latenta):
    """Salveaza transcrierea intr-un fisier text si unul JSON."""
    nume = Path(nume_fisier).stem
    output_txt  = os.path.join(CONFIG["OUTPUT_DIR"], f"{nume}_transcriere.txt")
    output_json = os.path.join(CONFIG["OUTPUT_DIR"], f"{nume}_transcriere.json")

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(text)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "id": nume,
            "fisier_audio": str(nume_fisier),
            "timestamp": datetime.now().isoformat(),
            "latenta_stt": latenta,
            "text": text,
        }, f, ensure_ascii=False, indent=2)

    return output_txt


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transcriere audio cu zevo STT")
    grup = parser.add_mutually_exclusive_group(required=True)
    grup.add_argument("--fisier", help="Un singur fisier audio (.mp3 sau .wav)")
    grup.add_argument("--folder", help="Folder cu fisiere audio")
    args = parser.parse_args()

    fisiere = [Path(args.fisier)] if args.fisier else \
              sorted(Path(args.folder).glob("*.mp3")) + \
              sorted(Path(args.folder).glob("*.wav"))

    if not fisiere:
        print("Niciun fisier audio gasit.")
        return

    print(f"\nFisiere de transcris: {len(fisiere)}")
    print(f"Output: {CONFIG['OUTPUT_DIR']}/\n")

    succes, erori = 0, 0

    for i, cale in enumerate(fisiere, 1):
        print(f"[{i:02d}/{len(fisiere)}] {cale.name} ...", end=" ", flush=True)
        try:
            text, latenta = transcrie_fisier(cale)
            if not text.strip():
                print(f"ATENTIE: transcriere goala ({latenta}s)")
                erori += 1
                continue
            output = salveaza_transcriere(cale.name, text, latenta)
            print(f"OK ({latenta}s) -> {Path(output).name}")
            print(f"    \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
            succes += 1
        except Exception as e:
            print(f"EROARE: {e}")
            erori += 1

    print(f"\n{'='*50}")
    print(f"Transcrise cu succes: {succes}/{len(fisiere)}")
    if erori:
        print(f"Erori: {erori}")
    print(f"Fisiere salvate in: {CONFIG['OUTPUT_DIR']}/")


if __name__ == "__main__":
    main()