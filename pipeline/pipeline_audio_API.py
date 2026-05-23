"""
Pipeline audio multi-model API — STT + analiza pe toate 3 modelele API
Fluxul: fisier audio -> zevo STT (o singura data) -> GPT + Gemini + command-r7b

Utilizare:
    python3 pipeline_audio_api.py --folder conversatii_subset_audio/
    python3 pipeline_audio_api.py --fisier conversatie_BNK_006.mp3 --domeniu banking
"""

import os
import sys
import json
import time
import asyncio
import websockets
import ssl
import argparse
import unicodedata
import tempfile
import speech_recognition as sr
from datetime import datetime
from pathlib import Path
from openai import OpenAI

CONFIG = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
    "COHERE_API_KEY": os.environ.get("COHERE_API_KEY", ""),
    "MODEL_GPT":    "gpt-4.1-mini",
    "MODEL_GEMINI": "gemini-2.5-flash",
    "MODEL_COHERE": "command-r7b-12-2024",
    "STT_API_KEY": "icvsilab2026",
    "STT_DOMAIN":  "ro-RO_general-2026.1",
    "STT_SERVER":  "wss://live-transcriber.zevo-tech.com:2053",
    "RESULTS_DIR": "./rezultate_pipeline_audio",
}

os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
client_gpt = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

PREFIXE_DOMENII = {
    "BNK": "banking", "MED": "medicina", "RET": "retail",
    "TEL": "telecom", "SP":  "servicii_publice",
}

INTENTII_DOMENII = {
    "banking": ["problema_credit","tranzactie_gresita","card_blocat","tranzactie_suspecta",
                "problema_transfer","problema_schimb_valutar","problema_sold","card_pierdut"],
    "medicina": ["rezultate_analize","problema_reteta","problema_asigurare","reclamatie_personal",
                 "consultatie_anulata","problema_facturare","problema_programare","anulare_programare"],
    "retail": ["produs_lipsa_stoc","comanda_gresita","problema_livrare","problema_garantie",
               "reclamatie_produs","anulare_comanda","comanda_intarziata","retur_produs"],
    "telecom": ["problema_modificare_abonament","portare_esuata","problema_internet",
                "problema_roaming","factura_gresita","reziliere_contract","activare_esuata","problema_semnal"],
    "servicii_publice": ["dosar_respins","contestatie_decizie","informatii_program","reclamatie_serviciu",
                         "sesizare_problema","problema_plata_taxa","acte_incomplete","programare_ghiseu"],
}

TIP_REZUMAT = {
    "pozitiv": {"tip": "SCURT",  "min_cuv": 20, "max_cuv": 40,  "propozitii": "1-2 propozitii"},
    "neutru":  {"tip": "MEDIU",  "min_cuv": 40, "max_cuv": 70,  "propozitii": "3-4 propozitii"},
    "negativ": {"tip": "LUNG",   "min_cuv": 60, "max_cuv": 100, "propozitii": "5-7 propozitii"},
}

# ─── UTILITARE ────────────────────────────────────────────────────────────────

def normalizeaza(text):
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower().strip()

def detecteaza_domeniu(nume_fisier):
    nume = Path(nume_fisier).stem.upper()
    for prefix, domeniu in PREFIXE_DOMENII.items():
        if prefix in nume:
            return domeniu
    return None

def extrage_intentie(raspuns, domeniu):
    raspuns_norm = normalizeaza(raspuns)
    for intentie in INTENTII_DOMENII.get(domeniu, []):
        if normalizeaza(intentie) in raspuns_norm:
            return intentie
    return "alta_solicitare"

def extrage_satisfactie(raspuns):
    raspuns_norm = normalizeaza(raspuns)
    for clasa in ["pozitiv", "neutru", "negativ"]:
        if clasa in raspuns_norm:
            return clasa
    return "necunoscut"

def get_tip_rezumat(satisfactie):
    return TIP_REZUMAT.get(satisfactie, TIP_REZUMAT["neutru"])

# ─── STT ─────────────────────────────────────────────────────────────────────

async def speech_to_text_ws(audio_data, api_key, domain,
                             sample_rate=16000, chunk_size=16000,
                             server_uri="wss://live-transcriber.zevo-tech.com:2053"):
    config_payload = {"config": {"key": api_key, "sample_rate": str(sample_rate), "domain": domain}}
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    async with websockets.connect(server_uri, ssl=ssl_ctx) as ws:
        await ws.send(json.dumps(config_payload))
        await ws.recv()
        offset = 0
        while offset < len(audio_data):
            chunk = audio_data[offset:offset + chunk_size]
            await ws.send(chunk)
            response = await ws.recv()
            if "message" in response:
                break
            offset += chunk_size
        await ws.send('{"eof" : 1}')
        return await ws.recv()

def transcrie_fisier(cale_fisier):
    cale_str = str(cale_fisier)
    fisier_tmp = None
    if cale_str.lower().endswith(".mp3"):
        fisier_tmp = tempfile.mktemp(suffix=".wav")
        ret = os.system(f'ffmpeg -i "{cale_str}" -ar 16000 -ac 1 -sample_fmt s16 "{fisier_tmp}" -y -loglevel quiet')
        if ret != 0:
            raise RuntimeError("Conversie MP3->WAV esuata. Instaleaza ffmpeg.")
        cale_de_citit = fisier_tmp
    else:
        cale_de_citit = cale_str
    try:
        with sr.AudioFile(cale_de_citit) as source:
            audio = sr.Recognizer().record(source)
        wav_data = audio.get_wav_data()
        start = time.time()
        result_json = asyncio.run(speech_to_text_ws(wav_data, CONFIG["STT_API_KEY"], CONFIG["STT_DOMAIN"]))
        latenta = round(time.time() - start, 2)
        result = json.loads(result_json)
        text = result.get("text_pp", result.get("text", ""))
        return text, latenta
    finally:
        if fisier_tmp and os.path.exists(fisier_tmp):
            os.remove(fisier_tmp)

# ─── PROMPTURI ────────────────────────────────────────────────────────────────

def prompt_intentie(dialog, domeniu):
    intentii = INTENTII_DOMENII.get(domeniu, [])
    return (
        "Lucrezi ca analist de date intr-un call-center din domeniul " + domeniu + ". "
        "Identifica motivul pentru care a sunat clientul.\n\n"
        "REGULI:\n- Include DOAR ce a cerut clientul\n- Alege DOAR din lista de intentii\n\n"
        "INTENTII DISPONIBILE: " + ", ".join(intentii) + "\n\n"
        "Conversatie:\n" + dialog + "\n\nIntentie identificata:"
    )

def prompt_satisfactie(dialog):
    exemple = {
        "pozitiv": ("CLIENT: Multumesc mult!", "pozitiv"),
        "neutru":  ("CLIENT: Ok, am inteles.", "neutru"),
        "negativ": ("CLIENT: Bine, ce sa fac...", "negativ"),
    }
    exemple_text = "".join("CONVERSATIE:\n"+d+"\nSATISFACTIE: "+s+"\n\n" for _,(d,s) in exemple.items())
    return (
        "Esti un expert in analiza satisfactiei clientilor.\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "DEFINITII:\n- pozitiv: clientul pleaca multumit\n"
        "- neutru: problema rezolvata, clientul indiferent\n"
        "- negativ: clientul pleaca frustrat\n\n"
        "EXEMPLE:\n" + exemple_text + "SATISFACTIE IDENTIFICATA:"
    )

def prompt_rezumat(dialog, satisfactie):
    tip_info = get_tip_rezumat(satisfactie)
    tip = tip_info["tip"]
    exemple = {
        "SCURT": ("CLIENT: Am pierdut cardul.\nOPERATOR: L-am blocat.",
                  "Clientul a solicitat blocarea unui card pierdut. Operatorul a rezolvat imediat."),
        "MEDIU": ("CLIENT: Comanda nu a sosit.\nOPERATOR: Vine maine.\nCLIENT: Ok.",
                  "Clientul a reclamat o comanda nelivrata. Operatorul a identificat o intarziere. Clientul a acceptat."),
        "LUNG":  ("CLIENT: A treia oara sun.\nOPERATOR: Investigam.\nCLIENT: Astept.",
                  "Clientul a contactat call-center-ul pentru a treia oara pentru aceeasi problema. Clientul si-a exprimat nemultumirea. Operatorul a initiat o investigatie. Problema ramane deschisa."),
    }
    d_ex, r_ex = exemple[tip]
    return (
        "Genereaza un rezumat de tip " + tip + ".\n"
        "Lungime: " + str(tip_info["min_cuv"]) + "-" + str(tip_info["max_cuv"]) + " cuvinte. Limba: romana.\n\n"
        "EXEMPLU:\nCONVERSATIE:\n" + d_ex + "\nREZUMAT:\n" + r_ex + "\n\n"
        "CONVERSATIE:\n" + dialog + "\n\nREZUMAT " + tip + ":"
    )

# ─── API CALLS ────────────────────────────────────────────────────────────────

def call_gpt(prompt, max_tokens=50):
    start = time.time()
    raspuns = ""
    for chunk in client_gpt.chat.completions.create(
        model=CONFIG["MODEL_GPT"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, stream=True
    ):
        if chunk.choices[0].delta.content:
            raspuns += chunk.choices[0].delta.content
    return raspuns.strip(), round(time.time() - start, 3)

def call_gemini(prompt):
    from google import genai
    client = genai.Client(api_key=CONFIG["GEMINI_API_KEY"])
    start = time.time()
    raspuns = ""
    for chunk in client.models.generate_content_stream(model=CONFIG["MODEL_GEMINI"], contents=prompt):
        if chunk.text:
            raspuns += chunk.text
    return raspuns.strip(), round(time.time() - start, 3)

def call_cohere(prompt, max_tokens=50):
    import cohere
    co = cohere.ClientV2(api_key=CONFIG["COHERE_API_KEY"])
    start = time.time()
    response = co.chat(model=CONFIG["MODEL_COHERE"],
                       messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return response.message.content[0].text.strip(), round(time.time() - start, 3)

MODELE_API = {}
if CONFIG["OPENAI_API_KEY"]:  MODELE_API["GPT-4.1-mini"]        = lambda p, m=50: call_gpt(p, m)
if CONFIG["GEMINI_API_KEY"]:  MODELE_API["Gemini-2.5-flash"]    = lambda p, m=50: call_gemini(p)
if CONFIG["COHERE_API_KEY"]:  MODELE_API["command-r7b-12-2024"] = lambda p, m=50: call_cohere(p, m)

# ─── ANALIZA ─────────────────────────────────────────────────────────────────

def analizeaza_cu_model(dialog, domeniu, nume_model, fn_call):
    intentie_pred, sat_pred, rez_pred = "necunoscut", "necunoscut", ""
    lat_i = lat_s = lat_r = 0.0

    try:
        r_i, lat_i = fn_call(prompt_intentie(dialog, domeniu), 30)
        intentie_pred = extrage_intentie(r_i, domeniu)
    except Exception as e:
        r_i = f"EROARE: {e}"

    try:
        r_s, lat_s = fn_call(prompt_satisfactie(dialog), 20)
        sat_pred = extrage_satisfactie(r_s)
    except Exception as e:
        r_s = f"EROARE: {e}"

    try:
        tip_info = get_tip_rezumat(sat_pred)
        r_r, lat_r = fn_call(prompt_rezumat(dialog, sat_pred), 200)
        nr_cuv = len(r_r.split())
        in_limite = tip_info["min_cuv"] <= nr_cuv <= tip_info["max_cuv"]
        rez_pred = r_r
    except Exception as e:
        r_r, nr_cuv, in_limite, tip_info = f"EROARE: {e}", 0, False, get_tip_rezumat("neutru")

    print(f"    {nume_model:<25} I={intentie_pred:<30} S={sat_pred:<10} R={tip_info['tip']}({nr_cuv}cuv) lat={round(lat_i+lat_s+lat_r,2)}s")

    return {
        "model": nume_model,
        "intentie": {"valoare": intentie_pred, "raspuns_brut": r_i, "latenta": lat_i},
        "satisfactie": {"valoare": sat_pred, "raspuns_brut": r_s, "latenta": lat_s},
        "rezumat": {"tip": tip_info["tip"], "valoare": rez_pred,
                    "nr_cuvinte": nr_cuv, "in_limite": in_limite, "latenta": lat_r},
        "latenta_totala": round(lat_i + lat_s + lat_r, 3)
    }

# ─── PROCESARE FISIER ─────────────────────────────────────────────────────────

def proceseaza_fisier(cale_fisier, domeniu):
    nume = Path(cale_fisier).stem
    print(f"\n{'='*60}")
    print(f"Fisier: {Path(cale_fisier).name} | Domeniu: {domeniu}")
    print(f"{'='*60}")

    rezultat = {
        "id": nume, "domeniu": domeniu,
        "fisier_audio": str(cale_fisier),
        "timestamp": datetime.now().isoformat(),
    }

    # STT — o singura data pentru toate modelele
    print(f"  [STT] Transcriere...")
    try:
        text, lat_stt = transcrie_fisier(cale_fisier)
        rezultat["transcriere"] = {"text": text, "latenta_stt": lat_stt}
        print(f"  [STT] {lat_stt}s: {text[:100]}...")
    except Exception as e:
        print(f"  [EROARE STT]: {e}")
        rezultat["eroare_stt"] = str(e)
        return rezultat

    if not text.strip():
        print(f"  [ATENTIE] Transcriere goala.")
        rezultat["eroare_stt"] = "Transcriere goala"
        return rezultat

    # Analiza cu fiecare model API
    print(f"  [LLM] Analiza cu {len(MODELE_API)} modele...")
    rezultate_modele = []
    for nume_model, fn_call in MODELE_API.items():
        rez_model = analizeaza_cu_model(text, domeniu, nume_model, fn_call)
        rezultate_modele.append(rez_model)

    rezultat["rezultate_modele"] = rezultate_modele
    return rezultat

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline audio multi-model API")
    grup = parser.add_mutually_exclusive_group(required=True)
    grup.add_argument("--fisier", help="Un singur fisier audio")
    grup.add_argument("--folder", help="Folder cu fisiere audio")
    parser.add_argument("--domeniu", choices=list(INTENTII_DOMENII.keys()),
                        help="Domeniu fallback daca nu se detecteaza automat")
    args = parser.parse_args()

    if not MODELE_API:
        print("Nicio cheie API setata. Seteaza OPENAI_API_KEY, GEMINI_API_KEY sau COHERE_API_KEY.")
        return

    print(f"Modele active: {', '.join(MODELE_API.keys())}")

    fisiere = [Path(args.fisier)] if args.fisier else \
              sorted(Path(args.folder).glob("*.mp3")) + sorted(Path(args.folder).glob("*.wav"))

    if not fisiere:
        print("Niciun fisier gasit.")
        return

    print(f"Fisiere de procesat: {len(fisiere)}")
    toate_rezultatele = []

    for cale in fisiere:
        domeniu = detecteaza_domeniu(cale.name) or args.domeniu
        if not domeniu:
            print(f"\n[SKIP] Domeniu nedetectat: {cale.name}")
            continue
        rezultat = proceseaza_fisier(cale, domeniu)
        toate_rezultatele.append(rezultat)

    # Salvare
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = os.path.join(CONFIG["RESULTS_DIR"], f"rezultate_api_{timestamp}.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "modele": list(MODELE_API.keys()),
                   "nr_fisiere": len(fisiere), "rezultate": toate_rezultatele},
                  f, ensure_ascii=False, indent=2)
    print(f"\nRezultate salvate in: {output}")

if __name__ == "__main__":
    main()