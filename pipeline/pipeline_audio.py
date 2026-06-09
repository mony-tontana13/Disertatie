"""
Pipeline audio API — STT o singura data + analiza cu GPT, Gemini si command-r7b.
Fluxul: fisier .mp3/.wav -> Zevo STT (o singura transciere) -> 3 modele API in paralel

Utilizare:
    python3 pipeline_audio_api.py --fisier conversatie_BNK_006.mp3 --domeniu banking
    python3 pipeline_audio_api.py --folder ./audio_conversatii/ --auto_domeniu
    python3 pipeline_audio_api.py --folder ./audio_conversatii/ --domeniu retail
"""

import os
import json
import time
import argparse
import unicodedata
import asyncio
import websockets
import ssl
from datetime import datetime
from pathlib import Path

from openai import OpenAI
import google.generativeai as genai
import cohere

# ─── CONFIGURARE ─────────────────────────────────────────────────────────────

CONFIG = {
    "OPENAI_API_KEY":  os.environ.get("OPENAI_API_KEY", ""),
    "GEMINI_API_KEY":  os.environ.get("GEMINI_API_KEY", ""),
    "COHERE_API_KEY":  os.environ.get("COHERE_API_KEY", ""),
    "STT_API_KEY":     "icvsilab2026",
    "STT_DOMAIN":      "ro-RO_general-2026.1",
    "STT_SERVER":      "wss://live-transcriber.zevo-tech.com:2053",
    "RESULTS_DIR":     "./rezultate_pipeline_audio",
}

os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

client_gpt    = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])
genai.configure(api_key=CONFIG["GEMINI_API_KEY"])
client_cohere = cohere.ClientV2(api_key=CONFIG["COHERE_API_KEY"])

MODELE = {
    "gpt":     "gpt-4.1-mini",
    "gemini":  "gemini-2.5-flash",
    "command": "command-r7b-12-2024",
}

# ─── DATE DOMENII ─────────────────────────────────────────────────────────────

PREFIXE_DOMENII = {
    "BNK": "banking", "MED": "medicina", "RET": "retail",
    "TEL": "telecom", "SP":  "servicii_publice",
}

INTENTII_DOMENII = {
    "banking": [
        "problema_credit", "tranzactie_gresita", "card_blocat", "tranzactie_suspecta",
        "problema_transfer", "problema_schimb_valutar", "problema_sold", "card_pierdut"
    ],
    "medicina": [
        "rezultate_analize", "problema_reteta", "problema_asigurare", "reclamatie_personal",
        "consultatie_anulata", "problema_facturare", "problema_programare", "anulare_programare"
    ],
    "retail": [
        "produs_lipsa_stoc", "comanda_gresita", "problema_livrare", "problema_garantie",
        "reclamatie_produs", "anulare_comanda", "comanda_intarziata", "retur_produs"
    ],
    "telecom": [
        "problema_modificare_abonament", "portare_esuata", "problema_internet",
        "problema_roaming", "factura_gresita", "reziliere_contract",
        "activare_esuata", "problema_semnal"
    ],
    "servicii_publice": [
        "dosar_respins", "contestatie_decizie", "informatii_program", "reclamatie_serviciu",
        "sesizare_problema", "problema_plata_taxa", "acte_incomplete", "programare_ghiseu"
    ],
}

TIP_REZUMAT = {
    "pozitiv": {"tip": "SCURT",  "min_cuv": 20, "max_cuv": 40},
    "neutru":  {"tip": "MEDIU",  "min_cuv": 40, "max_cuv": 70},
    "negativ": {"tip": "LUNG",   "min_cuv": 60, "max_cuv": 100},
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
    intentii = INTENTII_DOMENII.get(domeniu, [])
    raspuns_norm = normalizeaza(raspuns)
    for intentie in intentii:
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

# ─── PROMPTURI ────────────────────────────────────────────────────────────────

def prompt_intentie(dialog, domeniu):
    intentii = INTENTII_DOMENII.get(domeniu, [])
    return (
        f"Lucrezi ca analist de date intr-un call-center din domeniul {domeniu}. "
        f"Identifica motivul pentru care a sunat clientul.\n\n"
        f"REGULI:\n"
        f"- Include DOAR ce a cerut sau intrebat clientul\n"
        f"- Alege DOAR din lista de intentii de mai jos\n\n"
        f"INTENTII DISPONIBILE: {', '.join(intentii)}\n\n"
        f"Conversatie:\n{dialog}\n\n"
        f"De ce a sunat clientul? Raspunde cu una sau doua intentii din lista:"
    )

def prompt_satisfactie(dialog):
    exemple_text = (
        "CONVERSATIE:\nCLIENT: Multumesc mult, sunteti prompti!\nSATISFACTIE: pozitiv\n\n"
        "CONVERSATIE:\nCLIENT: Ok, am inteles.\nSATISFACTIE: neutru\n\n"
        "CONVERSATIE:\nCLIENT: Bine, ce sa fac...\nSATISFACTIE: negativ\n\n"
    )
    return (
        f"Esti un expert in analiza satisfactiei clientilor.\n\n"
        f"CONVERSATIE:\n{dialog}\n\n"
        f"DEFINITII:\n"
        f"- pozitiv: clientul pleaca multumit si o exprima clar\n"
        f"- neutru: problema rezolvata dar clientul nu exprima nicio emotie\n"
        f"- negativ: clientul pleaca frustrat, chiar daca accepta situatia\n\n"
        f"REGULI:\n"
        f"1. Uita-te la TONUL GENERAL si REZULTATUL FINAL\n"
        f"2. Frustrarea implicita conteaza: ce sa fac, bine..., ironie\n\n"
        f"EXEMPLE:\n{exemple_text}"
        f"SATISFACTIE IDENTIFICATA:"
    )

def prompt_rezumat(dialog, satisfactie):
    tip_info = get_tip_rezumat(satisfactie)
    tip = tip_info["tip"]
    exemple = {
        "SCURT": ("CLIENT: Am pierdut cardul.\nOPERATOR: L-am blocat imediat.",
                  "Clientul a solicitat blocarea unui card pierdut. Operatorul a rezolvat imediat."),
        "MEDIU": ("CLIENT: Comanda nu a sosit.\nOPERATOR: Intarziere la curier, vine maine.\nCLIENT: Ok.",
                  "Clientul a reclamat o comanda nelivrata. Operatorul a identificat o intarziere si a reprogramat livrarea."),
        "LUNG":  ("CLIENT: A treia oara sun pentru aceeasi problema.\nOPERATOR: Investigam.\nCLIENT: Astept.",
                  "Clientul a contactat call-center-ul pentru a treia oara pentru aceeasi problema nerezolvata. Clientul si-a exprimat nemultumirea. Operatorul a initiat o investigatie. Problema ramane deschisa."),
    }
    d_ex, r_ex = exemple[tip]
    return (
        f"Genereaza un rezumat de tip {tip} al conversatiei.\n"
        f"Lungime: {tip_info['min_cuv']}-{tip_info['max_cuv']} cuvinte. Limba: romana.\n\n"
        f"EXEMPLU:\nCONVERSATIE:\n{d_ex}\nREZUMAT:\n{r_ex}\n\n"
        f"CONVERSATIE:\n{dialog}\n\nREZUMAT {tip}:"
    )

# ─── APELURI LLM ──────────────────────────────────────────────────────────────

def call_gpt(prompt, max_tokens=200):
    start = time.time()
    raspuns = ""
    stream = client_gpt.chat.completions.create(
        model=MODELE["gpt"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            raspuns += chunk.choices[0].delta.content
    return raspuns.strip(), round(time.time() - start, 3)

def call_gemini(prompt, max_tokens=200):
    start = time.time()
    model = genai.GenerativeModel(MODELE["gemini"])
    response = model.generate_content(prompt)
    raspuns = response.text.strip() if response.text else ""
    return raspuns, round(time.time() - start, 3)

def call_command(prompt, max_tokens=200):
    start = time.time()
    response = client_cohere.chat(
        model=MODELE["command"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    raspuns = response.message.content[0].text.strip()
    return raspuns, round(time.time() - start, 3)

CALL_FN = {
    "gpt":     call_gpt,
    "gemini":  call_gemini,
    "command": call_command,
}

# ─── STT ─────────────────────────────────────────────────────────────────────

async def transcrie_audio_ws(audio_data, sample_rate=16000, chunk_size=4096):
    config_msg = json.dumps({
        "config": {
            "key":         CONFIG["STT_API_KEY"],
            "sample_rate": str(sample_rate),
            "domain":      CONFIG["STT_DOMAIN"]
        }
    })
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode    = ssl.CERT_NONE
    sleep_per_chunk = chunk_size / (sample_rate * 2)
    rezultate = []
    partial_curent = ""

    async with websockets.connect(CONFIG["STT_SERVER"], ssl=ssl_ctx) as ws:
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
        eof_resp = await asyncio.wait_for(ws.recv(), timeout=60)
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

    return " ".join(rezultate)

def transcrie_fisier(cale_fisier):
    import wave, tempfile
    cale_str = str(cale_fisier)
    fisier_tmp = None

    if cale_str.lower().endswith(".mp3"):
        fisier_tmp = tempfile.mktemp(suffix=".wav")
        ret = os.system(
            f'ffmpeg -i "{cale_str}" -ar 16000 -ac 1 -sample_fmt s16 '
            f'"{fisier_tmp}" -y -loglevel quiet'
        )
        if ret != 0:
            raise RuntimeError("Conversie MP3->WAV esuata. Instaleaza ffmpeg: brew install ffmpeg")
        cale_de_citit = fisier_tmp
    else:
        cale_de_citit = cale_str

    try:
        with wave.open(cale_de_citit, "rb") as wf:
            wav_data    = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()

        print(f"  [STT] Transcriere in curs ({Path(cale_fisier).name})...")
        start = time.time()
        text = asyncio.run(transcrie_audio_ws(wav_data, sample_rate=sample_rate))
        lat  = round(time.time() - start, 2)
        print(f"  [STT] Gata in {lat}s: {text[:80]}...")
        return text, lat
    finally:
        if fisier_tmp and os.path.exists(fisier_tmp):
            os.remove(fisier_tmp)

# ─── ANALIZA UN MODEL ─────────────────────────────────────────────────────────

def analizeaza_cu_model(dialog, domeniu, model_key):
    call_fn = CALL_FN[model_key]

    # Intentie
    r_i, lat_i = call_fn(prompt_intentie(dialog, domeniu), max_tokens=30)
    intentie    = extrage_intentie(r_i, domeniu)

    # Satisfactie
    r_s, lat_s  = call_fn(prompt_satisfactie(dialog), max_tokens=20)
    satisfactie = extrage_satisfactie(r_s)

    # Rezumat
    tip_info    = get_tip_rezumat(satisfactie)
    r_r, lat_r  = call_fn(prompt_rezumat(dialog, satisfactie), max_tokens=200)
    nr_cuv      = len(r_r.split())
    in_limite   = tip_info["min_cuv"] <= nr_cuv <= tip_info["max_cuv"]

    return {
        "intentie":    {"valoare": intentie,    "raspuns_brut": r_i, "latenta": lat_i},
        "satisfactie": {"valoare": satisfactie, "raspuns_brut": r_s, "latenta": lat_s},
        "rezumat": {
            "tip": tip_info["tip"], "valoare": r_r,
            "nr_cuvinte": nr_cuv, "in_limite": in_limite, "latenta": lat_r
        },
        "latenta_analiza": round(lat_i + lat_s + lat_r, 3)
    }

# ─── PROCESARE UN FISIER ──────────────────────────────────────────────────────

def proceseaza_fisier(cale_fisier, domeniu, modele_selectate):
    nume = Path(cale_fisier).stem
    print(f"\n{'='*60}")
    print(f"Fisier: {Path(cale_fisier).name} | Domeniu: {domeniu}")
    print(f"{'='*60}")

    rezultat = {
        "id":          nume,
        "domeniu":     domeniu,
        "fisier_audio": str(cale_fisier),
        "timestamp":   datetime.now().isoformat(),
        "modele":      {}
    }

    # STT — o singura data pentru toate modelele
    try:
        dialog, lat_stt = transcrie_fisier(cale_fisier)
    except Exception as e:
        print(f"  [EROARE STT]: {e}")
        rezultat["eroare_stt"] = str(e)
        return rezultat

    if not dialog.strip():
        print(f"  [ATENTIE] Transcriere goala.")
        rezultat["eroare_stt"] = "Transcriere goala"
        return rezultat

    rezultat["transcriere"] = {"text": dialog, "latenta_stt": lat_stt}

    # Analiza cu fiecare model selectat
    for model_key in modele_selectate:
        print(f"\n  [{model_key.upper()}] Analiza...")
        try:
            analiza = analizeaza_cu_model(dialog, domeniu, model_key)
            rezultat["modele"][model_key] = analiza
            print(f"    Intentie:    {analiza['intentie']['valoare']}")
            print(f"    Satisfactie: {analiza['satisfactie']['valoare']}")
            print(f"    Rezumat ({analiza['rezumat']['tip']}, "
                  f"{analiza['rezumat']['nr_cuvinte']} cuv, "
                  f"{'OK' if analiza['rezumat']['in_limite'] else 'OUT'}):")
            print(f"    {analiza['rezumat']['valoare'][:100]}...")
            print(f"    Latenta: {analiza['latenta_analiza']}s")
        except Exception as e:
            print(f"    [EROARE {model_key.upper()}]: {e}")
            rezultat["modele"][model_key] = {"eroare": str(e)}

    return rezultat

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline audio API: STT o data + analiza GPT + Gemini + command-r7b"
    )
    grup = parser.add_mutually_exclusive_group(required=True)
    grup.add_argument("--fisier", help="Un singur fisier audio (.mp3 sau .wav)")
    grup.add_argument("--folder", help="Folder cu fisiere audio")

    parser.add_argument("--domeniu", choices=list(INTENTII_DOMENII.keys()),
                        help="Domeniu (necesar daca nu folosesti --auto_domeniu)")
    parser.add_argument("--auto_domeniu", action="store_true",
                        help="Detecteaza automat domeniu din numele fisierului")
    parser.add_argument("--modele", nargs="+",
                        choices=["gpt", "gemini", "command"],
                        default=["gpt", "gemini", "command"],
                        help="Modele de rulat (implicit: toate 3)")
    args = parser.parse_args()

    if not args.domeniu and not args.auto_domeniu:
        parser.error("Specifica --domeniu sau --auto_domeniu")

    fisiere = (
        [Path(args.fisier)] if args.fisier
        else sorted(Path(args.folder).glob("*.mp3")) +
             sorted(Path(args.folder).glob("*.wav"))
    )
    if not fisiere:
        print("Niciun fisier audio gasit.")
        return

    print(f"\nFisiere: {len(fisiere)} | Modele: {args.modele}")

    toate_rezultatele = []
    erori = []

    for cale in fisiere:
        domeniu = detecteaza_domeniu(cale.name) if args.auto_domeniu else None
        if not domeniu:
            if args.domeniu:
                domeniu = args.domeniu
            else:
                print(f"\n[SKIP] Domeniu nedetectat: {cale.name}")
                erori.append({"fisier": str(cale), "eroare": "Domeniu nedetectat"})
                continue

        rezultat = proceseaza_fisier(cale, domeniu, args.modele)
        toate_rezultatele.append(rezultat)

    # Salvare
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        CONFIG["RESULTS_DIR"], f"rezultate_audio_api_{timestamp}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":     timestamp,
            "modele_rulate": args.modele,
            "nr_fisiere":    len(fisiere),
            "nr_procesate":  len(toate_rezultatele),
            "nr_erori":      len(erori),
            "rezultate":     toate_rezultatele,
            "erori":         erori
        }, f, ensure_ascii=False, indent=2)

    # Sumar
    procesate_ok = [r for r in toate_rezultatele if "modele" in r]
    print(f"\n{'='*60}")
    print(f"SUMAR FINAL — {len(procesate_ok)}/{len(fisiere)} procesate")
    print(f"{'='*60}")
    for r in procesate_ok:
        print(f"\n  {r['id']} ({r['domeniu']}):")
        for model_key, analiza in r.get("modele", {}).items():
            if "eroare" in analiza:
                print(f"    {model_key.upper()}: EROARE — {analiza['eroare']}")
            else:
                print(f"    {model_key.upper()}: "
                      f"{analiza['intentie']['valoare']} | "
                      f"{analiza['satisfactie']['valoare']} | "
                      f"{analiza['rezumat']['tip']} "
                      f"({'OK' if analiza['rezumat']['in_limite'] else 'OUT'})")
    print(f"\n  Salvat in: {output_file}")


if __name__ == "__main__":
    main()