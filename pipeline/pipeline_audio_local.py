"""
Pipeline audio modele locale — STT + analiza cu RoMistral si RoGemma
Fluxul: fisier audio -> zevo STT (o singura data) -> RoMistral + RoGemma

Utilizare:
    python3 pipeline_audio_local.py --folder conversatii_subset_audio/
    python3 pipeline_audio_local.py --fisier conversatie_BNK_006.mp3 --domeniu banking
    python3 pipeline_audio_local.py --folder conversatii_subset_audio/ --model romistral
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
import torch
import speech_recognition as sr
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_intentie_local import INTENTII_DOMENII, extrage_intentie
from utils_satisfactie_local import extrage_satisfactie
from utils_rezumat_local import get_tip_rezumat

CONFIG = {
    "STT_API_KEY": "icvsilab2026",
    "STT_DOMAIN":  "ro-RO_general-2026.1",
    "STT_SERVER":  "wss://live-transcriber.zevo-tech.com:2053",
    "RESULTS_DIR": "./rezultate_pipeline_audio",
}

MODELE_LOCALE = {
    "romistral": "OpenLLM-Ro/RoMistral-7b-Instruct",
    "rogemma":   "OpenLLM-Ro/RoGemma-7b-Instruct",
}

PREFIXE_DOMENII = {
    "BNK": "banking", "MED": "medicina", "RET": "retail",
    "TEL": "telecom", "SP":  "servicii_publice",
}

TIP_REZUMAT = {
    "pozitiv": {"tip": "SCURT",  "min_cuv": 20, "max_cuv": 40,  "propozitii": "1-2 propozitii"},
    "neutru":  {"tip": "MEDIU",  "min_cuv": 40, "max_cuv": 70,  "propozitii": "3-4 propozitii"},
    "negativ": {"tip": "LUNG",   "min_cuv": 60, "max_cuv": 100, "propozitii": "5-7 propozitii"},
}

os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

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

def get_tip_rez(satisfactie):
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

# ─── MODEL LOCAL ─────────────────────────────────────────────────────────────

def incarca_model(nume_model):
    print(f"\n  [MODEL] Incarcare {nume_model}...")
    model_id = MODELE_LOCALE[nume_model]
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  [MODEL] Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    model.eval()
    print(f"  [MODEL] {nume_model} incarcat.")
    return tokenizer, model, device

def genereaza(tokenizer, model, device, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    latenta = round(time.time() - start, 2)
    text_generat = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text_generat.strip(), latenta

# ─── PROMPTURI ────────────────────────────────────────────────────────────────

def prompt_intentie_v4(dialog, domeniu, intentii_domenii):
    intentii = intentii_domenii.get(domeniu, [])
    exemple_lungi = {
        "banking": [("OPERATOR: Buna ziua.\nCLIENT: L-am pierdut cardul.\nOPERATOR: Inteleg.", "card_pierdut"),
                    ("OPERATOR: Va ascult.\nCLIENT: De ce a crescut rata?\nOPERATOR: Verific.", "problema_credit")],
        "medicina": [("OPERATOR: Buna ziua.\nCLIENT: Vreau rezultatele analizelor.\nOPERATOR: Va caut.", "rezultate_analize"),
                     ("OPERATOR: Cu ce va ajut?\nCLIENT: Vreau sa anulez programarea.\nOPERATOR: Sigur.", "anulare_programare")],
        "retail": [("OPERATOR: Buna ziua.\nCLIENT: Am primit produse gresite.\nOPERATOR: Imi pare rau.", "comanda_gresita"),
                   ("OPERATOR: Cu ce va ajut?\nCLIENT: Pachetul nu a ajuns.\nOPERATOR: Verific.", "problema_livrare")],
        "telecom": [("OPERATOR: Buna ziua.\nCLIENT: Portarea a fost respinsa.\nOPERATOR: Verific.", "portare_esuata"),
                    ("OPERATOR: Cu ce va ajut?\nCLIENT: Nu pot schimba abonamentul.\nOPERATOR: Va ajut.", "problema_modificare_abonament")],
        "servicii_publice": [("OPERATOR: Primaria.\nCLIENT: Dosarul meu a fost respins.\nOPERATOR: Caut.", "dosar_respins"),
                             ("OPERATOR: Cu ce va ajut?\nCLIENT: Vreau o programare la ghiseu.\nOPERATOR: Va programez.", "programare_ghiseu")],
    }
    exemple = exemple_lungi.get(domeniu, [])
    exemple_text = "".join("CONVERSATIE:\n"+d+"\nINTENTIE IDENTIFICATA: "+i+"\n\n" for d,i in exemple)
    return (
        "Esti un expert in clasificarea intentiilor pentru call-center-uri din domeniul " + domeniu + ".\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "SARCINA: Identifica intentia clientului.\n\n"
        "REGULI:\n1. Include DOAR ce a cerut clientul\n"
        "2. Alege EXCLUSIV din lista furnizata\n"
        "3. Returneaza MAXIM doua intentii, separate prin virgula\n\n"
        "INTENTII DISPONIBILE: " + ", ".join(intentii) + "\n\n"
        "EXEMPLE:\n" + exemple_text + "INTENTIE IDENTIFICATA:"
    )

def prompt_satisfactie_rogemma(dialog):
    return (
        "Esti un expert in analiza satisfactiei clientilor in conversatii de call-center.\n"
        "Determina nivelul de satisfactie al clientului la finalul conversatiei de mai jos.\n\n"
        "DEFINITII DETALIATE:\n"
        "- pozitiv: clientul pleaca multumit si o exprima explicit\n"
        "  Expresii tipice: multumesc mult, excelent, exact ce aveam nevoie\n"
        "- neutru: problema rezolvata dar clientul nu exprima nicio emotie\n"
        "  Expresii tipice: ok, am inteles, bine, la revedere fara caldura\n"
        "- negativ: clientul pleaca frustrat, chiar daca accepta situatia\n"
        "  Expresii tipice: ce sa fac, bine..., nu am de ales, ironie\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "Raspunde DOAR cu unul dintre cuvintele: pozitiv, neutru, negativ:"
    )

def prompt_satisfactie_romistral(dialog):
    return (
        "Analizeaza urmatoarea conversatie telefonica si determina nivelul de satisfactie al clientului.\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "Care este satisfactia clientului la finalul conversatiei? "
        "Raspunde cu un singur cuvant: pozitiv, neutru sau negativ:"
    )

def prompt_rezumat_v2(dialog, satisfactie):
    tip_info = get_tip_rez(satisfactie)
    return (
        "Esti un expert in sumarizarea conversatiilor telefonice din call-center.\n"
        "Genereaza un rezumat de tip " + tip_info["tip"] + " al conversatiei de mai jos.\n\n"
        "CERINTE:\n"
        "- " + tip_info["propozitii"] + ", " + str(tip_info["min_cuv"]) + "-" + str(tip_info["max_cuv"]) + " cuvinte\n"
        "- Scrie in limba romana\n"
        "- Mentioneaza problema principala si rezultatul final\n"
        "- Nu adauga informatii care nu apar in conversatie\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "Rezumat " + tip_info["tip"] + ":"
    )

# ─── ANALIZA CU MODEL LOCAL ───────────────────────────────────────────────────

def analizeaza_cu_model_local(dialog, domeniu, nume_model,
                               tokenizer, model, device, intentii_domenii):
    # Intentie — V4-1 pentru ambele modele locale
    prompt_i = prompt_intentie_v4(dialog, domeniu, intentii_domenii)
    r_i, lat_i = genereaza(tokenizer, model, device, prompt_i, max_new_tokens=40)
    intentie_pred = extrage_intentie(r_i, domeniu)

    # Satisfactie — V2-2 pentru RoGemma, V1-1 pentru RoMistral
    if "gemma" in nume_model.lower():
        prompt_s = prompt_satisfactie_rogemma(dialog)
    else:
        prompt_s = prompt_satisfactie_romistral(dialog)
    r_s, lat_s = genereaza(tokenizer, model, device, prompt_s, max_new_tokens=20)
    satisfactie_pred = extrage_satisfactie(r_s)

    # Rezumat — V2-1 pentru ambele
    tip_info = get_tip_rez(satisfactie_pred)
    prompt_r = prompt_rezumat_v2(dialog, satisfactie_pred)
    r_r, lat_r = genereaza(tokenizer, model, device, prompt_r, max_new_tokens=150)
    nr_cuv = len(r_r.split())
    in_limite = tip_info["min_cuv"] <= nr_cuv <= tip_info["max_cuv"]

    lat_tot = round(lat_i + lat_s + lat_r, 2)
    print(f"    {nume_model:<15} I={intentie_pred:<30} S={satisfactie_pred:<10} "
          f"R={tip_info['tip']}({nr_cuv}cuv) lat={lat_tot}s")

    return {
        "model": nume_model,
        "intentie": {"valoare": intentie_pred, "raspuns_brut": r_i, "latenta": lat_i},
        "satisfactie": {"valoare": satisfactie_pred, "raspuns_brut": r_s, "latenta": lat_s},
        "rezumat": {"tip": tip_info["tip"], "valoare": r_r,
                    "nr_cuvinte": nr_cuv, "in_limite": in_limite, "latenta": lat_r},
        "latenta_totala": lat_tot
    }

# ─── PROCESARE FISIER ─────────────────────────────────────────────────────────

def proceseaza_fisier(cale_fisier, domeniu, modele_incarcate, intentii_domenii):
    nume = Path(cale_fisier).stem
    print(f"\n{'='*60}")
    print(f"Fisier: {Path(cale_fisier).name} | Domeniu: {domeniu}")
    print(f"{'='*60}")

    rezultat = {
        "id": nume, "domeniu": domeniu,
        "fisier_audio": str(cale_fisier),
        "timestamp": datetime.now().isoformat(),
    }

    # STT — o singura data
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

    # Analiza cu fiecare model local
    print(f"  [LLM] Analiza cu {len(modele_incarcate)} modele locale...")
    rezultate_modele = []
    for nume_model, (tokenizer, model, device) in modele_incarcate.items():
        rez = analizeaza_cu_model_local(
            text, domeniu, nume_model, tokenizer, model, device, intentii_domenii
        )
        rezultate_modele.append(rez)

    rezultat["rezultate_modele"] = rezultate_modele
    return rezultat

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline audio modele locale")
    grup = parser.add_mutually_exclusive_group(required=True)
    grup.add_argument("--fisier", help="Un singur fisier audio")
    grup.add_argument("--folder", help="Folder cu fisiere audio")
    parser.add_argument("--domeniu", choices=list(INTENTII_DOMENII.keys()),
                        help="Domeniu fallback daca nu se detecteaza automat")
    parser.add_argument("--model", choices=["romistral", "rogemma"],
                        help="Ruleaza doar un model (default: ambele)")
    args = parser.parse_args()

    # Selecteaza modelele de rulat
    modele_de_rulat = [args.model] if args.model else ["romistral", "rogemma"]

    # Incarca modelele o singura data
    modele_incarcate = {}
    for nume in modele_de_rulat:
        try:
            tokenizer, model, device = incarca_model(nume)
            modele_incarcate[nume] = (tokenizer, model, device)
        except Exception as e:
            print(f"  [EROARE] Nu s-a putut incarca {nume}: {e}")

    if not modele_incarcate:
        print("Niciun model incarcat.")
        return

    # Fisierele de procesat
    fisiere = [Path(args.fisier)] if args.fisier else \
              sorted(Path(args.folder).glob("*.mp3")) + sorted(Path(args.folder).glob("*.wav"))

    if not fisiere:
        print("Niciun fisier gasit.")
        return

    print(f"\nFisiere de procesat: {len(fisiere)}")
    print(f"Modele: {', '.join(modele_incarcate.keys())}")

    toate_rezultatele = []
    for cale in fisiere:
        domeniu = detecteaza_domeniu(cale.name) or args.domeniu
        if not domeniu:
            print(f"\n[SKIP] Domeniu nedetectat: {cale.name}")
            continue
        rezultat = proceseaza_fisier(cale, domeniu, modele_incarcate, INTENTII_DOMENII)
        toate_rezultatele.append(rezultat)

    # Salvare
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = os.path.join(CONFIG["RESULTS_DIR"], f"rezultate_local_{timestamp}.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "modele": list(modele_incarcate.keys()),
            "nr_fisiere": len(fisiere),
            "rezultate": toate_rezultatele
        }, f, ensure_ascii=False, indent=2)
    print(f"\nRezultate salvate in: {output}")

if __name__ == "__main__":
    main()