"""
Pipeline audio modele locale — STT o singura data + RoMistral si RoGemma secvential.
Fluxul: fisiere audio -> Zevo STT (toate odata) -> RoMistral -> RoGemma

Utilizare:
    python3 pipeline_audio_local.py --folder conversatii_subset_audio/
    python3 pipeline_audio_local.py --fisier conversatie_BNK_006.wav --domeniu banking
    python3 pipeline_audio_local.py --folder conversatii_subset_audio/ --model rogemma
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
import gc
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "/Users/antoniadumitru/Desktop/facultate/Disertatie/prompt_engineering_local/intentii")
sys.path.insert(0, "/Users/antoniadumitru/Desktop/facultate/Disertatie/prompt_engineering_local/satisfactie")
sys.path.insert(0, "/Users/antoniadumitru/Desktop/facultate/Disertatie/prompt_engineering_local/rezumat")
from utils_intentie_local import INTENTII_DOMENII, extrage_intentie
from utils_satisfactie_local import extrage_satisfactie

# ─── CONFIGURARE ─────────────────────────────────────────────────────────────

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
    import wave
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

        start = time.time()
        text  = asyncio.run(transcrie_audio_ws(wav_data, sample_rate=sample_rate))
        lat   = round(time.time() - start, 2)
        return text, lat
    finally:
        if fisier_tmp and os.path.exists(fisier_tmp):
            os.remove(fisier_tmp)


# ─── FAZA 1: STT PE TOATE FISIERELE ──────────────────────────────────────────

def transcrie_toate(fisiere, domeniu_fallback=None):
    """Transcrie toate fisierele audio si returneaza dict id->transcriere."""
    print(f"\n{'='*60}")
    print(f"FAZA 1 — Transcriere STT ({len(fisiere)} fisiere)")
    print(f"{'='*60}")

    transcrieri = {}
    for i, cale in enumerate(fisiere, 1):
        domeniu = detecteaza_domeniu(cale.name) or domeniu_fallback
        if not domeniu:
            print(f"  [{i}/{len(fisiere)}] SKIP {cale.name} — domeniu nedetectat")
            continue

        print(f"  [{i}/{len(fisiere)}] {cale.name} ({domeniu})...", end=" ", flush=True)
        try:
            text, lat = transcrie_fisier(cale)
            if text.strip():
                transcrieri[cale] = {"text": text, "latenta_stt": lat, "domeniu": domeniu}
                print(f"OK ({lat}s) — {text[:60]}...")
            else:
                print(f"GOALA ({lat}s)")
        except Exception as e:
            print(f"EROARE — {e}")

    print(f"\n  Transcrise cu succes: {len(transcrieri)}/{len(fisiere)}")
    return transcrieri


# ─── MODEL LOCAL ─────────────────────────────────────────────────────────────

def incarca_model(nume_model):
    print(f"\n  Incarcare {MODELE_LOCALE[nume_model]}...")
    model_id = MODELE_LOCALE[nume_model]
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    print(f"  {nume_model} incarcat.")
    return tokenizer, model, device


def elibereaza_model(model, device):
    """Elibereaza memoria dupa terminarea unui model."""
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    print("  Memorie eliberata.")


def genereaza(tokenizer, model, device, prompt, max_new_tokens=100, timeout=120):
    """Genereaza text cu timeout hard — opreste dupa `timeout` secunde."""
    import signal

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Truncheaza promptul daca e prea lung (max 1024 tokens)
    if inputs["input_ids"].shape[1] > 1024:
        inputs = {k: v[:, -1024:] for k, v in inputs.items()}

    rezultat = {"text": "", "latenta": 0}

    def _genereaza():
        start = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,
            )
        rezultat["latenta"] = round(time.time() - start, 2)
        rezultat["text"] = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    # Timeout via threading
    import threading
    t = threading.Thread(target=_genereaza)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        print(f" [TIMEOUT {timeout}s]", end=" ", flush=True)
        rezultat["text"] = ""
        rezultat["latenta"] = timeout
        # Nu putem opri thread-ul fortat pe MPS, dar macar continuam

    return rezultat["text"], rezultat["latenta"]


# ─── PROMPTURI ────────────────────────────────────────────────────────────────

def prompt_intentie_v4(dialog, domeniu):
    intentii = INTENTII_DOMENII.get(domeniu, [])
    exemple_lungi = {
        "banking": [
            ("OPERATOR: Buna ziua.\nCLIENT: L-am pierdut cardul.\nOPERATOR: Inteleg.", "card_pierdut"),
            ("OPERATOR: Va ascult.\nCLIENT: De ce a crescut rata?\nOPERATOR: Verific.", "problema_credit"),
        ],
        "medicina": [
            ("OPERATOR: Buna ziua.\nCLIENT: Vreau rezultatele analizelor.\nOPERATOR: Va caut.", "rezultate_analize"),
            ("OPERATOR: Cu ce va ajut?\nCLIENT: Vreau sa anulez programarea.\nOPERATOR: Sigur.", "anulare_programare"),
        ],
        "retail": [
            ("OPERATOR: Buna ziua.\nCLIENT: Am primit produse gresite.\nOPERATOR: Imi pare rau.", "comanda_gresita"),
            ("OPERATOR: Cu ce va ajut?\nCLIENT: Pachetul nu a ajuns.\nOPERATOR: Verific.", "problema_livrare"),
        ],
        "telecom": [
            ("OPERATOR: Buna ziua.\nCLIENT: Portarea a fost respinsa.\nOPERATOR: Verific.", "portare_esuata"),
            ("OPERATOR: Cu ce va ajut?\nCLIENT: Nu pot schimba abonamentul.\nOPERATOR: Va ajut.", "problema_modificare_abonament"),
        ],
        "servicii_publice": [
            ("OPERATOR: Primaria.\nCLIENT: Dosarul meu a fost respins.\nOPERATOR: Caut.", "dosar_respins"),
            ("OPERATOR: Cu ce va ajut?\nCLIENT: Vreau o programare la ghiseu.\nOPERATOR: Va programez.", "programare_ghiseu"),
        ],
    }
    exemple = exemple_lungi.get(domeniu, [])
    exemple_text = "".join(
        f"CONVERSATIE:\n{d}\nINTENTIE IDENTIFICATA: {i}\n\n" for d, i in exemple
    )
    return (
        f"Esti un expert in clasificarea intentiilor pentru call-center-uri din domeniul {domeniu}.\n\n"
        f"CONVERSATIE:\n{dialog}\n\n"
        f"SARCINA: Identifica intentia clientului.\n\n"
        f"REGULI:\n1. Include DOAR ce a cerut clientul\n"
        f"2. Alege EXCLUSIV din lista furnizata\n"
        f"3. Returneaza MAXIM doua intentii, separate prin virgula\n\n"
        f"INTENTII DISPONIBILE: {', '.join(intentii)}\n\n"
        f"EXEMPLE:\n{exemple_text}INTENTIE IDENTIFICATA:"
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
        f"Conversatie:\n{dialog}\n\n"
        "Raspunde DOAR cu unul dintre cuvintele: pozitiv, neutru, negativ:"
    )

def prompt_satisfactie_romistral(dialog):
    return (
        "Analizeaza urmatoarea conversatie telefonica.\n\n"
        f"Conversatie:\n{dialog}\n\n"
        "Satisfactia clientului este pozitiv, neutru sau negativ?\n"
        "Raspunde DOAR cu unul dintre aceste cuvinte exacte: pozitiv, neutru, negativ\n"
        "Satisfactia este:"
    )

def prompt_rezumat_romistral(dialog, satisfactie):
    """V2-1 — cel mai bun pentru RoMistral."""
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

def prompt_rezumat_rogemma(dialog, satisfactie):
    """V2-2 — cel mai bun pentru RoGemma."""
    tip_info = get_tip_rez(satisfactie)
    return (
        "Esti un expert in sumarizarea conversatiilor telefonice din call-center.\n"
        "Genereaza un rezumat de tip " + tip_info["tip"] + " al conversatiei de mai jos.\n\n"
        "CERINTE DE FORMAT:\n"
        "- Lungime: " + str(tip_info["min_cuv"]) + "-" + str(tip_info["max_cuv"]) + " cuvinte (" + tip_info["propozitii"] + ")\n"
        "- Limba: romana\n"
        "- Nu adauga informatii care nu apar in conversatie\n\n"
        "STRUCTURA:\n"
        "- Incepe cu motivul apelului clientului\n"
        "- Continua cu actiunile intreprinse de operator\n"
        "- Incheie cu rezultatul final al conversatiei\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "Rezumat " + tip_info["tip"] + ":"
    )


# ─── FAZA 2: ANALIZA CU UN MODEL LOCAL ───────────────────────────────────────

def truncheaza_dialog(text, max_cuvinte=300):
    """Pastreaza primele si ultimele cuvinte pentru a nu pierde finalul conversatiei."""
    cuvinte = text.split()
    if len(cuvinte) <= max_cuvinte:
        return text
    jumatate = max_cuvinte // 2
    return " ".join(cuvinte[:jumatate]) + " [...] " + " ".join(cuvinte[-jumatate:])


def analizeaza_cu_model(transcrieri, nume_model, tokenizer, model, device):
    """Ruleaza un model pe toate transcrierile si returneaza rezultatele."""
    print(f"\n  Analiza cu {nume_model} pe {len(transcrieri)} conversatii...")
    rezultate = {}

    for cale, info in transcrieri.items():
        dialog_complet = info["text"]
        dialog  = truncheaza_dialog(dialog_complet, max_cuvinte=300)
        domeniu = info["domeniu"]
        print(f"    {Path(cale).stem}...", end=" ", flush=True)

        # Intentie
        r_i, lat_i   = genereaza(tokenizer, model, device,
                                  prompt_intentie_v4(dialog, domeniu), max_new_tokens=40)
        intentie_pred = extrage_intentie(r_i, domeniu)

        # Satisfactie
        prompt_s = (prompt_satisfactie_rogemma(dialog) if "gemma" in nume_model.lower()
                    else prompt_satisfactie_romistral(dialog))
        r_s, lat_s    = genereaza(tokenizer, model, device, prompt_s, max_new_tokens=20)
        satisfactie_pred = extrage_satisfactie(r_s)

        # Rezumat — prompt diferit per model
        tip_info = get_tip_rez(satisfactie_pred)
        fn_rez   = prompt_rezumat_rogemma if "gemma" in nume_model.lower() else prompt_rezumat_romistral
        r_r, lat_r = genereaza(tokenizer, model, device,
                               fn_rez(dialog, satisfactie_pred),
                               max_new_tokens=600)
        nr_cuv       = len(r_r.split())
        in_limite    = tip_info["min_cuv"] <= nr_cuv <= tip_info["max_cuv"]
        lat_tot      = round(lat_i + lat_s + lat_r, 2)

        print(f"I={intentie_pred} S={satisfactie_pred} lat={lat_tot}s")

        rezultate[str(cale)] = {
            "intentie":    {"valoare": intentie_pred, "raspuns_brut": r_i, "latenta": lat_i},
            "satisfactie": {"valoare": satisfactie_pred, "raspuns_brut": r_s, "latenta": lat_s},
            "rezumat":     {"tip": tip_info["tip"], "valoare": r_r,
                            "nr_cuvinte": nr_cuv, "in_limite": in_limite, "latenta": lat_r},
            "latenta_totala": lat_tot,
        }

    return rezultate


# ─── ASAMBLARE REZULTATE FINALE ───────────────────────────────────────────────

def asambleaza_rezultate(transcrieri, rezultate_per_model):
    """Combina transcrierile cu rezultatele per model intr-o structura finala."""
    finale = []
    for cale, info in transcrieri.items():
        intrare = {
            "id":           Path(cale).stem,
            "domeniu":      info["domeniu"],
            "fisier_audio": str(cale),
            "transcriere":  {"text": info["text"], "latenta_stt": info["latenta_stt"]},
            "rezultate_modele": {}
        }
        for model_key, rez_model in rezultate_per_model.items():
            if str(cale) in rez_model:
                intrare["rezultate_modele"][model_key] = rez_model[str(cale)]
        finale.append(intrare)
    return finale


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline audio local: STT o data + RoMistral + RoGemma secvential"
    )
    grup = parser.add_mutually_exclusive_group(required=True)
    grup.add_argument("--fisier", help="Un singur fisier audio (.mp3 sau .wav)")
    grup.add_argument("--folder", help="Folder cu fisiere audio")
    parser.add_argument("--domeniu", choices=list(INTENTII_DOMENII.keys()),
                        help="Domeniu fallback daca nu se detecteaza din numele fisierului")
    parser.add_argument("--model", choices=["romistral", "rogemma"],
                        help="Ruleaza doar un model (default: ambele, secvential)")
    parser.add_argument("--skip_stt", action="store_true",
                        help="Sare peste STT si foloseste cache-ul salvat anterior")
    parser.add_argument("--cache_stt",
                        default=None,
                        help="Cale catre fisierul cache STT (default: rezultate_pipeline_audio/cache_stt.json)")
    args = parser.parse_args()

    modele_de_rulat = [args.model] if args.model else ["romistral", "rogemma"]

    # ── FAZA 1: STT sau incarcare din cache ──────────────────────────────────
    if args.skip_stt:
        cache_path = args.cache_stt or os.path.join(CONFIG["RESULTS_DIR"], "cache_stt.json")
        print(f"\n[SKIP STT] Incarcare transcrieri din {cache_path}...")
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        transcrieri = {Path(k): v for k, v in raw.items()}
        print(f"  {len(transcrieri)} transcrieri incarcate.")
    else:
        fisiere = (
            [Path(args.fisier)] if args.fisier
            else sorted(Path(args.folder).glob("*.mp3")) + sorted(Path(args.folder).glob("*.wav"))
        )
        if not fisiere:
            print("Niciun fisier audio gasit.")
            return
        transcrieri = transcrie_toate(fisiere, domeniu_fallback=args.domeniu)

    if not transcrieri:
        print("Nicio transcriere reusita. Oprire.")
        return

    # Salveaza transcrierile pe disk ca sa poata fi reluate fara STT
    cache_stt = os.path.join(CONFIG["RESULTS_DIR"], "cache_stt.json")
    with open(cache_stt, "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v for k, v in transcrieri.items()},
            f, ensure_ascii=False, indent=2
        )
    print(f"  Transcrieri salvate in: {cache_stt}")
    print(f"  (Poti relua cu --skip_stt ca sa sari peste faza STT)")

    # ── FAZA 2: Analiza cu fiecare model, secvential ─────────────────────────
    rezultate_per_model = {}

    for i, nume_model in enumerate(modele_de_rulat):
        print(f"\n{'='*60}")
        print(f"FAZA 2.{i+1} — Model: {nume_model.upper()}")
        print(f"{'='*60}")

        try:
            tokenizer, model, device = incarca_model(nume_model)
            rez = analizeaza_cu_model(transcrieri, nume_model, tokenizer, model, device)
            rezultate_per_model[nume_model] = rez
        except Exception as e:
            print(f"  [EROARE] {e}")
            rezultate_per_model[nume_model] = {}
        finally:
            # Elibereaza memoria inainte de urmatorul model
            try:
                elibereaza_model(model, device)
            except Exception:
                pass

    # ── Asamblare si salvare ──────────────────────────────────────────────────
    rezultate_finale = asambleaza_rezultate(transcrieri, rezultate_per_model)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(CONFIG["RESULTS_DIR"], f"rezultate_local_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":     timestamp,
            "modele_rulate": modele_de_rulat,
            "nr_transcrise": len(transcrieri),
            "rezultate":     rezultate_finale,
        }, f, ensure_ascii=False, indent=2)

    # ── Sumar ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMAR FINAL")
    print(f"{'='*60}")
    for r in rezultate_finale:
        print(f"\n  {r['id']} ({r['domeniu']}):")
        for model_key, analiza in r.get("rezultate_modele", {}).items():
            print(f"    {model_key.upper():<12} "
                  f"I={analiza['intentie']['valoare']:<30} "
                  f"S={analiza['satisfactie']['valoare']:<10} "
                  f"R={analiza['rezumat']['tip']} "
                  f"({'OK' if analiza['rezumat']['in_limite'] else 'OUT'})")
    print(f"\n  Salvat in: {output_file}")


if __name__ == "__main__":
    main()