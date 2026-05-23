"""
Pipeline audio — STT + analiza LLM pe fisiere audio pre-inregistrate
Fluxul: fisier .mp3/.wav -> zevo STT -> intentie + satisfactie + rezumat

Utilizare:
    # Un singur fisier, domeniu specificat
    python3 pipeline_audio.py --fisier conversatie_BNK_006.mp3 --domeniu banking

    # Un folder cu mai multe fisiere audio
    python3 pipeline_audio.py --folder ./audio_conversatii/ --domeniu banking

    # Domeniu detectat automat din numele fisierului (ex: BNK -> banking)
    python3 pipeline_audio.py --folder ./audio_conversatii/ --auto_domeniu
"""

import os
import sys
import json
import time
import argparse
import unicodedata
from datetime import datetime
from pathlib import Path
from openai import OpenAI

try:
    import asyncio
    import websockets
    import ssl
    import speech_recognition as sr
    ZEVO_DISPONIBIL = True
except ImportError:
    ZEVO_DISPONIBIL = False
    print("ATENTIE: zevo_stt sau speech_recognition nu sunt disponibile.")

# ─── CONFIGURARE ─────────────────────────────────────────────────────────────

CONFIG = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    "MODEL": "gpt-4.1-mini",
    "STT_API_KEY": "icvsilab2026",
    "STT_DOMAIN": "ro-RO_general-2026.1",
    "STT_SERVER": "wss://live-transcriber.zevo-tech.com:2053",
    "RESULTS_DIR": "./rezultate_pipeline_audio",
}

os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

# ─── MAPARE DOMENII DIN NUMELE FISIERULUI ────────────────────────────────────

PREFIXE_DOMENII = {
    "BNK": "banking",
    "MED": "medicina",
    "RET": "retail",
    "TEL": "telecom",
    "SP":  "servicii_publice",
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
    "pozitiv": {"tip": "SCURT",  "min_cuv": 20, "max_cuv": 40,  "propozitii": "1-2 propozitii"},
    "neutru":  {"tip": "MEDIU",  "min_cuv": 40, "max_cuv": 70,  "propozitii": "3-4 propozitii"},
    "negativ": {"tip": "LUNG",   "min_cuv": 60, "max_cuv": 100, "propozitii": "5-7 propozitii"},
}


# ─── UTILITARE ────────────────────────────────────────────────────────────────

def normalizeaza(text):
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower().strip()


def detecteaza_domeniu_din_nume(nume_fisier):
    """Detecteaza domeniul din numele fisierului.
    Functioneaza cu formate: BNK_006, conversatie_BNK_006 etc.
    """
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


# ─── CONVERSIE AUDIO ─────────────────────────────────────────────────────────

def citeste_audio_sr(cale_fisier):
    """
    Citeste un fisier WAV si returneaza raw PCM bytes (fara header).
    MP3 e convertit la WAV 16kHz mono cu ffmpeg inainte de citire.
    """
    import wave
    import tempfile
    cale_str = str(cale_fisier)
    fisier_temporar = None

    if cale_str.lower().endswith(".mp3"):
        fisier_temporar = tempfile.mktemp(suffix=".wav")
        ret = os.system(
            f'ffmpeg -i "{cale_str}" -ar 16000 -ac 1 -sample_fmt s16             "{fisier_temporar}" -y -loglevel quiet'
        )
        if ret != 0:
            raise RuntimeError("Conversie MP3->WAV esuata. Verifica ca ffmpeg e instalat: brew install ffmpeg")
        cale_de_citit = fisier_temporar
    else:
        cale_de_citit = cale_str

    try:
        with wave.open(cale_de_citit, "rb") as wf:
            return wf.readframes(wf.getnframes())
    finally:
        if fisier_temporar and os.path.exists(fisier_temporar):
            os.remove(fisier_temporar)


# ─── STT CU ZEVO ─────────────────────────────────────────────────────────────

async def transcrie_audio_ws(audio_data, api_key, domain,
                              sample_rate=16000, chunk_size=4096,
                              server_uri="wss://live-transcriber.zevo-tech.com:2053"):
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
                    rezultate_finale.append(parsed["text_pp"].strip())
                    partial_curent = ""
                elif "text" in parsed and parsed["text"].strip() and "partial" not in parsed:
                    rezultate_finale.append(parsed["text"].strip())
                    partial_curent = ""
                elif "partial" in parsed and parsed["partial"].strip():
                    partial_curent = parsed["partial"].strip()
            except Exception:
                pass
            offset += chunk_size
            await asyncio.sleep(sleep_per_chunk)

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

def transcrie_fisier_audio(cale_fisier, domeniu):
    print(f"  [STT] Transcriere: {Path(cale_fisier).name}")

    try:
        print(f"  [STT] Citire fisier audio...")
        import wave
        with wave.open(str(cale_fisier), "rb") as wf:
            wav_data = wf.readframes(wf.getnframes())
            sample_rate = wf.getframerate()

        start = time.time()
        result_json = asyncio.run(
            transcrie_audio_ws(
                wav_data,
                CONFIG["STT_API_KEY"],
                CONFIG["STT_DOMAIN"],
                sample_rate=sample_rate,
                server_uri=CONFIG["STT_SERVER"]
            )
        )
        latenta_stt = round(time.time() - start, 2)
        result = json.loads(result_json)
        text = result.get("text_pp", result.get("text", ""))
        print(f"  [STT] Transcriere completa in {latenta_stt}s: {text[:80]}...")
        return text, latenta_stt

    except Exception as e:
        raise RuntimeError(str(e))

def call_llm(prompt, max_tokens=50):
    start = time.time()
    raspuns = ""
    stream = client.chat.completions.create(
        model=CONFIG["MODEL"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            raspuns += chunk.choices[0].delta.content
    return raspuns.strip(), round(time.time() - start, 3)


# ─── PROMPTURI CASTIGATOARE GPT ──────────────────────────────────────────────

def prompt_intentie(dialog, domeniu):
    intentii = INTENTII_DOMENII.get(domeniu, [])
    return (
        "Lucrezi ca analist de date intr-un call-center din domeniul " + domeniu + ". "
        "Identifica motivul pentru care a sunat clientul.\n\n"
        "REGULI:\n"
        "- Include DOAR ce a cerut sau intrebat clientul\n"
        "- Alege DOAR din lista de intentii de mai jos\n\n"
        "INTENTII DISPONIBILE: " + ", ".join(intentii) + "\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "De ce a sunat clientul? Raspunde cu una sau doua intentii din lista:"
    )


def prompt_satisfactie(dialog):
    exemple = {
        "pozitiv": ("CLIENT: Multumesc mult, sunteti promti!", "pozitiv"),
        "neutru":  ("CLIENT: Ok, am inteles.", "neutru"),
        "negativ": ("CLIENT: Bine, ce sa fac...", "negativ"),
    }
    exemple_text = "".join(
        "CONVERSATIE:\n" + d + "\nSATISFACTIE: " + s + "\n\n"
        for _, (d, s) in exemple.items()
    )
    return (
        "Esti un expert in analiza satisfactiei clientilor.\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "DEFINITII:\n"
        "- pozitiv: clientul pleaca multumit si o exprima clar\n"
        "- neutru: problema rezolvata dar clientul nu exprima nicio emotie\n"
        "- negativ: clientul pleaca frustrat, chiar daca accepta situatia\n\n"
        "REGULI:\n"
        "1. Uita-te la TONUL GENERAL si REZULTATUL FINAL\n"
        "2. Frustrarea implicita conteaza: ce sa fac, bine..., ironie\n\n"
        "EXEMPLE:\n" + exemple_text +
        "SATISFACTIE IDENTIFICATA:"
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
        "Genereaza un rezumat de tip " + tip + " al conversatiei.\n"
        "Lungime: " + str(tip_info["min_cuv"]) + "-" + str(tip_info["max_cuv"]) + " cuvinte. Limba: romana.\n\n"
        "EXEMPLU:\nCONVERSATIE:\n" + d_ex + "\nREZUMAT:\n" + r_ex + "\n\n"
        "CONVERSATIE:\n" + dialog + "\n\nREZUMAT " + tip + ":"
    )


# ─── ANALIZA LLM ─────────────────────────────────────────────────────────────

def analizeaza_dialog(dialog, domeniu):
    """Ruleaza cele 3 module LLM pe dialogul transcris."""

    # 1. Intentie
    raspuns_i, lat_i = call_llm(prompt_intentie(dialog, domeniu), max_tokens=30)
    intentie = extrage_intentie(raspuns_i, domeniu)

    # 2. Satisfactie
    raspuns_s, lat_s = call_llm(prompt_satisfactie(dialog), max_tokens=20)
    satisfactie = extrage_satisfactie(raspuns_s)

    # 3. Rezumat
    tip_info = get_tip_rezumat(satisfactie)
    raspuns_r, lat_r = call_llm(prompt_rezumat(dialog, satisfactie), max_tokens=200)
    nr_cuv = len(raspuns_r.split())
    in_limite = tip_info["min_cuv"] <= nr_cuv <= tip_info["max_cuv"]

    return {
        "intentie": {"valoare": intentie, "raspuns_brut": raspuns_i, "latenta": lat_i},
        "satisfactie": {"valoare": satisfactie, "raspuns_brut": raspuns_s, "latenta": lat_s},
        "rezumat": {
            "tip": tip_info["tip"], "valoare": raspuns_r,
            "nr_cuvinte": nr_cuv, "in_limite": in_limite, "latenta": lat_r
        },
        "latenta_analiza": round(lat_i + lat_s + lat_r, 3)
    }


# ─── PROCESARE UN FISIER ─────────────────────────────────────────────────────

def proceseaza_fisier(cale_fisier, domeniu):
    """Procesul complet: audio -> STT -> analiza LLM -> rezultat."""
    nume = Path(cale_fisier).stem
    print(f"\n{'='*60}")
    print(f"Fisier: {Path(cale_fisier).name} | Domeniu: {domeniu}")
    print(f"{'='*60}")

    rezultat = {
        "id": nume,
        "domeniu": domeniu,
        "fisier_audio": str(cale_fisier),
        "timestamp": datetime.now().isoformat(),
    }

    # STT
    try:
        dialog_transcris, lat_stt = transcrie_fisier_audio(str(cale_fisier), domeniu)
        rezultat["transcriere"] = {"text": dialog_transcris, "latenta_stt": lat_stt}
    except Exception as e:
        print(f"  [EROARE STT]: {e}")
        rezultat["eroare_stt"] = str(e)
        return rezultat

    if not dialog_transcris or not dialog_transcris.strip():
        print(f"  [ATENTIE]: Transcriere goala pentru {nume}")
        rezultat["eroare_stt"] = "Transcriere goala"
        return rezultat

    # Analiza LLM
    print(f"  [LLM] Analiza...")
    analiza = analizeaza_dialog(dialog_transcris, domeniu)
    rezultat["analiza"] = analiza

    # Afisare rezultate
    print(f"  Intentie:    {analiza['intentie']['valoare']}")
    print(f"  Satisfactie: {analiza['satisfactie']['valoare']}")
    print(f"  Rezumat ({analiza['rezumat']['tip']}, {analiza['rezumat']['nr_cuvinte']} cuv, "
          f"{'OK' if analiza['rezumat']['in_limite'] else 'OUT'}):")
    print(f"    {analiza['rezumat']['valoare'][:120]}...")
    print(f"  Latenta STT: {lat_stt}s | Analiza LLM: {analiza['latenta_analiza']}s | "
          f"Total: {round(lat_stt + analiza['latenta_analiza'], 2)}s")

    return rezultat


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline audio: STT + analiza LLM pe fisiere audio de call-center"
    )
    grup = parser.add_mutually_exclusive_group(required=True)
    grup.add_argument("--fisier", help="Cale catre un singur fisier audio (.mp3 sau .wav)")
    grup.add_argument("--folder", help="Folder cu mai multe fisiere audio")

    parser.add_argument("--domeniu",
                        choices=list(INTENTII_DOMENII.keys()),
                        help="Domeniu (necesar daca nu folosesti --auto_domeniu)")
    parser.add_argument("--auto_domeniu", action="store_true",
                        help="Detecteaza automat domeniul din numele fisierului (BNK/MED/RET/TEL/SP)")
    args = parser.parse_args()

    if not args.domeniu and not args.auto_domeniu:
        parser.error("Specifica --domeniu sau --auto_domeniu")

    # Colecteaza fisierele de procesat
    fisiere = []
    if args.fisier:
        fisiere = [Path(args.fisier)]
    else:
        folder = Path(args.folder)
        fisiere = sorted(list(folder.glob("*.mp3")) + list(folder.glob("*.wav")))
        if not fisiere:
            print(f"Niciun fisier audio gasit in {folder}")
            return

    print(f"\nFisiere de procesat: {len(fisiere)}")

    toate_rezultatele = []
    erori = []

    for cale in fisiere:
        # Determina domeniul — incearca automat, fallback la cel specificat
        domeniu = detecteaza_domeniu_din_nume(cale.name)
        if not domeniu:
            if args.domeniu:
                domeniu = args.domeniu
            else:
                print(f"\n[SKIP] Nu s-a putut detecta domeniu pentru: {cale.name}")
                erori.append({"fisier": str(cale), "eroare": "Domeniu nedetectat"})
                continue

        rezultat = proceseaza_fisier(cale, domeniu)
        toate_rezultatele.append(rezultat)

    # Salvare rezultate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(CONFIG["RESULTS_DIR"], f"rezultate_audio_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "nr_fisiere": len(fisiere),
            "nr_procesate": len(toate_rezultatele),
            "nr_erori": len(erori),
            "rezultate": toate_rezultatele,
            "erori": erori
        }, f, ensure_ascii=False, indent=2)

    # Sumar final
    procesate_ok = [r for r in toate_rezultatele if "analiza" in r]
    print(f"\n{'='*60}")
    print(f"SUMAR FINAL")
    print(f"{'='*60}")
    print(f"  Procesate cu succes: {len(procesate_ok)}/{len(fisiere)}")
    if procesate_ok:
        print(f"\n  {'Fisier':<20} {'Intentie':<30} {'Satisfactie':<12} {'Rezumat'}")
        print(f"  {'-'*80}")
        for r in procesate_ok:
            a = r["analiza"]
            print(f"  {r['id']:<20} {a['intentie']['valoare']:<30} "
                  f"{a['satisfactie']['valoare']:<12} "
                  f"{a['rezumat']['tip']} ({a['rezumat']['nr_cuvinte']}cuv)")
    print(f"\n  Rezultate salvate in: {output_file}")


if __name__ == "__main__":
    main()