"""
Pipeline final — Sistem de analiza conversatii call-center
Modele: GPT-4.1-mini (OpenAI)
Speech: zevo_stt (STT) + zevo_tts (TTS)

Utilizare:
    # Mod live — inregistreaza conversatia prin microfon
    python3 pipeline_final.py --mod live

    # Mod fisier — proceseaza o conversatie pre-inregistrata din JSON
    python3 pipeline_final.py --mod fisier --fisier conversatie.json

    # Mod text — primeste dialogul ca text direct
    python3 pipeline_final.py --mod text
"""

import os
import sys
import json
import time
import argparse
import unicodedata
import logging
from datetime import datetime
from openai import OpenAI

# ─── ZEVO ────────────────────────────────────────────────────────────────────

try:
    from zevo_tts import perform_text_to_speech
    from zevo_stt import record_and_transcribe
    ZEVO_DISPONIBIL = True
except ImportError:
    ZEVO_DISPONIBIL = False
    logging.warning("Modulele zevo_tts/zevo_stt nu sunt disponibile. Modul text activ.")

# ─── CONFIGURARE ─────────────────────────────────────────────────────────────

CONFIG = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    "MODEL": "gpt-4.1-mini",
    "TTS_API_KEY": "icvsilab2026",
    "TTS_VOICE": "maria",
    "STT_API_KEY": "icvsilab2026",
    "STT_DOMAIN": "ro-RO_general-2025.1",
    "RESULTS_DIR": "./rezultate_pipeline",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

# ─── INTENTII DISPONIBILE ─────────────────────────────────────────────────────

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


def extrage_intentie(raspuns, domeniu):
    intentii_valide = INTENTII_DOMENII.get(domeniu, [])
    raspuns_norm = normalizeaza(raspuns)
    for intentie in intentii_valide:
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


# ─── LLM CALLS ───────────────────────────────────────────────────────────────

def call_llm(prompt, max_tokens=50):
    """Apeleaza GPT-4.1-mini si returneaza raspunsul text."""
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
    latenta = round(time.time() - start, 3)
    return raspuns.strip(), latenta


# ─── PROMPTURI CASTIGATOARE ───────────────────────────────────────────────────

def prompt_intentie(dialog, domeniu):
    """V2-1 — cel mai bun pentru GPT-4.1-mini."""
    intentii = INTENTII_DOMENII.get(domeniu, [])
    return (
        "Lucrezi ca analist de date intr-un call-center din domeniul " + domeniu + ". "
        "Sarcina ta zilnica este sa identifici motivul pentru care clientii suna, "
        "pe baza transcripturilor conversatiilor cu operatorii.\n\n"
        "REGULI:\n"
        "- Include DOAR ce a cerut sau intrebat clientul, nu actiunile operatorului\n"
        "- Alege DOAR din lista de intentii de mai jos\n\n"
        "INTENTII DISPONIBILE: " + ", ".join(intentii) + "\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "De ce a sunat clientul? Raspunde cu una sau doua intentii din lista:"
    )


def prompt_satisfactie(dialog):
    """V3-2 — cel mai bun pentru GPT-4.1-mini."""
    exemple = {
        "pozitiv": (
            "CLIENT: Buna ziua, am pierdut cardul.\nOPERATOR: L-am blocat imediat, va trimitem altul.\nCLIENT: Multumesc mult, sunteti promti!",
            "pozitiv"
        ),
        "neutru": (
            "CLIENT: Comanda mea nu a sosit.\nOPERATOR: Va ajunge maine.\nCLIENT: Ok, am inteles.",
            "neutru"
        ),
        "negativ": (
            "CLIENT: Am sunat a treia oara pentru aceeasi problema.\nOPERATOR: Investigam.\nCLIENT: Bine, ce sa fac...",
            "negativ"
        ),
    }
    exemple_text = ""
    for clasa, (dialog_ex, sat_ex) in exemple.items():
        exemple_text += "CONVERSATIE:\n" + dialog_ex + "\nSATISFACTIE: " + sat_ex + "\n\n"

    return (
        "Esti un expert in analiza satisfactiei clientilor in conversatii de call-center.\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "SARCINA: Determina nivelul de satisfactie al clientului.\n\n"
        "DEFINITII:\n"
        "- pozitiv: problema rezolvata complet, clientul multumit si o exprima clar\n"
        "- neutru: problema rezolvata tehnic dar clientul nu prezinta nicio emotie\n"
        "- negativ: clientul pleaca frustrat sau nemultumit, chiar daca implicit\n\n"
        "REGULI:\n"
        "1. Un singur comentariu negativ urmat de acceptare NU inseamna automat negativ\n"
        "2. Uita-te la TONUL GENERAL si REZULTATUL FINAL al conversatiei\n"
        "3. Frustrarea implicita conteaza: bine inteleg, ce sa fac, remarci ironice\n\n"
        "EXEMPLE (acorda atentie diferentei dintre neutru si negativ):\n" + exemple_text +
        "SATISFACTIE IDENTIFICATA:"
    )


def prompt_rezumat(dialog, satisfactie):
    """V3-1 — cel mai bun pentru GPT-4.1-mini."""
    tip_info = get_tip_rezumat(satisfactie)
    tip = tip_info["tip"]
    exemple = {
        "SCURT": (
            "CLIENT: Am pierdut cardul, il vreau blocat.\nOPERATOR: L-am blocat, va trimitem altul in 3 zile.",
            "Clientul a sunat pentru a bloca un card pierdut. Operatorul a blocat cardul si a initiat emiterea unuia nou."
        ),
        "MEDIU": (
            "CLIENT: Comanda mea nu a sosit.\nOPERATOR: A fost o intarziere la curier. Va ajunge maine.\nCLIENT: Ok.",
            "Clientul a reclamat o comanda nelivrata la termen. Operatorul a verificat situatia si a identificat o intarziere la curier. Comanda urmeaza sa fie livrata a doua zi. Clientul a acceptat solutia."
        ),
        "LUNG": (
            "CLIENT: Am sunat de trei ori pentru aceeasi problema cu factura.\nOPERATOR: Investigam.\nCLIENT: Astept.",
            "Clientul a contactat call-center-ul pentru a treia oara in legatura cu aceeasi problema de facturare nerezolvata. Clientul si-a exprimat nemultumirea fata de lipsa unei solutii. Operatorul a initiat o investigatie interna. Clientul a acceptat sa astepte, exprimand frustrare evidenta. Problema ramane deschisa si necesita urmarire."
        ),
    }
    dialog_ex, rezumat_ex = exemple[tip]
    return (
        "Esti un expert in sumarizarea conversatiilor telefonice din call-center.\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "SARCINA: Genereaza un rezumat de tip " + tip + ".\n\n"
        "CERINTE:\n"
        "- Lungime: " + str(tip_info["min_cuv"]) + "-" + str(tip_info["max_cuv"]) + " cuvinte (" + tip_info["propozitii"] + ")\n"
        "- Limba: romana\n"
        "- Mentioneaza problema principala si rezultatul final\n"
        "- Nu adauga informatii care nu apar in conversatie\n\n"
        "EXEMPLU:\nCONVERSATIE:\n" + dialog_ex + "\nREZUMAT " + tip + ":\n" + rezumat_ex + "\n\n"
        "REZUMAT " + tip + ":"
    )


# ─── SPEECH (ZEVO) ───────────────────────────────────────────────────────────

def tts(text):
    """Reda textul prin zevo TTS. Fallback: print."""
    print(f"\n[SISTEM]: {text}")
    if ZEVO_DISPONIBIL:
        perform_text_to_speech(CONFIG["TTS_API_KEY"], text, CONFIG["TTS_VOICE"])


def stt():
    """Inregistreaza si transcrie prin zevo STT. Fallback: input()."""
    if ZEVO_DISPONIBIL:
        rezultat = record_and_transcribe(CONFIG["STT_API_KEY"], CONFIG["STT_DOMAIN"])
        if isinstance(rezultat, str) and rezultat.strip():
            print(f"[TRANSCRIS]: {rezultat.strip()}")
            return rezultat.strip()
        else:
            print("[STT esuat, introduceti manual]")
    return input("> ").strip()


# ─── INREGISTRARE CONVERSATIE LIVE ───────────────────────────────────────────

def inregistreaza_conversatie_live(domeniu):
    """
    Inregistreaza o conversatie live prin microfon.
    Alterneaza intre replica operator si replica client.
    Returneaza dialogul formatat ca string.
    """
    tts(f"Incepem inregistrarea conversatiei din domeniul {domeniu}.")
    tts("Voi alterna intre operator si client. Spuneti GATA cand ati terminat o replica.")

    replici = []
    roluri = ["OPERATOR", "CLIENT"]
    idx = 0

    while True:
        rol = roluri[idx % 2]
        tts(f"Replica {rol}. Vorbiti:")
        text_replica = stt()

        if not text_replica:
            continue
        if normalizeaza(text_replica) in ["gata", "stop", "incheiat", "final"]:
            break

        replici.append(f"{rol}: {text_replica}")
        idx += 1

        if idx >= 2:
            tts("Continuati sau spuneti GATA pentru a incheia conversatia.")

    dialog = "\n".join(replici)
    return dialog


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def ruleaza_pipeline(dialog, domeniu, conv_id=None):
    """
    Ruleaza cele 3 module LLM pe dialogul dat.
    Returneaza un dict cu toate rezultatele.
    """
    if not conv_id:
        conv_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"PIPELINE ANALIZA CONVERSATIE — {conv_id}")
    print(f"Domeniu: {domeniu}")
    print(f"{'='*60}\n")

    rezultat = {
        "id": conv_id,
        "domeniu": domeniu,
        "timestamp": datetime.now().isoformat(),
        "dialog": dialog,
    }

    # ── 1. DETECTARE INTENTIE ─────────────────────────────────────────────────
    print("[1/3] Detectare intentie...")
    raspuns_i, lat_i = call_llm(prompt_intentie(dialog, domeniu), max_tokens=30)
    intentie = extrage_intentie(raspuns_i, domeniu)

    print(f"  Intentie identificata: {intentie}")
    print(f"  Raspuns brut: {raspuns_i[:80]}")
    print(f"  Latenta: {lat_i}s")

    rezultat["intentie"] = {
        "valoare": intentie,
        "raspuns_brut": raspuns_i,
        "latenta": lat_i
    }

    # ── 2. ESTIMARE SATISFACTIE ───────────────────────────────────────────────
    print("\n[2/3] Estimare satisfactie...")
    raspuns_s, lat_s = call_llm(prompt_satisfactie(dialog), max_tokens=20)
    satisfactie = extrage_satisfactie(raspuns_s)

    print(f"  Satisfactie identificata: {satisfactie}")
    print(f"  Raspuns brut: {raspuns_s[:80]}")
    print(f"  Latenta: {lat_s}s")

    rezultat["satisfactie"] = {
        "valoare": satisfactie,
        "raspuns_brut": raspuns_s,
        "latenta": lat_s
    }

    # ── 3. GENERARE REZUMAT ───────────────────────────────────────────────────
    print("\n[3/3] Generare rezumat...")
    tip_info = get_tip_rezumat(satisfactie)
    raspuns_r, lat_r = call_llm(prompt_rezumat(dialog, satisfactie), max_tokens=200)

    nr_cuv = len(raspuns_r.split())
    in_limite = tip_info["min_cuv"] <= nr_cuv <= tip_info["max_cuv"]

    print(f"  Tip rezumat: {tip_info['tip']} ({tip_info['min_cuv']}-{tip_info['max_cuv']} cuvinte)")
    print(f"  Cuvinte generate: {nr_cuv} — {'OK' if in_limite else 'IN AFARA LIMITELOR'}")
    print(f"  Latenta: {lat_r}s")
    print(f"\n  REZUMAT:\n  {raspuns_r}")

    rezultat["rezumat"] = {
        "tip": tip_info["tip"],
        "valoare": raspuns_r,
        "nr_cuvinte": nr_cuv,
        "in_limite": in_limite,
        "latenta": lat_r
    }

    # ── SUMAR ─────────────────────────────────────────────────────────────────
    latenta_totala = lat_i + lat_s + lat_r
    rezultat["latenta_totala"] = round(latenta_totala, 3)

    print(f"\n{'─'*60}")
    print(f"SUMAR: Intentie={intentie} | Satisfactie={satisfactie} | "
          f"Rezumat={tip_info['tip']} ({nr_cuv} cuv) | Latenta={latenta_totala:.2f}s")
    print(f"{'─'*60}\n")

    return rezultat


def reda_rezultate(rezultat):
    """Reda rezultatele prin TTS."""
    intentie = rezultat["intentie"]["valoare"]
    satisfactie = rezultat["satisfactie"]["valoare"]
    rezumat = rezultat["rezumat"]["valoare"]

    tts(f"Analiza conversatiei este completa.")
    tts(f"Intentia identificata este: {intentie.replace('_', ' ')}.")
    tts(f"Nivelul de satisfactie al clientului este: {satisfactie}.")
    tts(f"Rezumatul conversatiei: {rezumat}")


def salveaza_rezultat(rezultat):
    """Salveaza rezultatul in JSON."""
    fisier = os.path.join(
        CONFIG["RESULTS_DIR"],
        f"rezultat_{rezultat['id']}.json"
    )
    with open(fisier, "w", encoding="utf-8") as f:
        json.dump(rezultat, f, ensure_ascii=False, indent=2)
    print(f"Rezultat salvat in: {fisier}")
    return fisier


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline analiza conversatii call-center cu GPT-4.1-mini si zevo"
    )
    parser.add_argument(
        "--mod", choices=["live", "fisier", "text"],
        default="text",
        help="Modul de intrare: live (microfon), fisier (JSON), text (stdin)"
    )
    parser.add_argument(
        "--fisier",
        help="Calea catre fisierul JSON de conversatie (pentru --mod fisier)"
    )
    parser.add_argument(
        "--domeniu",
        choices=list(INTENTII_DOMENII.keys()),
        default="banking",
        help="Domeniul conversatiei (default: banking)"
    )
    parser.add_argument(
        "--tts", action="store_true",
        help="Reda rezultatele prin TTS dupa analiza"
    )
    args = parser.parse_args()

    dialog = None
    domeniu = args.domeniu
    conv_id = None

    # ── MOD FISIER ────────────────────────────────────────────────────────────
    if args.mod == "fisier":
        if not args.fisier:
            print("Eroare: specificati --fisier pentru modul fisier.")
            sys.exit(1)
        with open(args.fisier, encoding="utf-8") as f:
            conv = json.load(f)
        dialog = "\n".join([
            r["rol"].upper() + ": " + r["text"]
            for r in conv.get("conversatie", [])
        ])
        domeniu = conv.get("domeniu", domeniu)
        conv_id = conv.get("id", None)
        print(f"Conversatie incarcata din fisier: {args.fisier}")

    # ── MOD LIVE ──────────────────────────────────────────────────────────────
    elif args.mod == "live":
        if not ZEVO_DISPONIBIL:
            print("Eroare: modulele zevo nu sunt disponibile pentru modul live.")
            sys.exit(1)

        tts("Buna ziua! Sistemul de analiza conversatii este activ.")
        tts(f"Domeniu selectat: {domeniu}.")

        dialog = inregistreaza_conversatie_live(domeniu)
        if not dialog:
            tts("Nu a fost inregistrata nicio conversatie.")
            sys.exit(0)

    # ── MOD TEXT ──────────────────────────────────────────────────────────────
    elif args.mod == "text":
        print(f"Introduceti dialogul (domeniu: {domeniu}).")
        print("Formatul asteptat: 'OPERATOR: text' / 'CLIENT: text', câte o replica pe linie.")
        print("Terminati cu o linie goala.\n")
        linii = []
        while True:
            linie = input()
            if not linie:
                break
            linii.append(linie)
        dialog = "\n".join(linii)

    if not dialog or not dialog.strip():
        print("Eroare: niciun dialog disponibil pentru analiza.")
        sys.exit(1)

    # ── PIPELINE ──────────────────────────────────────────────────────────────
    rezultat = ruleaza_pipeline(dialog, domeniu, conv_id)

    # ── TTS OPTIONAL ──────────────────────────────────────────────────────────
    if args.tts and ZEVO_DISPONIBIL:
        reda_rezultate(rezultat)

    # ── SALVARE ───────────────────────────────────────────────────────────────
    salveaza_rezultat(rezultat)


if __name__ == "__main__":
    main()
