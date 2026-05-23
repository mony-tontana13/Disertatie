"""
Robot telefonic call-center — GPT-4.1-mini + zevo STT/TTS
Domeniile disponibile: banking, medicina, retail, telecom, servicii_publice

Utilizare:
    python3 robot_telefonic.py
    python3 robot_telefonic.py --domeniu banking   # sare peste selectia domeniului
"""

import ssl
import certifi
ssl._create_default_https_context = ssl.create_default_context
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import os
import sys
import json
import time
import argparse
import unicodedata
import logging
from datetime import datetime
from openai import OpenAI

try:
    from zevo_tts import perform_text_to_speech
    from zevo_stt import record_and_transcribe
    ZEVO_DISPONIBIL = True
except ImportError:
    ZEVO_DISPONIBIL = False
    logging.warning("Modulele zevo nu sunt disponibile. Modul text activ.")

logging.basicConfig(level=logging.ERROR)

# ─── CONFIGURARE ─────────────────────────────────────────────────────────────

CONFIG = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    "MODEL": "gpt-4.1-mini",
    "TTS_API_KEY": "icvsilab2026",
    "TTS_VOICE": "gia",
    "STT_API_KEY": "icvsilab2026",
    "STT_DOMAIN": "ro-RO_general-2026.1",
    "RESULTS_DIR": "./rezultate_robot",
}

os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

# ─── DATE DOMENII ─────────────────────────────────────────────────────────────

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

NUME_DOMENII = {
    "banking": "serviciul bancar",
    "medicina": "clinica medicala",
    "retail": "serviciul clienti retail",
    "telecom": "operatorul de telecomunicatii",
    "servicii_publice": "serviciile publice",
}

SALUT_DOMENII = {
    "banking":          "Buna ziua, ati sunat la serviciul clienti al bancii. Cu ce va pot ajuta astazi?",
    "medicina":         "Buna ziua, ati sunat la clinica noastra medicala. Cu ce va pot ajuta?",
    "retail":           "Buna ziua, serviciul clienti, va ascult. Cu ce va pot ajuta?",
    "telecom":          "Buna ziua, ati contactat serviciul clienti al operatorului nostru. Cu ce va pot ajuta?",
    "servicii_publice": "Buna ziua, ati sunat la ghiseul de servicii publice. Cu ce va pot ajuta?",
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


# ─── SPEECH ───────────────────────────────────────────────────────────────────

def vorbeste(text):
    """Robotul vorbeste — TTS sau print."""
    print(f"\n[ROBOT]: {text}")
    if ZEVO_DISPONIBIL:
        perform_text_to_speech(CONFIG["TTS_API_KEY"], text, CONFIG["TTS_VOICE"])


def asculta(prompt_text=None):
    """Asculta clientul — STT sau input()."""
    if prompt_text:
        print(f"\n[ASTEAPTA INPUT CLIENT]")
    if ZEVO_DISPONIBIL:
        rezultat = record_and_transcribe(CONFIG["STT_API_KEY"], CONFIG["STT_DOMAIN"])
        if isinstance(rezultat, str) and rezultat.strip():
            print(f"[CLIENT]: {rezultat.strip()}")
            return rezultat.strip()
        else:
            print("[STT esuat, introduceti manual]")
    text = input("[CLIENT]: ").strip()
    return text


# ─── LLM ─────────────────────────────────────────────────────────────────────

def call_llm(prompt, max_tokens=100):
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


# ─── LOGICA ROBOT ─────────────────────────────────────────────────────────────

def detecteaza_intentie_si_detalii(dialog_pana_acum, domeniu):
    """Detecteaza intentia si ce detalii mai sunt necesare."""
    intentii = INTENTII_DOMENII.get(domeniu, [])
    prompt = (
        "Esti un sistem de analiza pentru un robot de call-center din domeniul " + domeniu + ".\n"
        "Analizeaza conversatia de pana acum si returneaza DOAR JSON valid:\n\n"
        "{\n"
        '  "intentie": "<una din lista sau alta_solicitare>",\n'
        '  "intentie_clara": true/false,\n'
        '  "detalii_lipsa": ["<detaliu1>", "<detaliu2>"],\n'
        '  "rezumat_problema": "<descriere scurta a problemei clientului>"\n'
        "}\n\n"
        "INTENTII DISPONIBILE: " + ", ".join(intentii) + "\n\n"
        "Detalii_lipsa = informatii esentiale pentru rezolvarea problemei care nu au fost mentionate inca.\n"
        "Exemple de detalii: numarul cardului, data tranzactiei, numarul comenzii, codul dosarului etc.\n"
        "Daca intentie_clara=false, detalii_lipsa trebuie sa contina cel putin un element.\n"
        "Daca intentie_clara=true si toate detaliile necesare sunt cunoscute, detalii_lipsa=[]\n\n"
        "CONVERSATIE:\n" + dialog_pana_acum + "\n\n"
        "JSON:"
    )
    raspuns, _ = call_llm(prompt, max_tokens=200)
    try:
        if raspuns.startswith("```"):
            raspuns = raspuns.replace("```json", "").replace("```", "").strip()
        return json.loads(raspuns)
    except Exception:
        return {
            "intentie": "alta_solicitare",
            "intentie_clara": False,
            "detalii_lipsa": ["problema clientului"],
            "rezumat_problema": ""
        }


def genereaza_replica_robot(dialog_pana_acum, domeniu, context):
    """Genereaza replica naturala a robotului bazata pe context."""
    prompt = (
        "Esti un operator robot de call-center din domeniul " + NUME_DOMENII.get(domeniu, domeniu) + ".\n"
        "Vorbesti in romana, politicos, concis si profesional.\n"
        "Nu esti un om — esti un sistem automat, dar vorbesti natural.\n\n"
        "CONTEXT CURENT:\n"
        "- Intentia identificata: " + context.get("intentie", "necunoscuta") + "\n"
        "- Problema clientului: " + context.get("rezumat_problema", "neprecizata") + "\n"
        "- Detalii care lipsesc: " + (", ".join(context.get("detalii_lipsa", [])) or "niciuna") + "\n"
        "- Faza conversatiei: " + context.get("faza", "colectare_informatii") + "\n\n"
        "CONVERSATIE PANA ACUM:\n" + dialog_pana_acum + "\n\n"
        "INSTRUCTIUNI:\n"
        "- Daca lipsesc detalii, cere-le politicos pe rand, nu pe toate odata\n"
        "- Daca ai toate informatiile, ofera o solutie sau un raspuns concret\n"
        "- Daca nu poti rezolva direct, explica pasii urmatori si ce se va intampla\n"
        "- Nu inventa informatii specifice (conturi, sume exacte etc.)\n"
        "- Fii empatic daca clientul este frustrat\n"
        "- Raspunde in maxim 2-3 propozitii\n\n"
        "REPLICA ROBOT:"
    )
    replica, lat = call_llm(prompt, max_tokens=150)
    return replica, lat


def genereaza_solutie_finala(dialog_complet, domeniu, intentie):
    """Genereaza raspunsul/solutia finala dupa colectarea tuturor detaliilor."""
    prompt = (
        "Esti un operator robot de call-center din domeniul " + NUME_DOMENII.get(domeniu, domeniu) + ".\n"
        "Ai colectat toate informatiile necesare de la client.\n"
        "Intentia identificata: " + intentie + "\n\n"
        "CONVERSATIE COMPLETA:\n" + dialog_complet + "\n\n"
        "Ofera acum un raspuns final complet si o solutie clara pentru problema clientului.\n"
        "Fii specific, mentionand pasii concreti sau termenele de rezolvare.\n"
        "Daca nu poti rezolva direct, explica clar ce se va intampla si in cat timp.\n"
        "Incheie cu o fraza de politete.\n"
        "Raspunde in 3-5 propozitii, in romana.\n\n"
        "RASPUNS FINAL:"
    )
    replica, lat = call_llm(prompt, max_tokens=200)
    return replica, lat


# ─── ANALIZA FINALA ───────────────────────────────────────────────────────────

def analizeaza_conversatie(dialog_complet, domeniu):
    """Ruleaza cele 3 module de analiza pe conversatia completa."""
    print("\n" + "="*60)
    print("ANALIZA AUTOMATA A CONVERSATIEI")
    print("="*60)

    # 1. Intentie
    intentii = INTENTII_DOMENII.get(domeniu, [])
    prompt_i = (
        "Lucrezi ca analist de date intr-un call-center din domeniul " + domeniu + ".\n"
        "Identifica intentia principala a clientului din conversatia de mai jos.\n"
        "Alege EXCLUSIV din lista: " + ", ".join(intentii) + "\n\n"
        "Conversatie:\n" + dialog_complet + "\n\n"
        "Intentie identificata:"
    )
    raspuns_i, lat_i = call_llm(prompt_i, max_tokens=30)
    intentie = extrage_intentie(raspuns_i, domeniu)
    print(f"  Intentie:    {intentie} ({lat_i}s)")

    # 2. Satisfactie
    exemple = {
        "pozitiv": ("CLIENT: Multumesc mult, sunteti promti!", "pozitiv"),
        "neutru":  ("CLIENT: Ok, am inteles.", "neutru"),
        "negativ": ("CLIENT: Bine, ce sa fac...", "negativ"),
    }
    exemple_text = "".join(
        "CONVERSATIE:\n" + d + "\nSATISFACTIE: " + s + "\n\n"
        for _, (d, s) in exemple.items()
    )
    prompt_s = (
        "Esti un expert in analiza satisfactiei clientilor in conversatii de call-center.\n\n"
        "CONVERSATIE:\n" + dialog_complet + "\n\n"
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
    raspuns_s, lat_s = call_llm(prompt_s, max_tokens=20)
    satisfactie = extrage_satisfactie(raspuns_s)
    print(f"  Satisfactie: {satisfactie} ({lat_s}s)")

    # 3. Rezumat
    tip_info = get_tip_rezumat(satisfactie)
    tip = tip_info["tip"]
    exemple_rez = {
        "SCURT": ("CLIENT: Am pierdut cardul.\nOPERATOR: L-am blocat imediat.",
                  "Clientul a solicitat blocarea unui card pierdut. Operatorul a rezolvat imediat."),
        "MEDIU": ("CLIENT: Comanda nu a sosit.\nOPERATOR: Intarziere la curier, vine maine.\nCLIENT: Ok.",
                  "Clientul a reclamat o comanda nelivrata. Operatorul a identificat o intarziere si a reprogramat livrarea. Clientul a acceptat."),
        "LUNG":  ("CLIENT: A treia oara sun pentru aceeasi problema.\nOPERATOR: Investigam.\nCLIENT: Astept.",
                  "Clientul a contactat call-center-ul pentru a treia oara pentru aceeasi problema nerezolvata. Clientul si-a exprimat nemultumirea. Operatorul a initiat o investigatie. Clientul a acceptat sa astepte, exprimand frustrare evidenta. Problema ramane deschisa."),
    }
    d_ex, r_ex = exemple_rez[tip]
    prompt_r = (
        "Genereaza un rezumat de tip " + tip + " al conversatiei de mai jos.\n"
        "Lungime: " + str(tip_info["min_cuv"]) + "-" + str(tip_info["max_cuv"]) + " cuvinte.\n"
        "Limba: romana. Mentioneaza problema si rezultatul final.\n\n"
        "EXEMPLU:\nCONVERSATIE:\n" + d_ex + "\nREZUMAT:\n" + r_ex + "\n\n"
        "CONVERSATIE:\n" + dialog_complet + "\n\nREZUMAT " + tip + ":"
    )
    raspuns_r, lat_r = call_llm(prompt_r, max_tokens=200)
    nr_cuv = len(raspuns_r.split())
    in_limite = tip_info["min_cuv"] <= nr_cuv <= tip_info["max_cuv"]
    print(f"  Rezumat:     {tip} ({nr_cuv} cuv, {'OK' if in_limite else 'OUT'}) ({lat_r}s)")
    print(f"\n  TEXT REZUMAT:\n  {raspuns_r}")

    return {
        "intentie": intentie,
        "satisfactie": satisfactie,
        "rezumat": {
            "tip": tip,
            "valoare": raspuns_r,
            "nr_cuvinte": nr_cuv,
            "in_limite": in_limite,
        },
        "latenta_analiza": round(lat_i + lat_s + lat_r, 3)
    }


# ─── CONVERSATIE PRINCIPALA ───────────────────────────────────────────────────

def selecteaza_domeniu():
    """Lasa clientul sa aleaga domeniul la inceput."""
    domenii_lista = list(INTENTII_DOMENII.keys())
    vorbeste(
        "Buna ziua! Ati sunat la serviciul nostru de relatii cu clientii. "
        "Va rog sa specificati domeniul pentru care sunati: "
        "banking, medicina, retail, telecom sau servicii publice."
    )
    while True:
        raspuns = asculta()
        raspuns_norm = normalizeaza(raspuns)
        for domeniu in domenii_lista:
            if domeniu in raspuns_norm or normalizeaza(NUME_DOMENII[domeniu]) in raspuns_norm:
                return domeniu
        # Fallback LLM pentru detectare domeniu
        prompt = (
            "Din urmatorul text, identifica despre ce domeniu vorbeste clientul.\n"
            "Domenii posibile: banking, medicina, retail, telecom, servicii_publice\n"
            "Returneaza DOAR unul din aceste cuvinte, nimic altceva.\n"
            "Text: " + raspuns
        )
        raspuns_llm, _ = call_llm(prompt, max_tokens=10)
        for domeniu in domenii_lista:
            if domeniu in normalizeaza(raspuns_llm):
                return domeniu
        vorbeste(
            "Nu am inteles domeniul. Va rog sa alegeti dintre: "
            "banking, medicina, retail, telecom sau servicii publice."
        )


def ruleaza_conversatie(domeniu):
    """
    Conduce conversatia completa cu clientul.
    Returneaza dialogul complet si rezultatele analizei.
    """
    replici = []  # lista de dict {rol, text}
    conv_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def adauga_replica(rol, text):
        replici.append({"rol": rol, "text": text})

    def get_dialog():
        return "\n".join(f"{r['rol'].upper()}: {r['text']}" for r in replici)

    # ── SALUT INITIAL ─────────────────────────────────────────────────────────
    salut = SALUT_DOMENII[domeniu]
    vorbeste(salut)
    adauga_replica("operator", salut)

    # ── BUCLA PRINCIPALA ──────────────────────────────────────────────────────
    MAX_TURE = 8  # max 8 schimburi inainte de inchidere fortata
    tura = 0
    intentie_finala = "alta_solicitare"
    colectare_completa = False

    while tura < MAX_TURE and not colectare_completa:
        # Asculta clientul
        text_client = asculta()
        if not text_client:
            continue

        # Detecteaza daca clientul vrea sa incheie
        if normalizeaza(text_client) in ["la revedere", "pa", "gata", "inchid", "multumesc la revedere"]:
            adauga_replica("client", text_client)
            break

        adauga_replica("client", text_client)
        tura += 1

        # Analiza context curent
        context = detecteaza_intentie_si_detalii(get_dialog(), domeniu)
        intentie_finala = context.get("intentie", "alta_solicitare")

        # Decide faza conversatiei
        detalii_lipsa = context.get("detalii_lipsa", [])
        intentie_clara = context.get("intentie_clara", False)

        if intentie_clara and len(detalii_lipsa) == 0:
            # Toate informatiile sunt colectate — ofera solutia finala
            colectare_completa = True
            context["faza"] = "solutie_finala"
            replica_robot, _ = genereaza_solutie_finala(get_dialog(), domeniu, intentie_finala)
        elif tura >= MAX_TURE - 1:
            # Ultima sansa — incearca sa ofere un raspuns oricum
            context["faza"] = "solutie_finala"
            replica_robot, _ = genereaza_solutie_finala(get_dialog(), domeniu, intentie_finala)
        else:
            # Continua colectarea de informatii
            context["faza"] = "colectare_informatii"
            replica_robot, _ = genereaza_replica_robot(get_dialog(), domeniu, context)

        vorbeste(replica_robot)
        adauga_replica("operator", replica_robot)

    # ── INCHEIERE ─────────────────────────────────────────────────────────────
    # Asculta ultima replica a clientului daca conversatia s-a incheiat natural
    if colectare_completa:
        text_client_final = asculta()
        if text_client_final:
            adauga_replica("client", text_client_final)

    incheiere = "Va multumim ca ati contactat serviciul nostru. O zi buna!"
    vorbeste(incheiere)
    adauga_replica("operator", incheiere)

    # ── ANALIZA ───────────────────────────────────────────────────────────────
    dialog_complet = get_dialog()
    analiza = analizeaza_conversatie(dialog_complet, domeniu)

    # ── SALVARE ───────────────────────────────────────────────────────────────
    rezultat = {
        "id": conv_id,
        "domeniu": domeniu,
        "timestamp": datetime.now().isoformat(),
        "conversatie": replici,
        "analiza": analiza
    }
    fisier = os.path.join(CONFIG["RESULTS_DIR"], f"conversatie_{conv_id}.json")
    with open(fisier, "w", encoding="utf-8") as f:
        json.dump(rezultat, f, ensure_ascii=False, indent=2)
    print(f"\nConversatie salvata in: {fisier}")

    return rezultat


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Robot telefonic call-center")
    parser.add_argument(
        "--domeniu", choices=list(INTENTII_DOMENII.keys()),
        help="Domeniu predefinit (sare peste selectia initiala)"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("ROBOT TELEFONIC CALL-CENTER")
    print("="*60 + "\n")

    # Selectare domeniu
    if args.domeniu:
        domeniu = args.domeniu
        print(f"Domeniu predefinit: {domeniu}")
    else:
        domeniu = selecteaza_domeniu()

    print(f"\nDomeniu selectat: {domeniu}\n")

    # Ruleaza conversatia
    rezultat = ruleaza_conversatie(domeniu)

    # Sumar final
    analiza = rezultat["analiza"]
    print("\n" + "="*60)
    print("SUMAR FINAL")
    print("="*60)
    print(f"  Intentie:    {analiza['intentie']}")
    print(f"  Satisfactie: {analiza['satisfactie']}")
    print(f"  Rezumat:     {analiza['rezumat']['tip']} — {analiza['rezumat']['nr_cuvinte']} cuvinte")
    print(f"  Latenta analiza: {analiza['latenta_analiza']}s")


if __name__ == "__main__":
    main()