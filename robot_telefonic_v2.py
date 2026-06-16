"""
Robot telefonic v2 — flux complet cu solutii predefinite + LLM personalizat
Fluxul:
  1. Salut + selectie domeniu
  2. STT -> LLM extrage intentia
  3. Colectare detalii necesare (LLM intreaba, STT asculta)
  4. Solutie: predefinita (generic) sau LLM (personalizat)
  5. Confirmare client
  6. Analiza finala: intentie + satisfactie + rezumat -> JSON

Utilizare:
    python3 robot_telefonic_v2.py
    python3 robot_telefonic_v2.py --domeniu banking
"""

import os
import json
import time
import random
import asyncio
import argparse
import unicodedata
from datetime import datetime
from openai import OpenAI
from solutii_predefinite import SOLUTII_PREDEFINITE, DETALII_NECESARE

from zevo_tts import perform_text_to_speech
import websockets
import ssl
import speech_recognition as sr
ZEVO_DISPONIBIL = True

async def _stt_ws(audio_data, api_key, domain, sample_rate=16000, chunk_size=4096,
                  server_uri="wss://live-transcriber.zevo-tech.com:2053"):
    """WebSocket STT — identic cu logica din zevo_stt.py + SSL fix."""
    config_msg = json.dumps({
        "config": {"key": api_key, "sample_rate": str(sample_rate), "domain": domain}
    })
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    rezultate = []
    partial_curent = ""
    sleep_per_chunk = chunk_size / (sample_rate * 2)

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
            elif "text" in parsed and parsed["text"].strip():
                rezultate.append(parsed["text"].strip())
            elif "partial" in parsed and parsed["partial"].strip():
                p = parsed["partial"].strip()
                if p not in " ".join(rezultate):
                    rezultate.append(p)
        except Exception:
            pass
    return " ".join(rezultate)

def record_and_transcribe(api_key, domain):
    """Inregistreaza de la microfon si transcrie cu zevo STT."""
    r = sr.Recognizer()
    try:
        r.pause_threshold = 1.2        # asteapta 1.2 sec de liniste — echilibru intre latenta si completitudine
        r.phrase_threshold = 0.1       # sensibil la inceput de vorbire
        r.non_speaking_duration = 1.0  # durata minima de liniste considerata pauza
        with sr.Microphone(sample_rate=16000) as source:
            print("Say something...")
            r.adjust_for_ambient_noise(source, duration=0.3)  # redus de la 0.5 la 0.3
            audio = r.listen(source, phrase_time_limit=20)  # max 20 sec per replica
        wav_data = audio.get_wav_data()
        text = asyncio.run(_stt_ws(wav_data, api_key, domain))
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

# ─── CONFIGURARE ─────────────────────────────────────────────────────────────

CONFIG = {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    "MODEL": "gpt-4.1-mini",
    "TTS_API_KEY": "icvsilab2026",
    "TTS_VOICE": "maria",
    "STT_API_KEY": "icvsilab2026",
    "STT_DOMAIN": "ro-RO_general-2026.1",
    "RESULTS_DIR": "./rezultate_robot",
}

os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
client = OpenAI(api_key=CONFIG["OPENAI_API_KEY"])

# ─── DATE DOMENII ─────────────────────────────────────────────────────────────

from date_robot import (
    INTENTII_DOMENII, SALUT_DOMENII, REPLICI_IDENTIFICARE,
    REPLICA_INCHEIERE, REPLICA_OPERATOR, NUME_DOMENII,
    TIP_REZUMAT, NIVELURI_DIFICULTATE, DETALII_NECESARE
)

def alege_dificultate():
    chei = list(NIVELURI_DIFICULTATE.keys())
    ponderi = [NIVELURI_DIFICULTATE[k]["pondere"] for k in chei]
    ales = random.choices(chei, weights=ponderi)[0]
    print(f"  [DIFICULTATE] {ales.upper()}")
    return ales

# ─── UTILITARE ────────────────────────────────────────────────────────────────

def normalizeaza(text):
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower().strip()

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

def client_vrea_sa_incheie(text):
    """
    Returneaza True DOAR cand clientul spune explicit ca vrea sa incheie.
    Foarte strict — evita false pozitive din cuvinte care contin silabe ca "pa", "da" etc.
    """
    import re
    norm = normalizeaza(text)

    # Fraze complete de incheiere — trebuie sa apara ca atare in text
    fraze_complete = [
        "la revedere",
        "multumesc la revedere",
        "nu mai am intrebari",
        "nu mai am nimic",
        "asta era tot",
        "asta e tot",
        "am rezolvat tot",
        "gata am inteles",
        "va multumesc la revedere",
    ]
    for fraza in fraze_complete:
        if fraza in norm:
            return True

    # "pa" DOAR daca e singurul cuvant sau urmat doar de "pa" (ex: "pa pa")
    if re.fullmatch(r'pa[\s!.]*', norm.strip()) or re.fullmatch(r'pa pa[\s!.]*', norm.strip()):
        return True

    # "gata" DOAR daca textul e scurt (max 3 cuvinte) si nu contine alte informatii
    if re.fullmatch(r'(ok\s+)?gata[\s!.]*', norm.strip()):
        return True

    return False

# ─── SPEECH ───────────────────────────────────────────────────────────────────

def vorbeste(text):
    print(f"\n  [ROBOT]: {text}")
    if ZEVO_DISPONIBIL:
        perform_text_to_speech(CONFIG["TTS_API_KEY"], text, CONFIG["TTS_VOICE"])

def deduplica_transcriere(text):
    """Elimina repetitiile evidente din transcrierea STT."""
    if not text or len(text.split()) < 4:
        return text
    prompt = (
        "Urmatorul text este o transcriere STT care poate contine repetitii "
        "ale aceleiasi fraze. Returneaza DOAR varianta curata, fara repetitii, "
        "fara explicatii suplimentare.\n\nTEXT: " + text + "\n\nTEXT CURAT:"
    )
    raspuns, _ = call_llm(prompt, max_tokens=100)
    return raspuns.strip() if raspuns.strip() else text

def asculta():
    if ZEVO_DISPONIBIL:
        rezultat = record_and_transcribe(CONFIG["STT_API_KEY"], CONFIG["STT_DOMAIN"])
        if isinstance(rezultat, str) and rezultat.strip():
            text_curat = deduplica_transcriere(rezultat.strip())
            print(f"  [CLIENT]: {text_curat}")
            return text_curat
        print("  [STT esuat, input manual]")
    return input("  [CLIENT]: ").strip()

# ─── LLM ─────────────────────────────────────────────────────────────────────

def call_llm(prompt, max_tokens=150):
    start = time.time()
    raspuns = ""
    for chunk in client.chat.completions.create(
        model=CONFIG["MODEL"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, stream=True
    ):
        if chunk.choices[0].delta.content:
            raspuns += chunk.choices[0].delta.content
    return raspuns.strip(), round(time.time() - start, 2)

def extrage_intentie_llm(dialog, domeniu):
    intentii = INTENTII_DOMENII.get(domeniu, [])
    prompt = (
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
    raspuns, _ = call_llm(prompt, max_tokens=20)
    return extrage_intentie(raspuns, domeniu)

def genereaza_replica_robot(dialog, domeniu, intentie, dificultate, faza):
    """Un singur LLM care stie tot contextul si decide ce sa spuna."""
    instructiune_dificultate = NIVELURI_DIFICULTATE[dificultate]["instructiune"]
    detalii_necesare = DETALII_NECESARE.get(intentie, []) if intentie and intentie != "alta_solicitare" else []

    if faza == "identificare_intentie":
        instructiune_faza = (
            "Nu ai înțeles încă clar problema clientului. "
            "Cere-i politicos să explice mai detaliat motivul apelului. "
            "Maxim 1-2 propoziții."
        )
    elif faza == "colectare_detalii":
        instructiune_faza = (
            "Ai identificat intenția clientului: " + (intentie or "necunoscută") + ". "
            "Trebuie să colectezi următoarele detalii necesare: " + ", ".join(detalii_necesare) + ". "
            "Verifică ce detalii au fost deja menționate în conversație și cere DOAR ce lipsește. "
            "Cere un singur detaliu pe rând. Maxim 2 propoziții."
        )
    elif faza == "solutie":
        instructiune_faza = (
            "Ai toate informațiile necesare. Oferă acum soluția pentru problema clientului. "
            "INSTRUCȚIUNE IMPORTANTĂ: " + instructiune_dificultate + " "
            "Maxim 4-5 propoziții. Fii specific și concret."
        )
    elif faza == "confirmare":
        instructiune_faza = (
            "Ai oferit soluția. Răspunde la ce a spus sau întrebat clientul, "
            "empatic și profesionist. "
            "La finalul replicii tale, întreabă întotdeauna dacă mai poți ajuta "
            "cu altceva sau dacă mai are întrebări. "
            "NU încheia conversația tu — lasă clientul să decidă când se termină. "
            "Maxim 3 propoziții."
        )
    else:
        instructiune_faza = "Răspunde politicos și profesionist. Maxim 2 propoziții."

    prompt = (
        "Ești un robot de call-center din domeniul " + NUME_DOMENII.get(domeniu, domeniu) + ". "
        "Vorbești în română, politicos și natural, folosind diacritice corecte. "
        "Nu repeta ce ai spus deja și nu folosi aceeași formulă de politețe de două ori în aceeași propoziție "
        "(ex: evită 'vă rog să îmi spuneți, vă rog').\n\n"
        "INSTRUCȚIUNE CURENTĂ: " + instructiune_faza + "\n\n"
        "CONVERSAȚIE PÂNĂ ACUM:\n" + dialog + "\n\n"
        "REPLICĂ ROBOT:"
    )
    raspuns, _ = call_llm(prompt, max_tokens=150)
    return raspuns

def verifica_detalii_complete(dialog, intentie):
    """LLM verifica daca toate detaliile necesare au fost mentionate in conversatie."""
    detalii_necesare = DETALII_NECESARE.get(intentie, [])
    if not detalii_necesare:
        return True

    prompt = (
        "Analizează conversația și verifică dacă clientul a menționat următoarele informații:\n"
        + "\n".join(f"- {d}" for d in detalii_necesare) +
        "\n\nCONVERSATIE:\n" + dialog +
        "\n\nRăspunde DOAR cu: DA (toate informațiile sunt prezente) sau NU (lipsesc informații).\n"
        "RĂSPUNS:"
    )
    raspuns, _ = call_llm(prompt, max_tokens=5)
    return "da" in raspuns.lower()

# ─── ANALIZA FINALA ───────────────────────────────────────────────────────────

def adauga_punctuatie(dialog):
    prompt = (
        "Adauga punctuatie si majuscule corecte la urmatorul dialog transcris automat. "
        "Nu modifica cuvintele, doar adauga semne de punctuatie unde e necesar.\n\n"
        "DIALOG:\n" + dialog + "\n\nDIALOG CU PUNCTUATIE:"
    )
    raspuns, _ = call_llm(prompt, max_tokens=800)
    return raspuns.strip()


def analiza_finala(dialog_complet, domeniu):
    print("\n  [ANALIZA] Rulare analiza finala...")

    # Adauga punctuatie pentru analiza mai precisa
    dialog_pentru_analiza = adauga_punctuatie(dialog_complet)

    # ── SATISFACTIE ───────────────────────────────────────────────────────────
    EXEMPLE_LUNGI = {
        "pozitiv": (
            "OPERATOR: Buna ziua, cu ce va pot ajuta?\nCLIENT: Am o problema cu cardul, l-am pierdut.\nOPERATOR: L-am blocat imediat si va trimitem unul nou in 3 zile.\nCLIENT: Multumesc mult, sunteti foarte promti!",
            "pozitiv"
        ),
        "neutru": (
            "OPERATOR: Buna ziua.\nCLIENT: Vreau sa stiu statusul comenzii mele.\nOPERATOR: Comanda e in drum spre dumneavoastra, ajunge maine.\nCLIENT: Ok, am inteles. La revedere.",
            "neutru"
        ),
        "negativ": (
            "OPERATOR: Buna ziua.\nCLIENT: Am sunat a treia oara pentru aceeasi problema.\nOPERATOR: Imi pare rau, investigam.\nCLIENT: Bine, ce sa fac...",
            "negativ"
        ),
    }
    exemple_text = ""
    for clasa, (dialog_ex, satisfactie_ex) in EXEMPLE_LUNGI.items():
        exemple_text += "CONVERSATIE:\n" + dialog_ex + "\nSATISFACTIE: " + satisfactie_ex + "\n\n"

    prompt_s = (
        "Esti un expert in analiza satisfactiei clientilor in conversatii de call-center.\n\n"
        "ATENTIE: Conversatia este o transcriere automata fara punctuatie. "
        "Interpreteaza cu atentie frazele ambigue tinand cont de context.\n\n"
        "CONVERSATIE:\n" + dialog_pentru_analiza + "\n\n"
        "SARCINA: Pe baza conversatiei de mai sus, determina nivelul de satisfactie al clientului.\n\n"
        "DEFINITII:\n"
        "- pozitiv: problema rezolvata complet, clientul multumit si o exprima clar\n"
        "- neutru: problema rezolvata tehnic dar clientul nu prezinta nicio emotie,\n"
        "- negativ: clientul exprima suparare sau frustrare, uneori aceasta suparare nu este excesiva, dar ea tot exista\n"
        "  (ironie clara, resemnare, replici taioase, incheie brusc conversatia)\n\n"
        "REGULI:\n"
        "1. Un singur comentariu negativ urmat de acceptare NU inseamna automat negativ\n"
        "2. Uita-te la TONUL GENERAL si REZULTATUL FINAL al conversatiei\n"
        "3. Frustrarea implicita conteaza: bine inteleg, ce sa fac, remarci ironice\n"
        # "4. Daca clientul incheie politicos DAR fara entuziasm sau apreciere explicita, este neutru\n"
        # "5. Daca clientul exprima apreciere clara, entuziasm si vorbeste foarte frumos, este pozitiv\n"
        # "6. Daca clientul exprima frustrare clara in timpul conversatiei si incheie resemnat, este negativ\n"
        "7. Ignora complet dificultatea sau complexitatea problemei — conteaza doar emotia clientului\n\n"
        "EXEMPLE (acorda atentie diferentei dintre neutru si negativ):\n" + exemple_text +
        "SATISFACTIE IDENTIFICATA:"
    )
    raspuns_s, _ = call_llm(prompt_s, max_tokens=10)
    satisfactie = extrage_satisfactie(raspuns_s)

    # ── REZUMAT ───────────────────────────────────────────────────────────────
    tip_info = TIP_REZUMAT.get(satisfactie, TIP_REZUMAT["neutru"])
    exemple_rez = {
        "SCURT": (
            "CLIENT: Am pierdut cardul, il vreau blocat.\nOPERATOR: L-am blocat, va trimitem altul in 3 zile.",
            "Clientul a sunat pentru a bloca un card pierdut. Operatorul a blocat cardul si a initiat emiterea unuia nou."
        ),
        "MEDIU": (
            "CLIENT: Comanda mea nu a sosit.\nOPERATOR: A fost o intarziere la curier. Va ajunge maine.\nCLIENT: Ok.",
            "Clientul a reclamat o comanda nelivrata la termen. Operatorul a verificat situatia si a identificat o intarziere la curier. Comanda urmeaza sa fie livrata a doua zi. Clientul a acceptat solutia."
        ),
        "LUNG": (
            "CLIENT: Am sunat de trei ori pentru aceeasi problema cu factura.\nOPERATOR: Imi pare rau, investigam.\nCLIENT: Bine, astept.",
            "Clientul a contactat call-center-ul pentru a treia oara in legatura cu aceeasi problema de facturare nerezolvata. Clientul si-a exprimat nemultumirea fata de lipsa unei solutii. Operatorul a initiat o investigatie. Problema ramane deschisa."
        ),
    }
    tip = tip_info["tip"]
    dialog_ex, rezumat_ex = exemple_rez[tip]
    prompt_r = (
        "Esti un expert in sumarizarea conversatiilor telefonice din call-center.\n\n"
        "ATENTIE: Conversatia este o transcriere automata fara punctuatie. "
        "Interpreteaza cu atentie frazele ambigue tinand cont de context.\n\n"
        "CONVERSATIE:\n" + dialog_pentru_analiza + "\n\n"
        "SARCINA: Genereaza un rezumat de tip " + tip + ".\n\n"
        "CERINTE:\n"
        "- Lungime: " + str(tip_info["min_cuv"]) + "-" + str(tip_info["max_cuv"]) + " cuvinte (" + tip_info["propozitii"] + ")\n"
        "- Limba: romana\n"
        "- Mentioneaza problema principala si rezultatul final\n"
        "- Nu adauga informatii care nu apar in conversatie\n\n"
        "EXEMPLU:\nCONVERSATIE:\n" + dialog_ex + "\nREZUMAT " + tip + ":\n" + rezumat_ex + "\n\n"
        "REZUMAT " + tip + ":"
    )
    rezumat, _ = call_llm(prompt_r, max_tokens=200)

    return satisfactie, rezumat


# ─── CONVERSATIE PRINCIPALA ───────────────────────────────────────────────────

def ruleaza_conversatie(domeniu):
    conv_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    replici = []
    detalii_colectate = {}

    def adauga(rol, text):
        replici.append({"rol": rol, "text": text})

    def get_dialog():
        return "\n".join(f"{r['rol'].upper()}: {r['text']}" for r in replici)

    # Alege dificultatea la inceputul conversatiei
    dificultate = alege_dificultate()

    print(f"\n{'='*60}")
    print(f"Conversatie: {conv_id} | Domeniu: {domeniu} | Dificultate: {dificultate}")
    print(f"{'='*60}")

    # Salut + prima intrebare de identificare — predefinite, fara LLM
    salut = SALUT_DOMENII[domeniu]
    prima_intrebare = REPLICI_IDENTIFICARE[domeniu][0]
    intro = salut + " " + prima_intrebare
    vorbeste(intro)
    adauga("operator", intro)

    intentie = None
    faza = "identificare_client"
    tururi_fara_progres = 0
    identificare_colectata = {}

    while True:
        # Asculta clientul
        text_client = asculta()
        if not text_client:
            continue

        adauga("client", text_client)

        if client_vrea_sa_incheie(text_client) and faza != "identificare_intentie":
            break

        # ── FAZA 0: Identificare client — replici predefinite, fara LLM ─────
        if faza == "identificare_client":
            replici_id = REPLICI_IDENTIFICARE.get(domeniu, ["Vă rog să îmi spuneți numele complet.", "Cu ce vă pot ajuta?"])
            nr_replici_id = len(replici_id) - 1  # ultima e "Cu ce va pot ajuta"
            nr_raspunsuri = len(identificare_colectata)

            # Salveaza raspunsul clientului la ultima intrebare pusa
            if nr_raspunsuri < nr_replici_id:
                identificare_colectata[f"detaliu_{nr_raspunsuri}"] = text_client

            nr_raspunsuri = len(identificare_colectata)

            if nr_raspunsuri < nr_replici_id:
                # Mai sunt detalii de colectat — urmatoarea replica predefinita
                replica = replici_id[nr_raspunsuri]
                vorbeste(replica)
                adauga("operator", replica)
                continue
            else:
                # Toate detaliile colectate — replica finala predefinita ("Cu ce va pot ajuta")
                replica = replici_id[-1]
                vorbeste(replica)
                adauga("operator", replica)
                faza = "identificare_intentie"
                continue

        # ── FAZA 1: Identificare intentie ────────────────────────────────────
        if faza == "identificare_intentie":
            intentie = extrage_intentie_llm(get_dialog(), domeniu)
            print(f"  [INTENTIE] {intentie}")

            if intentie == "alta_solicitare":
                tururi_fara_progres += 1
                if tururi_fara_progres >= 2:
                    mesaj = (
                        REPLICA_OPERATOR
                    )
                    vorbeste(mesaj)
                    adauga("operator", mesaj)
                    return conv_id, replici, intentie, dificultate, None, None
                else:
                    replica = genereaza_replica_robot(get_dialog(), domeniu, intentie, dificultate, "identificare_intentie")
                    vorbeste(replica)
                    adauga("operator", replica)
                    continue
            else:
                tururi_fara_progres = 0
                # Verifica daca are nevoie de detalii
                detalii_necesare = DETALII_NECESARE.get(intentie, [])
                if detalii_necesare:
                    faza = "colectare_detalii"
                    replica = genereaza_replica_robot(get_dialog(), domeniu, intentie, dificultate, "colectare_detalii")
                    vorbeste(replica)
                    adauga("operator", replica)
                    continue
                else:
                    faza = "solutie"

        # ── FAZA 2: Colectare detalii ─────────────────────────────────────────
        if faza == "colectare_detalii":
            # LLM verifica daca toate detaliile au fost mentionate
            if verifica_detalii_complete(get_dialog(), intentie):
                faza = "solutie"
            else:
                replica = genereaza_replica_robot(get_dialog(), domeniu, intentie, dificultate, "colectare_detalii")
                vorbeste(replica)
                adauga("operator", replica)
                continue

        # ── FAZA 3: Solutie ───────────────────────────────────────────────────
        if faza == "solutie":
            # Verifica redirect operator
            if SOLUTII_PREDEFINITE.get(intentie) == "OPERATOR":
                mesaj = (
                    REPLICA_OPERATOR
                )
                vorbeste(mesaj)
                adauga("operator", mesaj)
                return conv_id, replici, intentie, dificultate, None, None

            # LLM genereaza solutia in context
            solutie = genereaza_replica_robot(get_dialog(), domeniu, intentie, dificultate, "solutie")
            vorbeste(solutie)
            adauga("operator", solutie)
            faza = "confirmare"
            continue

        # ── FAZA 4: Confirmare + incheiere ────────────────────────────────────
        if faza == "confirmare":
            if client_vrea_sa_incheie(text_client):
                break
            else:
                # Proceseaza ce a spus clientul si intreaba daca mai poate ajuta
                replica = genereaza_replica_robot(
                    get_dialog(), domeniu, intentie, dificultate, "confirmare"
                )
                vorbeste(replica)
                adauga("operator", replica)
                # Nu incheia niciodata din cod — asteapta clientul sa decida

    # Incheiere
    incheiere = REPLICA_INCHEIERE
    vorbeste(incheiere)
    adauga("operator", incheiere)

    # Analiza finala
    satisfactie, rezumat = analiza_finala(get_dialog(), domeniu)
    print(f"\n  [ANALIZA] Satisfactie: {satisfactie}")
    print(f"  [ANALIZA] Rezumat: {rezumat[:100]}...")

    return conv_id, replici, intentie, dificultate, satisfactie, rezumat

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Robot telefonic v2")
    parser.add_argument("--domeniu", choices=list(INTENTII_DOMENII.keys()),
                        help="Domeniu predefinit")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("ROBOT TELEFONIC CALL-CENTER v2")
    print("="*60)

    # Selectare domeniu
    if args.domeniu:
        domeniu = args.domeniu
    else:
        vorbeste(
            "Bună ziua! Ați sunat la serviciul de relații cu clienții. "
            "Vă rog să specificați domeniul: banking, medicină, retail, "
            "telecomunicații sau servicii publice."
        )
        raspuns = asculta()
        domeniu = None
        for d in INTENTII_DOMENII:
            if d in normalizeaza(raspuns) or NUME_DOMENII.get(d, "") in normalizeaza(raspuns):
                domeniu = d
                break
        if not domeniu:
            # Fallback LLM
            prompt_d = (
                "Din textul urmator, identifica domeniul: banking, medicina, retail, "
                "telecom sau servicii_publice. Returneaza DOAR unul din aceste cuvinte.\n"
                "Text: " + raspuns
            )
            raspuns_d, _ = call_llm(prompt_d, max_tokens=10)
            for d in INTENTII_DOMENII:
                if d in normalizeaza(raspuns_d):
                    domeniu = d
                    break
        if not domeniu:
            domeniu = "banking"

    print(f"Domeniu: {domeniu}\n")

    # Ruleaza conversatia
    conv_id, replici, intentie, dificultate, satisfactie, rezumat = ruleaza_conversatie(domeniu)

    # Salvare
    rezultat = {
        "id": conv_id,
        "domeniu": domeniu,
        "timestamp": datetime.now().isoformat(),
        "dificultate": dificultate,
        "conversatie": replici,
        "analiza": {
            "intentie": intentie,
            "satisfactie": satisfactie,
            "rezumat": rezumat,
        }
    }

    fisier = os.path.join(CONFIG["RESULTS_DIR"], f"conversatie_{conv_id}.json")
    with open(fisier, "w", encoding="utf-8") as f:
        json.dump(rezultat, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMAR FINAL")
    print(f"{'='*60}")
    print(f"  Intentie:    {intentie}")
    print(f"  Dificultate: {dificultate}")
    print(f"  Satisfactie: {satisfactie}")
    print(f"  Salvat in:   {fisier}")

if __name__ == "__main__":
    main()