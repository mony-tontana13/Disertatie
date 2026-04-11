"""
Utilitare comune pentru toate versiunile de prompt - satisfactie modele API.
GPT-4.1-mini, Gemini-2.5-flash, command-r7b-12-2024
"""
import json
import os
import random
import time
import unicodedata
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

CLASE = ["pozitiv", "neutru", "negativ"]

EXEMPLE_SCURTE = {
    "pozitiv": (
        "CLIENT: Buna ziua, am o problema cu cardul meu blocat.\n"
        "OPERATOR: Va deblochez imediat. Gata, cardul este activ.\n"
        "CLIENT: Minunat, multumesc mult, chiar ma ajutati!",
        "pozitiv"
    ),
    "neutru": (
        "CLIENT: Buna ziua, vreau sa stiu statusul comenzii mele.\n"
        "OPERATOR: Comanda va ajunge maine intre 10 si 14.\n"
        "CLIENT: Ok, am inteles. Pa.",
        "neutru"
    ),
    "negativ": (
        "CLIENT: Sunt la a treia tentativa sa rezolv aceasta problema.\n"
        "OPERATOR: Inteleg, va transferam la un specialist.\n"
        "CLIENT: Bine, ce sa fac... transferati.",
        "negativ"
    ),
}

EXEMPLE_LUNGI = {
    "pozitiv": (
        "CLIENT: Buna ziua, am observat o taxa dubla pe factura mea.\n"
        "OPERATOR: Va verific factura. Da, aveti dreptate, a fost o eroare. Va restitui suma in 24 de ore.\n"
        "CLIENT: Perfect, exact asta aveam nevoie. Multumesc foarte mult, sunteti promti!\n"
        "OPERATOR: Cu placere, o zi buna!\n"
        "CLIENT: Si dumneavoastra, la revedere!",
        "pozitiv"
    ),
    "neutru": (
        "CLIENT: Buna ziua, comanda mea nu a ajuns la timp.\n"
        "OPERATOR: Imi pare rau, am verificat si va fi livrata maine.\n"
        "CLIENT: Bine, maine merge si asa.\n"
        "OPERATOR: Va multumim pentru intelegere.\n"
        "CLIENT: Da, la revedere.",
        "neutru"
    ),
    "negativ": (
        "CLIENT: Buna ziua, am sunat a doua oara pentru aceeasi problema cu internetul.\n"
        "OPERATOR: Imi pare rau, va pot programa un tehnician pentru joi.\n"
        "CLIENT: Joi... bine, ce sa fac, nu am de ales.\n"
        "OPERATOR: Va multumim pentru rabdare.\n"
        "CLIENT: Da... la revedere.",
        "negativ"
    ),
}


def normalizeaza(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().strip()


def extrage_satisfactie(raspuns):
    raspuns_norm = normalizeaza(raspuns)
    for clasa in CLASE:
        if clasa in raspuns_norm:
            return clasa
    return "necunoscut"


def selecteaza_subset(folder, n_per_domeniu=2):
    random.seed(42)
    subset = []
    domenii = ["banking", "medicina", "retail", "telecom", "servicii_publice"]
    for domeniu in domenii:
        domeniu_path = os.path.join(folder, domeniu)
        if not os.path.isdir(domeniu_path):
            print(f"  ATENTIE: folder lipsa pentru {domeniu}")
            continue
        fisiere = sorted([f for f in os.listdir(domeniu_path) if f.endswith(".json")])
        simple, complexe = [], []
        for fisier in fisiere:
            with open(os.path.join(domeniu_path, fisier), encoding="utf-8") as f:
                conv = json.load(f)
            if conv.get("complexitate") == "simpla":
                simple.append(conv)
            else:
                complexe.append(conv)
        selectate = []
        if simple:
            selectate += random.sample(simple, min(n_per_domeniu // 2 + n_per_domeniu % 2, len(simple)))
        if complexe:
            selectate += random.sample(complexe, min(n_per_domeniu // 2, len(complexe)))
        subset.extend(selectate[:n_per_domeniu])
        print(f"  {domeniu}: {len(selectate[:n_per_domeniu])} conversatii")
    return subset


def incarca_toate_conversatiile(folder):
    conversatii = []
    domenii = ["banking", "medicina", "retail", "telecom", "servicii_publice"]
    for domeniu in domenii:
        domeniu_path = os.path.join(folder, domeniu)
        if not os.path.isdir(domeniu_path):
            continue
        fisiere = sorted([f for f in os.listdir(domeniu_path) if f.endswith(".json")])
        for fisier in fisiere:
            with open(os.path.join(domeniu_path, fisier), encoding="utf-8") as f:
                conversatii.append(json.load(f))
        print(f"  {domeniu}: {len(fisiere)} conversatii")
    return conversatii


def call_gpt(prompt):
    if not OPENAI_API_KEY:
        return "necunoscut", 0.0, 0.0
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    start_total = time.time()
    ttft = None
    raspuns_complet = ""
    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        stream=True
    )
    for chunk in stream:
        if ttft is None and chunk.choices[0].delta.content:
            ttft = time.time() - start_total
        if chunk.choices[0].delta.content:
            raspuns_complet += chunk.choices[0].delta.content
    latenta_totala = time.time() - start_total
    return raspuns_complet.strip(), round(ttft or latenta_totala, 3), round(latenta_totala, 3)


def call_gemini(prompt):
    if not GEMINI_API_KEY:
        return "necunoscut", 0.0, 0.0
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    start_total = time.time()
    ttft = None
    raspuns_complet = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=20)
    ):
        if ttft is None and chunk.text:
            ttft = time.time() - start_total
        if chunk.text:
            raspuns_complet += chunk.text
    latenta_totala = time.time() - start_total
    return raspuns_complet.strip(), round(ttft or latenta_totala, 3), round(latenta_totala, 3)


def call_cohere(prompt):
    if not COHERE_API_KEY:
        return "necunoscut", 0.0, 0.0
    import cohere
    co = cohere.ClientV2(api_key=COHERE_API_KEY)
    start_total = time.time()
    response = co.chat(
        model="command-r7b-12-2024",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20
    )
    ttft = time.time() - start_total
    raspuns_complet = response.message.content[0].text.strip()
    return raspuns_complet, round(ttft, 3), round(ttft, 3)


MODELE_API = {
    "GPT-4.1-mini": call_gpt,
    "Gemini-2.5-flash": call_gemini,
    "command-r7b-12-2024": call_cohere
}


def ruleaza_evaluare_api(conversatii, get_prompt_fn, versiune, results_dir):
    disponibile = []
    if OPENAI_API_KEY: disponibile.append("GPT-4.1-mini")
    if GEMINI_API_KEY: disponibile.append("Gemini-2.5-flash")
    if COHERE_API_KEY: disponibile.append("command-r7b-12-2024")

    if not disponibile:
        print("Nicio cheie API setata.")
        return {}

    print(f"Modele disponibile: {', '.join(disponibile)}")
    toate_rezultatele = {}

    for nume_model in disponibile:
        func_model = MODELE_API[nume_model]
        print(f"\n--- {nume_model} | {versiune} ---")
        rezultate = []

        for i, conv in enumerate(conversatii):
            conv_id = conv["id"]
            domeniu = conv.get("domeniu", "banking")
            complexitate = conv.get("complexitate", "?")
            satisfactie_gold = conv.get("satisfactie", "necunoscut")
            dialog = "\n".join([r["rol"].upper() + ": " + r["text"] for r in conv["conversatie"]])

            prompt = get_prompt_fn(dialog)
            raspuns_brut, ttft, latenta_totala = func_model(prompt)
            satisfactie_pred = extrage_satisfactie(raspuns_brut)
            corecta = satisfactie_pred == satisfactie_gold
            status = "OK" if corecta else "GRESIT"

            print(f"  [{i+1:03d}] {conv_id} [{complexitate}] | gold: {satisfactie_gold} | pred: {satisfactie_pred} | {status} | TTFT: {ttft:.2f}s")
            if not corecta:
                print(f"    brut: {raspuns_brut[:80]}")

            rezultate.append({
                "id": conv_id, "domeniu": domeniu, "complexitate": complexitate,
                "satisfactie_gold": satisfactie_gold, "satisfactie_pred": satisfactie_pred,
                "corecta": corecta, "raspuns_brut": raspuns_brut,
                "ttft": ttft, "latenta_totala": latenta_totala,
                "versiune_prompt": versiune, "model": nume_model
            })

        toate_rezultatele[nume_model] = rezultate

    return toate_rezultatele


def calculeaza_si_afiseaza_api(toate_rezultatele, versiune):
    toate_metrici = []

    for nume_model, rezultate in toate_rezultatele.items():
        gold = [r["satisfactie_gold"] for r in rezultate]
        pred = [r["satisfactie_pred"] for r in rezultate]
        acc = accuracy_score(gold, pred)
        f1 = f1_score(gold, pred, average="macro", zero_division=0, labels=CLASE)
        ttft_medie = sum(r["ttft"] for r in rezultate) / len(rezultate)
        latenta_medie = sum(r["latenta_totala"] for r in rezultate) / len(rezultate)

        linii = []
        linii.append(f"\n=== {nume_model} | Prompt {versiune} | {len(rezultate)} conversatii ===")
        linii.append(f"  Accuracy:         {acc:.2%}")
        linii.append(f"  F1 Macro:         {f1:.3f}")
        linii.append(f"  TTFT medie:       {ttft_medie:.3f}s")
        linii.append(f"  Latenta medie:    {latenta_medie:.3f}s")
        linii.append(f"  Distributie gold: {dict(Counter(gold))}")
        linii.append(f"  Distributie pred: {dict(Counter(pred))}")

        erori = [r for r in rezultate if not r["corecta"]]
        if erori:
            linii.append(f"  Erori ({len(erori)}/{len(rezultate)}):")
            for e in erori:
                linii.append(f"    {e['id']} [{e['complexitate']}]: gold={e['satisfactie_gold']} pred={e['satisfactie_pred']}")

        linii.append(f"  Per clasa:")
        per_clasa = {}
        for clasa in CLASE:
            rez_c = [r for r in rezultate if r["satisfactie_gold"] == clasa]
            if rez_c:
                acc_c = sum(1 for r in rez_c if r["corecta"]) / len(rez_c)
                per_clasa[clasa] = round(acc_c, 4)
                linii.append(f"    {clasa:<22}: {acc_c:.0%} ({sum(1 for r in rez_c if r['corecta'])}/{len(rez_c)})")

        raport_text = "\n".join(linii)
        print(raport_text)

        toate_metrici.append({
            "model": nume_model, "versiune": versiune,
            "accuracy": round(acc, 4), "f1": round(f1, 4),
            "ttft_medie": round(ttft_medie, 3),
            "latenta_medie": round(latenta_medie, 3),
            "nr_conversatii": len(rezultate),
            "per_clasa": per_clasa,
            "raport_text": raport_text
        })

    return toate_metrici
