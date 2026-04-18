"""
Utilitare comune pentru toate versiunile de prompt - intentie modele API.
GPT-4.1-mini, Gemini-2.5-flash, Aya Expanse (Cohere)
"""
import json
import os
import random
import time
import unicodedata
from sklearn.metrics import accuracy_score, f1_score

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

INTENTII_DOMENII = {
    "banking": ["problema_credit","tranzactie_gresita","card_blocat","tranzactie_suspecta","problema_transfer","problema_schimb_valutar","problema_sold","card_pierdut"],
    "medicina": ["rezultate_analize","problema_reteta","problema_asigurare","reclamatie_personal","consultatie_anulata","problema_facturare","problema_programare","anulare_programare"],
    "retail": ["produs_lipsa_stoc","comanda_gresita","problema_livrare","problema_garantie","reclamatie_produs","anulare_comanda","comanda_intarziata","retur_produs"],
    "telecom": ["problema_modificare_abonament","portare_esuata","problema_internet","problema_roaming","factura_gresita","reziliere_contract","activare_esuata","problema_semnal"],
    "servicii_publice": ["dosar_respins","contestatie_decizie","informatii_program","reclamatie_serviciu","sesizare_problema","problema_plata_taxa","acte_incomplete","programare_ghiseu"]
}

EXEMPLE_FEWSHOT_LUNGI = {
    "banking": [
        ("OPERATOR: Buna ziua, cu ce va pot ajuta?\nCLIENT: Buna ziua, am o problema cu cardul meu, l-am pierdut ieri si nu stiu ce sa fac.\nOPERATOR: Inteleg, va ajut imediat.", "card_pierdut"),
        ("OPERATOR: Serviciul clienti, va ascult.\nCLIENT: Buna ziua, as vrea sa stiu de ce rata mea la credit a crescut luna aceasta.\nOPERATOR: Va verific contul.", "problema_credit"),
    ],
    "medicina": [
        ("OPERATOR: Clinica MedCare, buna ziua.\nCLIENT: Buna ziua, am facut analize saptamana trecuta si vreau sa aflu rezultatele.\nOPERATOR: Va caut in sistem.", "rezultate_analize"),
        ("OPERATOR: Cu ce va pot ajuta?\nCLIENT: Am o programare maine dar nu mai pot veni, as vrea sa o anulez.\nOPERATOR: Sigur, cum va numiti?", "anulare_programare"),
    ],
    "retail": [
        ("OPERATOR: Serviciul clienti, buna ziua.\nCLIENT: Buna ziua, am primit o comanda dar produsele nu sunt cele pe care le-am comandat.\nOPERATOR: Imi pare rau, va ajut.", "comanda_gresita"),
        ("OPERATOR: Cu ce va pot ajuta?\nCLIENT: Pachetul meu nu a ajuns si a trecut termenul de livrare de trei zile.\nOPERATOR: Va verific comanda.", "problema_livrare"),
    ],
    "telecom": [
        ("OPERATOR: Buna ziua, serviciul clienti.\nCLIENT: Buna ziua, am vrut sa imi port numarul la voi dar cererea a fost respinsa.\nOPERATOR: Va verific situatia.", "portare_esuata"),
        ("OPERATOR: Cu ce va pot ajuta?\nCLIENT: As vrea sa schimb abonamentul meu dar nu reusesc sa fac asta din aplicatie.\nOPERATOR: Va ajut eu.", "problema_modificare_abonament"),
    ],
    "servicii_publice": [
        ("OPERATOR: Primaria, buna ziua.\nCLIENT: Buna ziua, dosarul meu a fost respins si nu inteleg de ce.\nOPERATOR: Va caut dosarul.", "dosar_respins"),
        ("OPERATOR: Cu ce va pot ajuta?\nCLIENT: As vrea sa fac o programare la ghiseu pentru un act de identitate.\nOPERATOR: Va programez.", "programare_ghiseu"),
    ],
}


def normalizeaza(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().strip()


def extrage_intentie(raspuns, domeniu):
    intentii_valide = INTENTII_DOMENII.get(domeniu, [])
    raspuns_norm = normalizeaza(raspuns)
    intentii_norm = {normalizeaza(i): i for i in intentii_valide}
    for intentie_norm, intentie_orig in intentii_norm.items():
        if intentie_norm in raspuns_norm:
            return intentie_orig
    return "alta_solicitare"


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


# ============================================================
# APELURI API CU TTFT
# ============================================================

def call_gpt(prompt):
    if not OPENAI_API_KEY:
        return "alta_solicitare", 0.0, 0.0
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    start_total = time.time()
    ttft = None
    raspuns_complet = ""
    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
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
        return "alta_solicitare", 0.0, 0.0
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    start_total = time.time()
    ttft = None
    raspuns_complet = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt
    ):
        if ttft is None and chunk.text:
            ttft = time.time() - start_total
        if chunk.text:
            raspuns_complet += chunk.text
    latenta_totala = time.time() - start_total
    return raspuns_complet.strip(), round(ttft or latenta_totala, 3), round(latenta_totala, 3)


def call_aya(prompt):
    if not COHERE_API_KEY:
        return "alta_solicitare", 0.0, 0.0
    import cohere
    co = cohere.ClientV2(api_key=COHERE_API_KEY)
    start_total = time.time()
    response = co.chat(
        model="command-r7b-12-2024",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    ttft = time.time() - start_total
    raspuns_complet = response.message.content[0].text.strip()
    return raspuns_complet, round(ttft, 3), round(ttft, 3)


MODELE_API = {
    "Gemini-2.5-flash": call_gemini,
    "GPT-4.1-mini": call_gpt,
    "command-r7b-12-2024": call_aya
}


def ruleaza_evaluare_api(conversatii, get_prompt_fn, versiune, results_dir):
    """Ruleaza evaluarea pe toate modelele API disponibile."""
    disponibile = []
    if GEMINI_API_KEY: disponibile.append("Gemini-2.5-flash")
    if OPENAI_API_KEY: disponibile.append("GPT-4.1-mini")
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
            dialog = "\n".join([r["rol"].upper() + ": " + r["text"] for r in conv["conversatie"]])

            intentie_gold = conv.get("intentie_gold", ["alta_solicitare"])
            if isinstance(intentie_gold, list):
                intentie_gold = intentie_gold[0]

            prompt = get_prompt_fn(dialog, domeniu)
            raspuns_brut, ttft, latenta_totala = func_model(prompt)
            intentie_pred = extrage_intentie(raspuns_brut, domeniu)
            corecta = intentie_pred == intentie_gold
            status = "OK" if corecta else "GRESIT"

            print(f"  [{i+1:03d}] {conv_id} [{complexitate}] | gold: {intentie_gold} | pred: {intentie_pred} | {status} | TTFT: {ttft:.2f}s | Total: {latenta_totala:.2f}s")
            if not corecta:
                print(f"    brut: {raspuns_brut[:80]}")

            rezultate.append({
                "id": conv_id, "domeniu": domeniu, "complexitate": complexitate,
                "intentie_gold": intentie_gold, "intentie_pred": intentie_pred,
                "corecta": corecta, "raspuns_brut": raspuns_brut,
                "ttft": ttft, "latenta_totala": latenta_totala,
                "versiune_prompt": versiune, "model": nume_model
            })

        toate_rezultatele[nume_model] = rezultate

    return toate_rezultatele


def calculeaza_si_afiseaza_api(toate_rezultatele, versiune):
    """Calculeaza si afiseaza metricile pentru toate modelele."""
    toate_metrici = []

    for nume_model, rezultate in toate_rezultatele.items():
        gold = [r["intentie_gold"] for r in rezultate]
        pred = [r["intentie_pred"] for r in rezultate]
        acc = accuracy_score(gold, pred)
        f1 = f1_score(gold, pred, average="macro", zero_division=0)
        ttft_medie = sum(r["ttft"] for r in rezultate) / len(rezultate)
        latenta_medie = sum(r["latenta_totala"] for r in rezultate) / len(rezultate)

        linii = []
        linii.append(f"\n=== {nume_model} | Prompt {versiune} | {len(rezultate)} conversatii ===")
        linii.append(f"  Accuracy:      {acc:.2%}")
        linii.append(f"  F1 Macro:      {f1:.3f}")
        linii.append(f"  TTFT medie:    {ttft_medie:.3f}s")
        linii.append(f"  Latenta medie: {latenta_medie:.3f}s")

        erori = [r for r in rezultate if not r["corecta"]]
        if erori:
            linii.append(f"  Erori ({len(erori)}/{len(rezultate)}):")
            for e in erori:
                linii.append(f"    {e['id']} [{e['complexitate']}]: gold={e['intentie_gold']} pred={e['intentie_pred']}")

        per_domeniu = {}
        linii.append(f"  Per domeniu:")
        for domeniu in ["banking", "medicina", "retail", "telecom", "servicii_publice"]:
            rez_d = [r for r in rezultate if r["domeniu"] == domeniu]
            if rez_d:
                acc_d = sum(1 for r in rez_d if r["corecta"]) / len(rez_d)
                per_domeniu[domeniu] = round(acc_d, 4)
                linii.append(f"    {domeniu:<22}: {acc_d:.0%} ({sum(1 for r in rez_d if r['corecta'])}/{len(rez_d)})")

        raport_text = "\n".join(linii)
        print(raport_text)

        toate_metrici.append({
            "model": nume_model, "versiune": versiune,
            "accuracy": round(acc, 4), "f1": round(f1, 4),
            "ttft_medie": round(ttft_medie, 3),
            "latenta_medie": round(latenta_medie, 3),
            "nr_conversatii": len(rezultate),
            "per_domeniu": per_domeniu,
            "raport_text": raport_text
        })

    return toate_metrici
