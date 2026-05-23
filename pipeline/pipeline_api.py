"""
Pipeline complet - Modele API (GPT + Gemini + command-r7b)
Cu checkpoint saving — continua de unde a ramas daca se blocheaza.

Utilizare:
    python3 pipeline_api.py
    python3 pipeline_api.py --tot_setul
    python3 pipeline_api.py --model gemini  # ruleaza doar Gemini
"""
import json
import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_engineering_API.intentii.utils_intentie_api import INTENTII_DOMENII, extrage_intentie
from prompt_engineering_API.satisfactie.utils_satisfactie_api import extrage_satisfactie, EXEMPLE_LUNGI
from prompt_engineering_API.rezumat.utils_rezumat_api import get_tip_rezumat, calculeaza_rouge, calculeaza_bertscore
from sklearn.metrics import f1_score

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

RESULTS_DIR = "./rezultate_evaluare/pipeline_api"
CHECKPOINT_DIR = "./rezultate_evaluare/pipeline_api/checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PROMPTURI_CASTIGATOARE = {
    "GPT-4.1-mini":        {"intentie": "V2_v1", "satisfactie": "V3_v2", "rezumat": "V3_v1"},
    "Gemini-2.5-flash":    {"intentie": "V2_v1", "satisfactie": "V1_v1", "rezumat": "V3_v1"},
    "command-r7b-12-2024": {"intentie": "V4_v1", "satisfactie": "V2_v1", "rezumat": "V3_v1"},
}

MODELE_DISPONIBILE = {
    "gpt":    "GPT-4.1-mini",
    "gemini": "Gemini-2.5-flash",
    "cohere": "command-r7b-12-2024",
}


# ─── CHECKPOINT ───────────────────────────────────────────────────────────────

def checkpoint_path(nume_model, descriere_set):
    nume_safe = nume_model.replace("-", "_").replace(".", "_")
    return os.path.join(CHECKPOINT_DIR, f"checkpoint_{nume_safe}_{descriere_set}.json")

def incarca_checkpoint(nume_model, descriere_set, ids_valide):
    """Incarca checkpoint si pastreaza DOAR rezultatele cu ID-uri din setul curent."""
    path = checkpoint_path(nume_model, descriere_set)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Pastreaza doar rezultatele care apartin setului curent
        data_filtrata = [r for r in data if r["id"] in ids_valide]
        ids_procesate = {r["id"] for r in data_filtrata}
        if len(data) != len(data_filtrata):
            print(f"  Checkpoint filtrat: {len(data)} -> {len(data_filtrata)} (eliminat rulari din alte seturi)")
        if ids_procesate:
            print(f"  Checkpoint gasit: {len(ids_procesate)}/{len(ids_valide)} conversatii deja procesate.")
        return data_filtrata, ids_procesate
    return [], set()

def salveaza_checkpoint(rezultate, nume_model, descriere_set):
    path = checkpoint_path(nume_model, descriere_set)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rezultate, f, ensure_ascii=False, indent=2)

def sterge_checkpoint(nume_model, descriere_set):
    path = checkpoint_path(nume_model, descriere_set)
    if os.path.exists(path):
        os.remove(path)


# ─── PROMPTURI INTENTIE ───────────────────────────────────────────────────────

def prompt_intentie_v2_1(dialog, domeniu):
    intentii = INTENTII_DOMENII.get(domeniu, [])
    return (
        "Lucrezi ca analist de date intr-un call-center din domeniul " + domeniu + ". "
        "Sarcina ta zilnica este sa identifici motivul pentru care clientii suna.\n\n"
        "REGULI:\n"
        "- Include DOAR ce a cerut sau intrebat clientul, nu actiunile operatorului\n"
        "- Alege DOAR din lista de intentii de mai jos\n\n"
        "INTENTII DISPONIBILE: " + ", ".join(intentii) + "\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "De ce a sunat clientul? Raspunde cu una sau doua intentii din lista:"
    )

def prompt_intentie_v4_1(dialog, domeniu):
    intentii = INTENTII_DOMENII.get(domeniu, [])
    exemple_lungi = {
        "banking": [
            ("OPERATOR: Buna ziua, cu ce va pot ajuta?\nCLIENT: L-am pierdut cardul ieri.\nOPERATOR: Inteleg.", "card_pierdut"),
            ("OPERATOR: Va ascult.\nCLIENT: De ce a crescut rata la credit?\nOPERATOR: Verific.", "problema_credit"),
        ],
        "medicina": [
            ("OPERATOR: Buna ziua.\nCLIENT: Vreau rezultatele analizelor de saptamana trecuta.\nOPERATOR: Va caut.", "rezultate_analize"),
            ("OPERATOR: Cu ce va ajut?\nCLIENT: Vreau sa anulez programarea de maine.\nOPERATOR: Sigur.", "anulare_programare"),
        ],
        "retail": [
            ("OPERATOR: Buna ziua.\nCLIENT: Am primit produse gresite in comanda.\nOPERATOR: Imi pare rau.", "comanda_gresita"),
            ("OPERATOR: Cu ce va ajut?\nCLIENT: Pachetul nu a ajuns desi a trecut termenul.\nOPERATOR: Verific.", "problema_livrare"),
        ],
        "telecom": [
            ("OPERATOR: Buna ziua.\nCLIENT: Portarea numărului a fost respinsa.\nOPERATOR: Verific.", "portare_esuata"),
            ("OPERATOR: Cu ce va ajut?\nCLIENT: Nu pot schimba abonamentul din aplicatie.\nOPERATOR: Va ajut.", "problema_modificare_abonament"),
        ],
        "servicii_publice": [
            ("OPERATOR: Primaria.\nCLIENT: Dosarul meu a fost respins.\nOPERATOR: Caut dosarul.", "dosar_respins"),
            ("OPERATOR: Cu ce va ajut?\nCLIENT: Vreau o programare la ghiseu pentru buletin.\nOPERATOR: Va programez.", "programare_ghiseu"),
        ],
    }
    exemple = exemple_lungi.get(domeniu, [])
    exemple_text = "".join(
        "CONVERSATIE:\n" + d + "\nINTENTIE IDENTIFICATA: " + i + "\n\n"
        for d, i in exemple
    )
    return (
        "Esti un expert in clasificarea intentiilor pentru call-center-uri din domeniul " + domeniu + ".\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "SARCINA: Identifica intentia clientului.\n\n"
        "REGULI:\n"
        "1. Include DOAR ce a cerut clientul\n"
        "2. Alege EXCLUSIV din lista furnizata\n"
        "3. Returneaza MAXIM doua intentii, separate prin virgula\n"
        "4. Prima intentie = cea principala\n\n"
        "INTENTII DISPONIBILE: " + ", ".join(intentii) + "\n\n"
        "EXEMPLE:\n" + exemple_text +
        "INTENTIE IDENTIFICATA:"
    )


# ─── PROMPTURI SATISFACTIE ────────────────────────────────────────────────────

def prompt_satisfactie_v1_1(dialog):
    return (
        "Analizeaza urmatoarea conversatie telefonica si determina nivelul de satisfactie al clientului.\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "Care este satisfactia clientului la finalul conversatiei? "
        "Raspunde cu un singur cuvant: pozitiv, neutru sau negativ:"
    )

def prompt_satisfactie_v2_1(dialog):
    return (
        "Esti un expert in analiza satisfactiei clientilor in conversatii de call-center.\n"
        "Determina nivelul de satisfactie al clientului la finalul conversatiei de mai jos.\n\n"
        "DEFINITII:\n"
        "- pozitiv: problema rezolvata complet, clientul multumit si o exprima clar\n"
        "- neutru: problema rezolvata tehnic dar clientul nu prezinta nicio emotie\n"
        "- negativ: clientul pleaca frustrat sau nemultumit, chiar daca implicit\n\n"
        "REGULI:\n"
        "- Un singur comentariu negativ urmat de acceptare NU inseamna automat negativ\n"
        "- Uita-te la TONUL GENERAL si REZULTATUL FINAL al conversatiei\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "Raspunde DOAR cu unul dintre cuvintele: pozitiv, neutru, negativ:"
    )

def prompt_satisfactie_v3_2(dialog):
    exemple_text = ""
    for clasa, (dialog_ex, satisfactie_ex) in EXEMPLE_LUNGI.items():
        exemple_text += "CONVERSATIE:\n" + dialog_ex + "\nSATISFACTIE: " + satisfactie_ex + "\n\n"
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


# ─── PROMPTURI REZUMAT ────────────────────────────────────────────────────────

def prompt_rezumat_v3_1(dialog, satisfactie):
    tip_info = get_tip_rezumat(satisfactie)
    tip = tip_info["tip"]
    exemple = {
        "SCURT": (
            "CLIENT: Am pierdut cardul, il vreau blocat.\nOPERATOR: L-am blocat, va trimitem altul in 3 zile.",
            "Clientul a sunat pentru a bloca un card pierdut. Operatorul a blocat cardul si a initiat emiterea unuia nou."
        ),
        "MEDIU": (
            "CLIENT: Comanda mea nu a sosit.\nOPERATOR: A fost o intarziere la curier. Va ajunge maine.\nCLIENT: Ok.",
            "Clientul a reclamat o comanda nelivrata la termen. Operatorul a verificat si a identificat o intarziere la curier. Comanda urmeaza sa fie livrata a doua zi. Clientul a acceptat solutia."
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


# ─── API CALLS ────────────────────────────────────────────────────────────────

def call_gpt(prompt, max_tokens=20):
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    start = time.time()
    raspuns = ""
    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            raspuns += chunk.choices[0].delta.content
    return raspuns.strip(), round(time.time() - start, 3)

def call_gemini(prompt, max_tokens=None, retry=3, wait=30):
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    for attempt in range(retry):
        try:
            start = time.time()
            raspuns = ""
            for chunk in client.models.generate_content_stream(
                model="gemini-2.5-flash", contents=prompt
            ):
                if chunk.text:
                    raspuns += chunk.text
            return raspuns.strip(), round(time.time() - start, 3)
        except Exception as e:
            print(f"\n    [Gemini eroare attempt {attempt+1}/{retry}: {str(e)[:60]}]")
            if attempt < retry - 1:
                print(f"    Astept {wait}s...")
                time.sleep(wait)
            else:
                raise

def call_cohere(prompt, max_tokens=20):
    import cohere
    co = cohere.ClientV2(api_key=COHERE_API_KEY)
    start = time.time()
    response = co.chat(
        model="command-r7b-12-2024",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.message.content[0].text.strip(), round(time.time() - start, 3)


# ─── DATE ─────────────────────────────────────────────────────────────────────

def selecteaza_subset(folder, n_per_domeniu=2):
    random.seed(42)
    subset = []
    domenii = ["banking", "medicina", "retail", "telecom", "servicii_publice"]
    for domeniu in domenii:
        domeniu_path = os.path.join(folder, domeniu)
        if not os.path.isdir(domeniu_path):
            continue
        fisiere = sorted([f for f in os.listdir(domeniu_path) if f.endswith(".json")])
        simple_conv, complexe_conv = [], []
        for fisier in fisiere:
            with open(os.path.join(domeniu_path, fisier), encoding="utf-8") as f:
                conv = json.load(f)
            if conv.get("complexitate") == "simpla":
                simple_conv.append(conv)
            else:
                complexe_conv.append(conv)
        selectate = []
        if simple_conv:
            selectate += random.sample(simple_conv, min(n_per_domeniu // 2 + n_per_domeniu % 2, len(simple_conv)))
        if complexe_conv:
            selectate += random.sample(complexe_conv, min(n_per_domeniu // 2, len(complexe_conv)))
        subset.extend(selectate[:n_per_domeniu])
    return subset

def incarca_toate(folder):
    conversatii = []
    for domeniu in ["banking", "medicina", "retail", "telecom", "servicii_publice"]:
        p = os.path.join(folder, domeniu)
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(".json"):
                    with open(os.path.join(p, f), encoding="utf-8") as fh:
                        conversatii.append(json.load(fh))
    return conversatii


# ─── PROCESARE CU CHECKPOINT ──────────────────────────────────────────────────

def proceseaza_model(nume_model, fn_intentie, fn_satisfactie, fn_rezumat,
                     fn_api_scurt, fn_api_lung, conversatii, descriere_set):

    # Incarca checkpoint daca exista — filtrat pe ID-urile din setul curent
    ids_valide = {c["id"] for c in conversatii}
    rezultate, ids_procesate = incarca_checkpoint(nume_model, descriere_set, ids_valide)
    ramase = [c for c in conversatii if c["id"] not in ids_procesate]
    total = len(conversatii)

    if ids_procesate:
        print(f"  Reluare de la conversatia {len(ids_procesate)+1}/{total}")
    else:
        print(f"  Start de la inceput: {total} conversatii")

    for i, conv in enumerate(ramase):
        conv_id = conv["id"]
        domeniu = conv.get("domeniu", "banking")
        complexitate = conv.get("complexitate", "?")
        dialog = "\n".join([r["rol"].upper() + ": " + r["text"] for r in conv["conversatie"]])
        intentie_gold = conv.get("intentie_gold", ["alta_solicitare"])
        if isinstance(intentie_gold, list):
            intentie_gold = intentie_gold[0]
        satisfactie_gold = conv.get("satisfactie", "neutru")
        rezumat_gold = conv.get("rezumat", "")

        nr_curent = len(ids_procesate) + i + 1
        print(f"  [{nr_curent:03d}/{total}] {conv_id} [{domeniu}]", end=" ", flush=True)

        try:
            # 1. Intentie
            raspuns_i, lat_i = fn_api_scurt(fn_intentie(dialog, domeniu), 30)
            intentie_pred = extrage_intentie(raspuns_i, domeniu)

            # 2. Satisfactie
            raspuns_s, lat_s = fn_api_scurt(fn_satisfactie(dialog), 20)
            satisfactie_pred = extrage_satisfactie(raspuns_s)

            # 3. Rezumat
            tip_info = get_tip_rezumat(satisfactie_gold)
            raspuns_r, lat_r = fn_api_lung(fn_rezumat(dialog, satisfactie_gold))
            nr_cuv = len(raspuns_r.split())
            in_limite = tip_info["min_cuv"] <= nr_cuv <= tip_info["max_cuv"]

            lat_tot = lat_i + lat_s + lat_r
            print(f"I={'OK' if intentie_pred==intentie_gold else 'X'} "
                  f"S={'OK' if satisfactie_pred==satisfactie_gold else 'X'} "
                  f"R={nr_cuv}cuv {lat_tot:.1f}s")

            rezultate.append({
                "id": conv_id, "domeniu": domeniu, "complexitate": complexitate,
                "intentie_gold": intentie_gold, "intentie_pred": intentie_pred,
                "intentie_corecta": intentie_pred == intentie_gold,
                "satisfactie_gold": satisfactie_gold, "satisfactie_pred": satisfactie_pred,
                "satisfactie_corecta": satisfactie_pred == satisfactie_gold,
                "rezumat_gold": rezumat_gold, "rezumat_pred": raspuns_r,
                "tip_rezumat": tip_info["tip"], "nr_cuvinte_pred": nr_cuv,
                "in_limite_lungime": in_limite,
                "latenta_intentie": round(lat_i, 3),
                "latenta_satisfactie": round(lat_s, 3),
                "latenta_rezumat": round(lat_r, 3),
                "latenta_totala": round(lat_tot, 3),
                "model": nume_model,
                "prompturi": PROMPTURI_CASTIGATOARE[nume_model]
            })

            # Salveaza checkpoint dupa fiecare conversatie
            ids_procesate.add(conv_id)
            salveaza_checkpoint(rezultate, nume_model, descriere_set)

        except Exception as e:
            print(f"\n  EROARE la {conv_id}: {e}")
            print(f"  Checkpoint salvat ({len(rezultate)} conversatii). Ruleaza din nou pentru a continua.")
            salveaza_checkpoint(rezultate, nume_model, descriere_set)
            raise

    print(f"  Toate {total} conversatii procesate.")
    sterge_checkpoint(nume_model, descriere_set)
    return rezultate


# ─── METRICI ─────────────────────────────────────────────────────────────────

def calculeaza_si_afiseaza(rezultate, nume_model):
    n = len(rezultate)
    acc_i = sum(1 for r in rezultate if r["intentie_corecta"]) / n
    acc_s = sum(1 for r in rezultate if r["satisfactie_corecta"]) / n
    in_limite = sum(1 for r in rezultate if r["in_limite_lungime"]) / n
    lat_medie = sum(r["latenta_totala"] for r in rezultate) / n
    f1_s = f1_score(
        [r["satisfactie_gold"] for r in rezultate],
        [r["satisfactie_pred"] for r in rezultate],
        average="macro", zero_division=0
    )
    rouge = calculeaza_rouge(
        [r["rezumat_pred"] for r in rezultate],
        [r["rezumat_gold"] for r in rezultate]
    )
    bert = calculeaza_bertscore(
        [r["rezumat_pred"] for r in rezultate],
        [r["rezumat_gold"] for r in rezultate]
    )

    print(f"\n  === {nume_model} ({n} conv) ===")
    print(f"  Intentie Accuracy:    {acc_i:.2%}")
    print(f"  Satisfactie Accuracy: {acc_s:.2%} | F1 Macro: {f1_s:.3f}")
    print(f"  Rezumat ROUGE-1:      {rouge['rouge1']:.4f}")
    print(f"  Rezumat ROUGE-L:      {rouge['rougeL']:.4f}")
    print(f"  Rezumat BERT F1:      {bert['f1']:.4f}")
    print(f"  Rezumat In limite:    {in_limite:.0%}")
    print(f"  Latenta medie totala: {lat_medie:.3f}s")

    return {
        "intentie_accuracy": round(acc_i, 4),
        "satisfactie_accuracy": round(acc_s, 4),
        "satisfactie_f1": round(f1_s, 4),
        "rezumat_rouge1": rouge["rouge1"],
        "rezumat_rougeL": rouge["rougeL"],
        "rezumat_bert_f1": bert["f1"],
        "rezumat_in_limite": round(in_limite, 4),
        "latenta_medie_totala": round(lat_medie, 3)
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_domeniu", type=int, default=2)
    parser.add_argument("--tot_setul", action="store_true")
    parser.add_argument("--model", choices=["gpt", "gemini", "cohere"],
                        help="Ruleaza doar un model specific (default: toate)")
    args = parser.parse_args()

    FOLDER = "./conversatii_adnotate_corectate"
    if args.tot_setul:
        conversatii = incarca_toate(FOLDER)
        descriere_set = "tot_setul"
    else:
        conversatii = selecteaza_subset(FOLDER, n_per_domeniu=args.n_per_domeniu)
        descriere_set = f"subset_{args.n_per_domeniu}_per_domeniu"

    print(f"\n=== PIPELINE COMPLET — MODELE API ===")
    print(f"Conversatii: {len(conversatii)} | Set: {descriere_set}")
    if args.model:
        print(f"Model selectat: {MODELE_DISPONIBILE[args.model]}")
    print()

    # Decide ce modele sa ruleze
    modele_de_rulat = []
    if args.model:
        modele_de_rulat = [args.model]
    else:
        if GEMINI_API_KEY: modele_de_rulat.append("gemini")
        if OPENAI_API_KEY: modele_de_rulat.append("gpt")
        if COHERE_API_KEY: modele_de_rulat.append("cohere")

    toate_rezultatele = {}
    toate_metrici = {}

    config_modele = {
        "gemini": {
            "nume": "Gemini-2.5-flash",
            "fn_intentie": prompt_intentie_v2_1,
            "fn_satisfactie": prompt_satisfactie_v1_1,
            "fn_rezumat": prompt_rezumat_v3_1,
            "fn_api_scurt": lambda p, m: call_gemini(p),
            "fn_api_lung": lambda p: call_gemini(p),
            "activ": bool(GEMINI_API_KEY),
        },
        "gpt": {
            "nume": "GPT-4.1-mini",
            "fn_intentie": prompt_intentie_v2_1,
            "fn_satisfactie": prompt_satisfactie_v3_2,
            "fn_rezumat": prompt_rezumat_v3_1,
            "fn_api_scurt": lambda p, m: call_gpt(p, m),
            "fn_api_lung": lambda p: call_gpt(p, 200),
            "activ": bool(OPENAI_API_KEY),
        },
        "cohere": {
            "nume": "command-r7b-12-2024",
            "fn_intentie": prompt_intentie_v4_1,
            "fn_satisfactie": prompt_satisfactie_v2_1,
            "fn_rezumat": prompt_rezumat_v3_1,
            "fn_api_scurt": lambda p, m: call_cohere(p, m),
            "fn_api_lung": lambda p: call_cohere(p, 200),
            "activ": bool(COHERE_API_KEY),
        },
    }

    for cheie in modele_de_rulat:
        cfg = config_modele[cheie]
        if not cfg["activ"]:
            print(f"Cheie API lipsa pentru {cfg['nume']}, skip.")
            continue

        print(f"\n--- {cfg['nume']} ---")
        rez = proceseaza_model(
            cfg["nume"],
            fn_intentie=cfg["fn_intentie"],
            fn_satisfactie=cfg["fn_satisfactie"],
            fn_rezumat=cfg["fn_rezumat"],
            fn_api_scurt=cfg["fn_api_scurt"],
            fn_api_lung=cfg["fn_api_lung"],
            conversatii=conversatii,
            descriere_set=descriere_set
        )
        toate_rezultatele[cfg["nume"]] = rez
        toate_metrici[cfg["nume"]] = calculeaza_si_afiseaza(rez, cfg["nume"])

    if not toate_metrici:
        print("Niciun model procesat.")
        return

    # Tabel comparativ
    print(f"\n{'='*80}")
    print("TABEL COMPARATIV FINAL")
    print(f"{'='*80}")
    print(f"  {'Model':<22} {'Acc.Int':>8} {'Acc.Sat':>8} {'F1.Sat':>8} {'R-L':>8} {'BERT':>8} {'Limite':>8} {'Latenta':>10}")
    print(f"  {'-'*80}")
    for model, m in toate_metrici.items():
        print(f"  {model:<22} {m['intentie_accuracy']:>8.2%} {m['satisfactie_accuracy']:>8.2%} "
              f"{m['satisfactie_f1']:>8.3f} {m['rezumat_rougeL']:>8.4f} "
              f"{m['rezumat_bert_f1']:>8.4f} {m['rezumat_in_limite']:>8.0%} "
              f"{m['latenta_medie_totala']:>9.3f}s")

    # Salveaza rezultate finale
    toate_rez_flat = []
    for rl in toate_rezultatele.values():
        toate_rez_flat.extend(rl)

    output_file = os.path.join(RESULTS_DIR, f"pipeline_api_{descriere_set}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "set_date": descriere_set,
            "prompturi_castigatoare": PROMPTURI_CASTIGATOARE,
            "metrici": toate_metrici,
            "rezultate_detaliate": toate_rez_flat
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSalvat in: {output_file}")


if __name__ == "__main__":
    main()