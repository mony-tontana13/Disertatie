"""
Pipeline complet - Modele Locale (RoMistral + RoGemma)
Ruleaza cele 3 sarcini cu prompturile castigatoare per model:
  RoGemma:   Intentie=V4-1 | Satisfactie=V2-2 | Rezumat=V2-2
  RoMistral: Intentie=V4-1 | Satisfactie=V1-1 | Rezumat=V2-1

Utilizare:
    python3 pipeline_local.py --model romistral
    python3 pipeline_local.py --model rogemma
    python3 pipeline_local.py --model romistral --n_per_domeniu 4
"""
import json
import os
import sys
import time
import unicodedata
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_engineering_local.intentii.utils_intentie_local import INTENTII_DOMENII, EXEMPLE_FEWSHOT, incarca_model, genereaza_raspuns, extrage_intentie
from prompt_engineering_local.satisfactie.utils_satisfactie_local import extrage_satisfactie
from prompt_engineering_local.rezumat.utils_rezumat_local import get_tip_rezumat, calculeaza_rouge, calculeaza_bertscore

RESULTS_DIR = "./rezultate_evaluare/pipeline_local"
os.makedirs(RESULTS_DIR, exist_ok=True)

PROMPTURI_CASTIGATOARE = {
    "rogemma":   {"intentie": "V4-1", "satisfactie": "V2-2", "rezumat": "V2-2"},
    "romistral": {"intentie": "V4-1", "satisfactie": "V1-1", "rezumat": "V2-1"},
}


# ─── PROMPTURI INTENTIE ───────────────────────────────────────────────────────

def prompt_intentie_v4_1(dialog, domeniu):
    """V4 varianta 1 — cel mai bun pentru ambele modele locale."""
    intentii = INTENTII_DOMENII.get(domeniu, [])
    intentii_str = ", ".join(intentii)
    exemple_lungi = {
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
    exemple = exemple_lungi.get(domeniu, [])
    exemple_text = ""
    for dialog_ex, intentie_ex in exemple:
        exemple_text += "CONVERSATIE:\n" + dialog_ex + "\nINTENTIE IDENTIFICATA: " + intentie_ex + "\n\n"
    return (
        "Esti un expert in clasificarea intentiilor pentru call-center-uri din domeniul " + domeniu + ".\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "SARCINA: Pe baza conversatiei de mai sus, identifica intentia clientului.\n\n"
        "REGULI:\n"
        "1. Include DOAR ce a cerut sau intrebat clientul\n"
        "2. Alege EXCLUSIV din lista de intentii furnizata\n"
        "3. Returneaza MAXIM doua intentii, separate prin virgula\n"
        "4. Prima intentie trebuie sa fie cea principala\n\n"
        "INTENTII DISPONIBILE: " + intentii_str + "\n\n"
        "EXEMPLE:\n" + exemple_text +
        "INTENTIE IDENTIFICATA:"
    )


# ─── PROMPTURI SATISFACTIE ────────────────────────────────────────────────────

def prompt_satisfactie_v2_2(dialog):
    """V2-2 — cel mai bun pentru RoGemma."""
    return (
        "Esti un expert in analiza satisfactiei clientilor in conversatii de call-center.\n"
        "Determina nivelul de satisfactie al clientului la finalul conversatiei de mai jos.\n\n"
        "DEFINITII DETALIATE:\n"
        "- pozitiv: clientul pleaca multumit si o exprima explicit\n"
        "  Expresii tipice: multumesc mult, excelent, exact ce aveam nevoie, sunteti promti\n"
        "- neutru: problema rezolvata dar clientul nu exprima nicio emotie\n"
        "  Expresii tipice: ok, am inteles, bine, la revedere fara caldura\n"
        "- negativ: clientul pleaca frustrat, chiar daca accepta situatia\n"
        "  Expresii tipice: ce sa fac, bine..., nu am de ales, iarasi aceeasi problema, ironie\n\n"
        "REGULA CRITICA neutru vs negativ:\n"
        "- Accepta solutia fara entuziasm dar FARA frustrare vizibila -> neutru\n"
        "- Accepta solutia cu resemnare, ironie sau nemultumire implicita -> negativ\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "Raspunde DOAR cu unul dintre cuvintele: pozitiv, neutru, negativ:"
    )

def prompt_satisfactie_v1_1(dialog):
    """V1-1 — cel mai bun pentru RoMistral."""
    return (
        "Analizeaza urmatoarea conversatie telefonica si determina nivelul de satisfactie al clientului.\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "Care este satisfactia clientului la finalul conversatiei? "
        "Raspunde cu un singur cuvant: pozitiv, neutru sau negativ:"
    )


# ─── PROMPTURI REZUMAT ────────────────────────────────────────────────────────

def prompt_rezumat_v2_1(dialog, satisfactie):
    """V2-1 — cel mai bun pentru RoMistral."""
    tip_info = get_tip_rezumat(satisfactie)
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

def prompt_rezumat_v2_2(dialog, satisfactie):
    """V2-2 — cel mai bun pentru RoGemma."""
    tip_info = get_tip_rezumat(satisfactie)
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


# ─── SELECTARE SUBSET ─────────────────────────────────────────────────────────

def selecteaza_subset(folder, n_per_domeniu=2):
    random.seed(42)
    subset = []
    domenii = ["banking", "medicina", "retail", "telecom", "servicii_publice"]
    for domeniu in domenii:
        domeniu_path = os.path.join(folder, domeniu)
        if not os.path.isdir(domeniu_path):
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
    return subset


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["romistral", "rogemma"])
    parser.add_argument("--n_per_domeniu", type=int, default=2)
    args = parser.parse_args()

    FOLDER_ADNOTAT = "./conversatii_adnotate_corectate"
    conversatii = selecteaza_subset(FOLDER_ADNOTAT, n_per_domeniu=args.n_per_domeniu)
    descriere_set = f"subset_{args.n_per_domeniu}_per_domeniu"

    print(f"\n=== PIPELINE COMPLET — {args.model.upper()} ===")
    print(f"Prompturi: {PROMPTURI_CASTIGATOARE[args.model]}")
    print(f"Conversatii: {len(conversatii)}\n")

    tokenizer, model, device = incarca_model(args.model)

    # Alege prompturile in functie de model
    if args.model == "rogemma":
        fn_satisfactie = prompt_satisfactie_v2_2
        fn_rezumat = prompt_rezumat_v2_2
    else:
        fn_satisfactie = prompt_satisfactie_v1_1
        fn_rezumat = prompt_rezumat_v2_1

    rezultate = []
    for i, conv in enumerate(conversatii):
        conv_id = conv["id"]
        domeniu = conv.get("domeniu", "banking")
        complexitate = conv.get("complexitate", "?")
        dialog = "\n".join([r["rol"].upper() + ": " + r["text"] for r in conv["conversatie"]])
        intentie_gold = conv.get("intentie_gold", ["alta_solicitare"])
        if isinstance(intentie_gold, list):
            intentie_gold = intentie_gold[0]
        satisfactie_gold = conv.get("satisfactie", "neutru")
        rezumat_gold = conv.get("rezumat", "")

        print(f"[{i+1:02d}/{len(conversatii)}] {conv_id} [{domeniu}/{complexitate}]")

        # 1. Intentie
        t0 = time.time()
        raspuns_intentie, lat_intentie = genereaza_raspuns(tokenizer, model, device, prompt_intentie_v4_1(dialog, domeniu), max_new_tokens=40)
        intentie_pred = extrage_intentie(raspuns_intentie, domeniu)
        print(f"  Intentie: gold={intentie_gold} | pred={intentie_pred} | {lat_intentie:.1f}s")

        # 2. Satisfactie (foloseste gold pentru rezumat)
        raspuns_satisfactie, lat_satisfactie = genereaza_raspuns(tokenizer, model, device, fn_satisfactie(dialog), max_new_tokens=20)
        satisfactie_pred = extrage_satisfactie(raspuns_satisfactie)
        print(f"  Satisfactie: gold={satisfactie_gold} | pred={satisfactie_pred} | {lat_satisfactie:.1f}s")

        # 3. Rezumat (foloseste satisfactie gold ca in prompt engineering)
        tip_info = get_tip_rezumat(satisfactie_gold)
        raspuns_rezumat, lat_rezumat = genereaza_raspuns(tokenizer, model, device, fn_rezumat(dialog, satisfactie_gold), max_new_tokens=150)
        nr_cuv = len(raspuns_rezumat.split())
        in_limite = tip_info["min_cuv"] <= nr_cuv <= tip_info["max_cuv"]
        print(f"  Rezumat: {nr_cuv} cuv | {'OK' if in_limite else 'LUNGIME'} | {lat_rezumat:.1f}s")

        rezultate.append({
            "id": conv_id, "domeniu": domeniu, "complexitate": complexitate,
            "intentie_gold": intentie_gold, "intentie_pred": intentie_pred,
            "intentie_corecta": intentie_pred == intentie_gold,
            "satisfactie_gold": satisfactie_gold, "satisfactie_pred": satisfactie_pred,
            "satisfactie_corecta": satisfactie_pred == satisfactie_gold,
            "rezumat_gold": rezumat_gold, "rezumat_pred": raspuns_rezumat,
            "tip_rezumat": tip_info["tip"],
            "nr_cuvinte_pred": nr_cuv, "in_limite_lungime": in_limite,
            "latenta_intentie": round(lat_intentie, 2),
            "latenta_satisfactie": round(lat_satisfactie, 2),
            "latenta_rezumat": round(lat_rezumat, 2),
            "latenta_totala": round(lat_intentie + lat_satisfactie + lat_rezumat, 2),
            "model": args.model,
            "prompturi": PROMPTURI_CASTIGATOARE[args.model]
        })

    # Metrici finale
    print(f"\n=== REZULTATE FINALE — {args.model.upper()} ===")
    n = len(rezultate)
    acc_intentie = sum(1 for r in rezultate if r["intentie_corecta"]) / n
    acc_satisfactie = sum(1 for r in rezultate if r["satisfactie_corecta"]) / n
    in_limite_pct = sum(1 for r in rezultate if r["in_limite_lungime"]) / n
    lat_medie = sum(r["latenta_totala"] for r in rezultate) / n

    predictii_rez = [r["rezumat_pred"] for r in rezultate]
    referinte_rez = [r["rezumat_gold"] for r in rezultate]
    rouge = calculeaza_rouge(predictii_rez, referinte_rez)
    bert = calculeaza_bertscore(predictii_rez, referinte_rez)

    print(f"  Intentie   Accuracy:  {acc_intentie:.2%}")
    print(f"  Satisfactie Accuracy: {acc_satisfactie:.2%}")
    print(f"  Rezumat ROUGE-1:      {rouge['rouge1']:.4f}")
    print(f"  Rezumat ROUGE-L:      {rouge['rougeL']:.4f}")
    print(f"  Rezumat BERT F1:      {bert['f1']:.4f}")
    print(f"  Rezumat In limite:    {in_limite_pct:.0%}")
    print(f"  Latenta medie totala: {lat_medie:.1f}s")

    output = {
        "model": args.model, "set_date": descriere_set,
        "prompturi_castigatoare": PROMPTURI_CASTIGATOARE[args.model],
        "metrici": {
            "intentie_accuracy": round(acc_intentie, 4),
            "satisfactie_accuracy": round(acc_satisfactie, 4),
            "rezumat_rouge1": rouge["rouge1"], "rezumat_rougeL": rouge["rougeL"],
            "rezumat_bert_f1": bert["f1"],
            "rezumat_in_limite": round(in_limite_pct, 4),
            "latenta_medie_totala": round(lat_medie, 2)
        },
        "rezultate_detaliate": rezultate
    }

    output_file = os.path.join(RESULTS_DIR, f"pipeline_{args.model}_{descriere_set}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSalvat in: {output_file}")


if __name__ == "__main__":
    main()
