"""
Prompt V3 - Detectare Intentie - Modele API
GPT-4.1-mini, Gemini-2.5-flash, Aya-Expanse-8b

Utilizare:
    # Subset mic (10 conversatii)
    python3 intentie_api_V3.py

    # Subset personalizat
    python3 intentie_api_V3.py --n_per_domeniu 4

    # Tot setul
    python3 intentie_api_V3.py --tot_setul
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_engineering_API.intentii.utils_intentie_api import (
    INTENTII_DOMENII, EXEMPLE_FEWSHOT_LUNGI,
    selecteaza_subset, incarca_toate_conversatiile,
    ruleaza_evaluare_api, calculeaza_si_afiseaza_api
)

VERSIUNE = "V3"
RESULTS_DIR = "./rezultate_prompt_engineering"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_prompt(dialog, domeniu):
    """V3: Role + Structured + Constrained (Varianta C - conversatie prima)."""
    intentii = INTENTII_DOMENII.get(domeniu, [])
    intentii_str = ", ".join(intentii)
    return (
        "Esti un expert in analiza conversatiilor telefonice din domeniul " + domeniu + ".\n"
        "Identifica intentia sau intentiile clientului din conversatia de mai jos.\n\n"
        "REGULI STRICTE:\n"
        "- Include DOAR ce a cerut sau intrebat clientul, nu actiunile operatorului\n"
        "- Alege EXCLUSIV din lista de intentii furnizata, nu inventa etichete noi\n"
        "- Daca clientul are doua intentii, listeaza-le in ordinea in care au aparut in conversatie\n"
        "- PRIMA intentie din raspunsul tau trebuie sa fie cea principala, cu care a sunat clientul\n"
        "- Daca nu gasesti o potrivire clara, alege cea mai apropiata intentie din lista\n\n"
        "INTENTII DISPONIBILE: " + intentii_str + "\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "Raspunde DOAR cu una sau doua intentii din lista de mai sus, separate prin virgula.\n"
        "Prima intentie trebuie sa fie cea principala:"
    )

def get_prompt2(dialog, domeniu):
    """V3: Zero-shot + Role + Structured + Constrained — conversatie inainte de reguli."""
    intentii = INTENTII_DOMENII.get(domeniu, [])
    intentii_str = ", ".join(intentii)
    return (
        "Esti un expert in clasificarea intentiilor pentru call-center-uri "
        "din domeniul " + domeniu + ".\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "SARCINA: Pe baza conversatiei de mai sus, identifica intentia clientului.\n\n"
        "REGULI:\n"
        "1. Include DOAR ce a cerut sau intrebat clientul — ignora actiunile operatorului\n"
        "2. Alege EXCLUSIV din lista de intentii furnizata — nu inventa etichete noi\n"
        "3. Returneaza MAXIM doua intentii, separate prin virgula\n"
        "4. Prima intentie trebuie sa fie cea principala, cu care a sunat clientul\n\n"
        "INTENTII DISPONIBILE: " + intentii_str + "\n\n"
        "INTENTIE IDENTIFICATA:"
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prompt V3 - Detectare Intentie - Modele API")
    parser.add_argument("--n_per_domeniu", type=int, default=2,
                        help="Conversatii per domeniu pentru subset (default: 2 = 10 total)")
    parser.add_argument("--tot_setul", action="store_true",
                        help="Ruleaza pe toate cele 100 de conversatii")
    args = parser.parse_args()

    print(f"\n=== PROMPT V3 — DETECTARE INTENTIE — MODELE API ===")

    FOLDER_ADNOTAT = "./conversatii_adnotate_corectate"

    if args.tot_setul:
        print("\nIncarc TOATE conversatiile...")
        conversatii = incarca_toate_conversatiile(FOLDER_ADNOTAT)
        descriere_set = "tot_setul"
    else:
        print(f"\nSelectez subset: {args.n_per_domeniu} per domeniu...")
        conversatii = selecteaza_subset(FOLDER_ADNOTAT, n_per_domeniu=args.n_per_domeniu)
        descriere_set = f"subset_{args.n_per_domeniu}_per_domeniu"

    print(f"Total: {len(conversatii)} conversatii")

    toate_rezultatele = ruleaza_evaluare_api(conversatii, get_prompt2, VERSIUNE, RESULTS_DIR)
    toate_metrici = calculeaza_si_afiseaza_api(toate_rezultatele, VERSIUNE)

    # Tabel comparativ final
    print(f"\n{'='*75}")
    print("TABEL COMPARATIV MODELE API")
    print(f"{'='*75}")
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1':>8} {'TTFT':>10} {'Latenta':>10}")
    print(f"  {'-'*65}")
    for m in toate_metrici:
        print(f"  {m['model']:<22} {m['accuracy']:>10.2%} {m['f1']:>8.3f} {m['ttft_medie']:>9.3f}s {m['latenta_medie']:>9.3f}s")

    # Salveaza rezultatele
    output_file = os.path.join(RESULTS_DIR, f"intentie_api_V3_{descriere_set}.json")
    raport_file = os.path.join(RESULTS_DIR, f"intentie_api_V3_{descriere_set}_raport.txt")

    toate_rez_flat = []
    for rez_list in toate_rezultatele.values():
        toate_rez_flat.extend(rez_list)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "versiune": VERSIUNE, "set_date": descriere_set,
            "metrici": toate_metrici,
            "rezultate_detaliate": toate_rez_flat
        }, f, ensure_ascii=False, indent=2)

    raport_complet = "\n".join(m.get("raport_text", "") for m in toate_metrici)
    with open(raport_file, "w", encoding="utf-8") as f:
        f.write(raport_complet)

    print(f"\nRezultate salvate in: {output_file}")
    print(f"Raport text salvat in: {raport_file}")


if __name__ == "__main__":
    main()
