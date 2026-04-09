"""
Prompt V3 - Detectare Intentie - Modele Locale
Utilizeaza utils_intentie_local.py

Utilizare:
    # Subset mic (implicit 2 per domeniu = 10 total)
    python3 intentie_local_V3.py --model romistral

    # Subset personalizat
    python3 intentie_local_V3.py --model romistral --n_per_domeniu 4

    # Tot setul de date
    python3 intentie_local_V3.py --model romistral --tot_setul
"""
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_engineering_local.intentii.utils_intentie_local import (
    INTENTII_DOMENII, EXEMPLE_FEWSHOT,
    selecteaza_subset, incarca_toate_conversatiile,
    incarca_model, ruleaza_evaluare, calculeaza_si_afiseaza
)

VERSIUNE = "V3"
RESULTS_DIR = "./rezultate_prompt_engineering"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_prompt(dialog, domeniu):
    """V3: Structured + Constrained prompting (reguli detaliate, ordine intentii, cazuri ambigue)."""
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
    parser = argparse.ArgumentParser(description="Prompt V3 - Detectare Intentie - Modele Locale")
    parser.add_argument("--model", required=True, choices=["romistral", "rogemma"])
    parser.add_argument("--n_per_domeniu", type=int, default=2,
                        help="Conversatii per domeniu pentru subset (default: 2 = 10 total)")
    parser.add_argument("--tot_setul", action="store_true",
                        help="Ruleaza pe toate cele 100 de conversatii")
    args = parser.parse_args()

    print(f"\n=== PROMPT V3 — DETECTARE INTENTIE — MODELE LOCALE ===")
    print(f"Model: {args.model}")

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

    tokenizer, model, device = incarca_model(args.model)

    rezultate = ruleaza_evaluare(
        conversatii, tokenizer, model, device,
        get_prompt, VERSIUNE, args.model, RESULTS_DIR
    )
    metrici = calculeaza_si_afiseaza(rezultate, VERSIUNE, args.model)

    output_file = os.path.join(RESULTS_DIR, f"intentie_local_{args.model}_V3_{descriere_set}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "versiune": VERSIUNE, "set_date": descriere_set,
                    "metrici": metrici, "rezultate_detaliate": rezultate},
                  f, ensure_ascii=False, indent=2)
    print(f"\nRezultate salvate in: {output_file}")


if __name__ == "__main__":
    import argparse
    main()