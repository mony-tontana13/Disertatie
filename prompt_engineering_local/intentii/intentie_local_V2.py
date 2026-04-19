"""
Prompt V2 - Detectare Intentie - Modele Locale
Zero-shot + Role + Constrained — rol de expert, lista de intentii, reguli de baza.

Utilizare:
    python3 intentie_local_V2.py --model romistral
    python3 intentie_local_V2.py --model romistral --varianta 2
    python3 intentie_local_V2.py --model romistral --n_per_domeniu 4
    python3 intentie_local_V2.py --model romistral --tot_setul
    python3 intentie_local_V2.py --model romistral --varianta 2 --tot_setul
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_intentie_local import (
    INTENTII_DOMENII, EXEMPLE_FEWSHOT,
    selecteaza_subset, incarca_toate_conversatiile,
    incarca_model, ruleaza_evaluare, calculeaza_si_afiseaza
)

VERSIUNE = "V2"
RESULTS_DIR = "./rezultate_prompt_engineering/intentii_local"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_prompt(dialog, domeniu):
    """V2 varianta 1: Role + Constrained — agent specializat + lista de intentii."""
    intentii = INTENTII_DOMENII.get(domeniu, [])
    intentii_str = ", ".join(intentii)
    return (
        "Lucrezi ca analist de date intr-un call-center din domeniul " + domeniu + ". "
        "Sarcina ta zilnica este sa identifici motivul pentru care clientii suna, "
        "pe baza transcripturilor conversatiilor cu operatorii.\n\n"
        "REGULI:\n"
        "- Include DOAR ce a cerut sau intrebat clientul, nu actiunile operatorului\n"
        "- Alege DOAR din lista de intentii de mai jos\n\n"
        "INTENTII DISPONIBILE: " + intentii_str + "\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "De ce a sunat clientul? Raspunde cu una sau doua intentii din lista:"
    )

def get_prompt2(dialog, domeniu):
    """V2 varianta 2: Role + Constrained — analist de date + lista intentii + cerere 2-3 cuvinte."""
    intentii = INTENTII_DOMENII.get(domeniu, [])
    intentii_str = ", ".join(intentii)
    return (
        "Esti un agent specializat in analiza conversatiilor telefonice din call-center-uri "
        "din domeniul " + domeniu + ". Rolul tau este sa identifici intentia clientului.\n\n"
        "REGULI:\n"
        "- Include DOAR ce a cerut sau intrebat clientul, nu actiunile operatorului\n"
        "- Alege DOAR din lista de intentii de mai jos\n\n"
        "INTENTII DISPONIBILE: " + intentii_str + "\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "Raspunde cu una sau doua intentii din lista, separate prin virgula:"
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prompt V2 - Detectare Intentie - Modele Locale")
    parser.add_argument("--model", required=True, choices=["romistral", "rogemma"])
    parser.add_argument("--varianta", type=int, default=1, choices=[1, 2],
                        help="Varianta de prompt: 1 sau 2 (default: 1)")
    parser.add_argument("--n_per_domeniu", type=int, default=10,
                        help="Conversatii per domeniu pentru subset (default: 2 = 10 total)")
    parser.add_argument("--tot_setul", action="store_true",
                        help="Ruleaza pe toate cele 100 de conversatii")
    args = parser.parse_args()

    func_prompt = get_prompt if args.varianta == 1 else get_prompt2
    sufix_varianta = f"_v{args.varianta}"

    print(f"\n=== PROMPT V2 varianta {args.varianta} — DETECTARE INTENTIE — MODELE LOCALE ===")
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

    versiune_completa = f"V2{sufix_varianta}"
    rezultate = ruleaza_evaluare(
        conversatii, tokenizer, model, device,
        func_prompt, versiune_completa, args.model, RESULTS_DIR
    )
    metrici = calculeaza_si_afiseaza(rezultate, versiune_completa, args.model)

    output_file = os.path.join(RESULTS_DIR, f"intentie_local_{args.model}_V2{sufix_varianta}_{descriere_set}.json")
    raport_file = os.path.join(RESULTS_DIR, f"intentie_local_{args.model}_V2{sufix_varianta}_{descriere_set}_raport.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model, "versiune": versiune_completa, "set_date": descriere_set,
            "metrici": metrici, "rezultate_detaliate": rezultate
        }, f, ensure_ascii=False, indent=2)

    with open(raport_file, "w", encoding="utf-8") as f:
        f.write(metrici.get("raport_text", ""))

    print(f"\nRezultate salvate in: {output_file}")
    print(f"Raport text salvat in: {raport_file}")


if __name__ == "__main__":
    main()