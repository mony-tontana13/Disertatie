"""
Prompt V3 - Estimare Satisfactie - Modele Locale
Doua variante de prompt: get_prompt (varianta 1) si get_prompt2 (varianta 2).

Utilizare:
    python3 satisfactie_local_V3.py --model romistral
    python3 satisfactie_local_V3.py --model romistral --varianta 2
    python3 satisfactie_local_V3.py --model romistral --tot_setul
    python3 satisfactie_local_V3.py --model romistral --varianta 2 --tot_setul
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_satisfactie_local import (
    EXEMPLE_SCURTE, EXEMPLE_LUNGI,
    selecteaza_subset, incarca_toate_conversatiile,
    incarca_model, ruleaza_evaluare, calculeaza_si_afiseaza
)

VERSIUNE = "V3"
RESULTS_DIR = "./rezultate_prompt_engineering/satisfactie_local"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_prompt(dialog):
    """V3 varianta 1: Role + Constrained + Structured + Few-shot cu exemple scurte."""
    exemple_text = ""
    for clasa, (dialog_ex, satisfactie_ex) in EXEMPLE_SCURTE.items():
        exemple_text += (
            "CONVERSATIE:\n" + dialog_ex + "\n"
            "SATISFACTIE: " + satisfactie_ex + "\n\n"
        )

    return (
        "Esti un expert in analiza satisfactiei clientilor in conversatii de call-center.\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "SARCINA: Pe baza conversatiei de mai sus, determina nivelul de satisfactie al clientului.\n\n"
        "DEFINITII:\n"
        "- pozitiv: problema rezolvata complet, clientul multumit si o exprima clar\n"
        "- neutru: problema rezolvata tehnic dar clientul nu prezinta nicio emotie\n"
        "- negativ: clientul pleaca frustrat sau nemultumit, chiar daca implicit\n"
        "  (ironie, resemnare, replici taioase, incheie brusc conversatia)\n\n"
        "REGULI:\n"
        "1. Un singur comentariu negativ urmat de acceptare NU inseamna automat negativ\n"
        "2. Uita-te la TONUL GENERAL si REZULTATUL FINAL al conversatiei\n"
        "3. Frustrarea implicita conteaza: 'bine, inteleg', 'ce sa fac', remarci ironice\n\n"
        "EXEMPLE:\n" + exemple_text +
        "SATISFACTIE IDENTIFICATA:"
    )
def get_prompt2(dialog):
    """V3 varianta 2: Role + Constrained + Structured + Few-shot cu exemple lungi (granita neutru/negativ)."""
    exemple_text = ""
    for clasa, (dialog_ex, satisfactie_ex) in EXEMPLE_LUNGI.items():
        exemple_text += (
            "CONVERSATIE:\n" + dialog_ex + "\n"
            "SATISFACTIE: " + satisfactie_ex + "\n\n"
        )

    return (
        "Esti un expert in analiza satisfactiei clientilor in conversatii de call-center.\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "SARCINA: Pe baza conversatiei de mai sus, determina nivelul de satisfactie al clientului.\n\n"
        "DEFINITII:\n"
        "- pozitiv: problema rezolvata complet, clientul multumit si o exprima clar\n"
        "- neutru: problema rezolvata tehnic dar clientul nu prezinta nicio emotie\n"
        "- negativ: clientul pleaca frustrat sau nemultumit, chiar daca implicit\n"
        "  (ironie, resemnare, replici taioase, incheie brusc conversatia)\n\n"
        "REGULI:\n"
        "1. Un singur comentariu negativ urmat de acceptare NU inseamna automat negativ\n"
        "2. Uita-te la TONUL GENERAL si REZULTATUL FINAL al conversatiei\n"
        "3. Frustrarea implicita conteaza: 'bine, inteleg', 'ce sa fac', remarci ironice\n\n"
        "EXEMPLE (acorda atentie diferentei dintre neutru si negativ):\n" + exemple_text +
        "SATISFACTIE IDENTIFICATA:"
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prompt V3 - Satisfactie - Modele Locale")
    parser.add_argument("--model", required=True, choices=["romistral", "rogemma"])
    parser.add_argument("--varianta", type=int, default=1, choices=[1, 2],
                        help="Varianta de prompt: 1 sau 2 (default: 1)")
    parser.add_argument("--n_per_domeniu", type=int, default=2)
    parser.add_argument("--tot_setul", action="store_true")
    args = parser.parse_args()

    func_prompt = get_prompt if args.varianta == 1 else get_prompt2
    sufix_varianta = f"_v{args.varianta}"

    print(f"\n=== PROMPT V3 varianta {args.varianta} — SATISFACTIE — MODELE LOCALE ===")
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

    versiune_completa = f"V3{sufix_varianta}"
    rezultate = ruleaza_evaluare(
        conversatii, tokenizer, model, device,
        func_prompt, versiune_completa, args.model, RESULTS_DIR
    )
    metrici = calculeaza_si_afiseaza(rezultate, versiune_completa, args.model)

    output_file = os.path.join(RESULTS_DIR, f"satisfactie_local_{args.model}_V3{sufix_varianta}_{descriere_set}.json")
    raport_file = os.path.join(RESULTS_DIR, f"satisfactie_local_{args.model}_V3{sufix_varianta}_{descriere_set}_raport.txt")

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
