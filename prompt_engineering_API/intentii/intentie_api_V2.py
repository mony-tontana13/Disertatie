"""
Prompt V2 - Detectare Intentie - Modele API
Zero-shot + Role + Constrained — rol de expert, lista de intentii, reguli de baza.
GPT-4.1-mini, Gemini-2.5-flash, command-r7b-12-2024

Utilizare:
    python3 intentie_api_V2.py
    python3 intentie_api_V2.py --varianta 2
    python3 intentie_api_V2.py --n_per_domeniu 4
    python3 intentie_api_V2.py --tot_setul
    python3 intentie_api_V2.py --varianta 2 --tot_setul
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_intentie_api import (
    INTENTII_DOMENII, EXEMPLE_FEWSHOT_LUNGI,
    selecteaza_subset, incarca_toate_conversatiile,
    ruleaza_evaluare_api, calculeaza_si_afiseaza_api
)

VERSIUNE = "V2"
RESULTS_DIR = "./rezultate_prompt_engineering/intentii_api"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_prompt(dialog, domeniu):
    """V2 varianta 1: Role + Constrained — analist de date + lista intentii."""
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
    """V2 varianta 2: Role + Constrained — agent specializat + lista intentii."""
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
    parser = argparse.ArgumentParser(description="Prompt V2 - Detectare Intentie - Modele API")
    parser.add_argument("--varianta", type=int, default=1, choices=[1, 2],
                        help="Varianta de prompt: 1 sau 2 (default: 1)")
    parser.add_argument("--n_per_domeniu", type=int, default=2,
                        help="Conversatii per domeniu pentru subset (default: 2 = 10 total)")
    parser.add_argument("--tot_setul", action="store_true",
                        help="Ruleaza pe toate cele 100 de conversatii")
    args = parser.parse_args()

    func_prompt = get_prompt if args.varianta == 1 else get_prompt2
    sufix_varianta = f"_v{args.varianta}"
    versiune_completa = f"V2{sufix_varianta}"

    print(f"\n=== PROMPT V2 varianta {args.varianta} — DETECTARE INTENTIE — MODELE API ===")

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

    toate_rezultatele = ruleaza_evaluare_api(conversatii, func_prompt, versiune_completa, RESULTS_DIR)
    toate_metrici = calculeaza_si_afiseaza_api(toate_rezultatele, versiune_completa)

    print(f"\n{'='*75}")
    print("TABEL COMPARATIV MODELE API")
    print(f"{'='*75}")
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1':>8} {'TTFT':>10} {'Latenta':>10}")
    print(f"  {'-'*65}")
    for m in toate_metrici:
        print(f"  {m['model']:<22} {m['accuracy']:>10.2%} {m['f1']:>8.3f} {m['ttft_medie']:>9.3f}s {m['latenta_medie']:>9.3f}s")
    output_file = os.path.join(RESULTS_DIR, f"intentie_api_V2{sufix_varianta}_{descriere_set}.json")
    raport_file = os.path.join(RESULTS_DIR, f"intentie_api_V2{sufix_varianta}_{descriere_set}_raport.txt")

    toate_rez_flat = []
    for rez_list in toate_rezultatele.values():
        toate_rez_flat.extend(rez_list)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "versiune": versiune_completa, "set_date": descriere_set,
            "metrici": toate_metrici, "rezultate_detaliate": toate_rez_flat
        }, f, ensure_ascii=False, indent=2)

    raport_complet = "\n".join(m.get("raport_text", "") for m in toate_metrici)
    with open(raport_file, "w", encoding="utf-8") as f:
        f.write(raport_complet)

    print(f"\nRezultate salvate in: {output_file}")
    print(f"Raport text salvat in: {raport_file}")


if __name__ == "__main__":
    main()