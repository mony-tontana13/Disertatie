"""
Prompt V2 - Generare Rezumat - Modele API
Doua variante: get_prompt (varianta 1) si get_prompt2 (varianta 2).

Utilizare:
    python3 rezumat_api_V2.py
    python3 rezumat_api_V2.py --varianta 2
    python3 rezumat_api_V2.py --tot_setul
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_rezumat_api import (
    get_tip_rezumat,
    selecteaza_subset, incarca_toate_conversatiile,
    ruleaza_evaluare_api, calculeaza_si_afiseaza_api
)

VERSIUNE = "V2"
RESULTS_DIR = "./rezultate_prompt_engineering/rezumat_api"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_prompt(dialog, satisfactie):
    """V2 varianta 1: Role + Constrained — tip rezumat + cerinte de baza."""
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
def get_prompt2(dialog, satisfactie):
    """V2 varianta 2: Role + Constrained — structura explicita a rezumatului."""
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--varianta", type=int, default=1, choices=[1, 2])
    parser.add_argument("--n_per_domeniu", type=int, default=2)
    parser.add_argument("--tot_setul", action="store_true")
    args = parser.parse_args()

    func_prompt = get_prompt if args.varianta == 1 else get_prompt2
    sufix_varianta = f"_v{args.varianta}"
    versiune_completa = f"V2{sufix_varianta}"

    print(f"\n=== PROMPT V2 varianta {args.varianta} — REZUMAT — MODELE API ===")

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

    print(f"\n{'='*65}")
    print("TABEL COMPARATIV")
    print(f"  {'Model':<22} {'ROUGE-1':>8} {'ROUGE-L':>8} {'BERT F1':>8} {'TTFT':>8}")
    print(f"  {'—'*56}")
    for m in toate_metrici:
        print(f"  {m['model']:<22} {m['rouge1']:>8.4f} {m['rougeL']:>8.4f} {m['bertscore_f1']:>8.4f} {m['ttft_medie']:>7.3f}s")

    output_file = os.path.join(RESULTS_DIR, f"rezumat_api_V2{sufix_varianta}_{descriere_set}.json")
    raport_file = os.path.join(RESULTS_DIR, f"rezumat_api_V2{sufix_varianta}_{descriere_set}_raport.txt")

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
