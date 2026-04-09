"""
Prompt V2 - Detectare Intentie - Modele API
GPT-4.1-mini, Gemini-2.5-flash, Aya-Expanse-8b

Utilizare:
    # Subset mic (10 conversatii)
    python3 intentie_api_V2.py

    # Subset personalizat
    python3 intentie_api_V2.py --n_per_domeniu 4

    # Tot setul
    python3 intentie_api_V2.py --tot_setul
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
RESULTS_DIR = "./rezultate_prompt_engineering"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_prompt(dialog, domeniu):
    """V2: Zero-shot + Role prompting (Varianta B - analist de date)."""
    return (
        "Esti un agent specializat in analiza conversatiilor telefonice din call-center-uri "
        "din domeniul " + domeniu + ". Rolul tau este sa identifici intentia clientului.\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "Care este intentia clientului? Raspunde cu un singur cuvant sau expresie scurta:"
    )

def get_prompt2(dialog, domeniu):
    return (
        "Lucrezi ca analist de date intr-un call-center din domeniul " + domeniu + ". "
        "Sarcina ta zilnica este sa identifici motivul pentru care clientii suna, "
        "pe baza transcripturilor conversatiilor cu operatorii.\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "De ce a sunat clientul? Raspunde scurt, in 2-3 cuvinte:"
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prompt V2 - Detectare Intentie - Modele API")
    parser.add_argument("--n_per_domeniu", type=int, default=2,
                        help="Conversatii per domeniu pentru subset (default: 2 = 10 total)")
    parser.add_argument("--tot_setul", action="store_true",
                        help="Ruleaza pe toate cele 100 de conversatii")
    args = parser.parse_args()

    print(f"\n=== PROMPT V2 — DETECTARE INTENTIE — MODELE API ===")

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
    output_file = os.path.join(RESULTS_DIR, f"intentie_api_V2_{descriere_set}_2.json")
    raport_file = os.path.join(RESULTS_DIR, f"intentie_api_V2_{descriere_set}_raport_2.txt")

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
