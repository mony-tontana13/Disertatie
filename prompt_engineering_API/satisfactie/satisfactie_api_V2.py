"""
Prompt V2 - Estimare Satisfactie - Modele API
Doua variante: get_prompt (varianta 1) si get_prompt2 (varianta 2).

Utilizare:
    python3 satisfactie_api_V2.py
    python3 satisfactie_api_V2.py --varianta 2
    python3 satisfactie_api_V2.py --tot_setul
    python3 satisfactie_api_V2.py --varianta 2 --tot_setul
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_satisfactie_api import (
    EXEMPLE_SCURTE, EXEMPLE_LUNGI,
    selecteaza_subset, incarca_toate_conversatiile,
    ruleaza_evaluare_api, calculeaza_si_afiseaza_api
)

VERSIUNE = "V2"
RESULTS_DIR = "./rezultate_prompt_engineering/satisfactie_api"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_prompt(dialog):
    """V2 varianta 1: Role + Constrained — definitii scurte + reguli de baza."""
    return (
        "Esti un expert in analiza satisfactiei clientilor in conversatii de call-center.\n"
        "Determina nivelul de satisfactie al clientului la finalul conversatiei de mai jos.\n\n"
        "DEFINITII:\n"
        "- pozitiv: problema rezolvata complet, clientul multumit si o exprima clar\n"
        "- neutru: problema rezolvata tehnic dar clientul nu prezinta nicio emotie\n"
        "- negativ: clientul pleaca frustrat sau nemultumit, chiar daca implicit\n"
        "  (ironie, resemnare, replici taioase, incheie brusc conversatia)\n\n"
        "REGULI:\n"
        "- Un singur comentariu negativ urmat de acceptare NU inseamna automat negativ\n"
        "- Uita-te la TONUL GENERAL si REZULTATUL FINAL al conversatiei\n"
        "- Frustrarea implicita conteaza: bine inteleg, ce sa fac, remarci ironice\n\n"
        "Conversatie:\n" + dialog + "\n\n"
        "Raspunde DOAR cu unul dintre cuvintele: pozitiv, neutru, negativ:"
    )
def get_prompt2(dialog):
    """V2 varianta 2: Role + Constrained — definitii detaliate cu expresii indicator."""
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

    print(f"\n=== PROMPT V2 varianta {args.varianta} — SATISFACTIE — MODELE API ===")

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

    print(f"\n{\'=\'*65}")
    print("TABEL COMPARATIV")
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1':>8} {'TTFT':>10}")
    print(f"  {'—'*52}")
    for m in toate_metrici:
        print(f"  {m[\'model\']:<22} {m[\'accuracy\']:>10.2%} {m[\'f1\']:>8.3f} {m[\'ttft_medie\']:>9.3f}s")

    output_file = os.path.join(RESULTS_DIR, f"satisfactie_api_V2{sufix_varianta}_{descriere_set}.json")
    raport_file = os.path.join(RESULTS_DIR, f"satisfactie_api_V2{sufix_varianta}_{descriere_set}_raport.txt")

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
