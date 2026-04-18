"""
Prompt V3 - Generare Rezumat - Modele API
Doua variante: get_prompt (varianta 1) si get_prompt2 (varianta 2).

Utilizare:
    python3 rezumat_api_V3.py
    python3 rezumat_api_V3.py --varianta 2
    python3 rezumat_api_V3.py --tot_setul
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

VERSIUNE = "V3"
RESULTS_DIR = "./rezultate_prompt_engineering/rezumat_api"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_prompt(dialog, satisfactie):
    """V3 varianta 1: Role + Constrained + Structured + Few-shot exemple scurte."""
    tip_info = get_tip_rezumat(satisfactie)
    exemple = {
        "SCURT": (
            "CLIENT: Am pierdut cardul, il vreau blocat.\nOPERATOR: L-am blocat, va trimitem altul in 3 zile.",
            "Clientul a sunat pentru a bloca un card pierdut. Operatorul a blocat cardul si a initiat emiterea unuia nou."
        ),
        "MEDIU": (
            "CLIENT: Comanda mea nu a sosit.\nOPERATOR: A fost o intarziere la curier. Va ajunge maine.\nCLIENT: Ok.",
            "Clientul a reclamat o comanda nelivrata la termen. Operatorul a verificat situatia si a identificat o intarziere la curier. Comanda urmeaza sa fie livrata a doua zi. Clientul a acceptat solutia."
        ),
        "LUNG": (
            "CLIENT: Am sunat de trei ori pentru aceeasi problema cu factura.\nOPERATOR: Imi pare rau, investigam.\nCLIENT: Bine, astept.",
            "Clientul a contactat call-center-ul pentru a treia oara in legatura cu aceeasi problema de facturare nerezolvata. Clientul si-a exprimat nemultumirea fata de lipsa unei solutii dupa contactele anterioare. Operatorul si-a cerut scuze si a initiat o investigatie interna. Clientul a acceptat sa astepte, exprimand insa o frustrare evidenta. Problema ramane deschisa si necesita urmarire."
        ),
    }
    tip = tip_info["tip"]
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
def get_prompt2(dialog, satisfactie):
    """V3 varianta 2: Role + Constrained + Structured + Few-shot exemple cu structura explicita."""
    tip_info = get_tip_rezumat(satisfactie)
    exemple = {
        "SCURT": (
            "CLIENT: Am pierdut cardul, il vreau blocat.\nOPERATOR: L-am blocat, va trimitem altul in 3 zile.",
            "Clientul a solicitat blocarea unui card pierdut. Operatorul a rezolvat imediat si a initiat emiterea unui card nou."
        ),
        "MEDIU": (
            "CLIENT: Comanda mea nu a sosit.\nOPERATOR: A fost o intarziere la curier. Va ajunge maine.\nCLIENT: Ok.",
            "Motivul apelului: comanda nelivrata la termen. Actiuni operator: verificarea statusului si identificarea intarzierii la curier. Rezultat: livrare reprogramata pentru a doua zi, clientul a acceptat."
        ),
        "LUNG": (
            "CLIENT: Am sunat de trei ori pentru aceeasi problema cu factura.\nOPERATOR: Investigam.\nCLIENT: Astept.",
            "Motivul apelului: eroare de facturare persistenta, al treilea contact al clientului pe aceeasi problema. Context: lipsa solutiei in contactele anterioare a generat frustrare acumulata. Actiuni operator: initierea unei investigatii interne, solicitare timp suplimentar. Reactia clientului: acceptare cu resemnare evidenta, fara satisfactie. Rezultat: problema ramane deschisa, necesita monitorizare prioritara."
        ),
    }
    tip = tip_info["tip"]
    dialog_ex, rezumat_ex = exemple[tip]
    return (
        "Esti un expert in sumarizarea conversatiilor telefonice din call-center.\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "SARCINA: Genereaza un rezumat in romana de tip " + tip + ".\n\n"
        "CERINTE:\n"
        "- Lungime: " + str(tip_info["min_cuv"]) + "-" + str(tip_info["max_cuv"]) + " cuvinte (" + tip_info["propozitii"] + ")\n"
        "- Mentioneaza motivul apelului, actiunile operatorului si rezultatul final\n"
        "- Nu adauga informatii care nu apar in conversatie\n\n"
        "EXEMPLU:\nCONVERSATIE:\n" + dialog_ex + "\nREZUMAT " + tip + ":\n" + rezumat_ex + "\n\n"
        "REZUMAT " + tip + ":"
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
    versiune_completa = f"V3{sufix_varianta}"

    print(f"\n=== PROMPT V3 varianta {args.varianta} — REZUMAT — MODELE API ===")

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

    output_file = os.path.join(RESULTS_DIR, f"rezumat_api_V3{sufix_varianta}_{descriere_set}.json")
    raport_file = os.path.join(RESULTS_DIR, f"rezumat_api_V3{sufix_varianta}_{descriere_set}_raport.txt")

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
