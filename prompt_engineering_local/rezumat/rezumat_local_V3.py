"""
Prompt V3 - Generare Rezumat - Modele Locale
Doua variante: get_prompt (varianta 1) si get_prompt2 (varianta 2).

Utilizare:
    python3 rezumat_local_V3.py --model romistral
    python3 rezumat_local_V3.py --model romistral --varianta 2
    python3 rezumat_local_V3.py --model romistral --tot_setul
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_rezumat_local import (
    get_tip_rezumat,
    selecteaza_subset, incarca_toate_conversatiile,
    incarca_model, ruleaza_evaluare, calculeaza_si_afiseaza
)

VERSIUNE = "V3"
RESULTS_DIR = "./rezultate_prompt_engineering/rezumat_local"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_prompt(dialog, satisfactie):
    """V3 varianta 1: Role + Constrained + Structured + Few-shot cu exemple scurte per tip."""
    tip_info = get_tip_rezumat(satisfactie)

    exemple = {
        "SCURT": (
            "CLIENT: Am pierdut cardul, il vreau blocat.\nOPERATOR: L-am blocat, va trimitem altul in 3 zile.",
            "Clientul a sunat pentru a bloca un card pierdut. Operatorul a blocat cardul si a initiat emiterea unuia nou."
        ),
        "MEDIU": (
            "CLIENT: Comanda mea nu a sosit.\nOPERATOR: Verificam, a fost o intarziere la curier. Va ajunge maine.\nCLIENT: Ok.",
            "Clientul a contactat call-center-ul pentru a reclama o comanda nelivrata la termen. Operatorul a verificat situatia si a informat clientul ca livrarea a intarziat din cauza curierului. Comanda urmeaza sa fie livrata a doua zi. Clientul a acceptat solutia."
        ),
        "LUNG": (
            "CLIENT: Am sunat de trei ori pentru aceeasi problema cu factura.\nOPERATOR: Imi pare rau, investigam.\nCLIENT: Bine, astept.",
            "Clientul a contactat call-center-ul pentru a treia oara in legatura cu aceeasi problema de facturare nerezolvata. Clientul si-a exprimat nemultumirea fata de lipsa unei solutii dupa contactele anterioare. Operatorul si-a cerut scuze si a initiat o investigatie interna. Clientul a acceptat sa astepte, exprimand insa o frustrare evidenta fata de procesul de rezolvare. Problema ramane deschisa si necesita urmarire."
        ),
    }

    tip = tip_info["tip"]
    dialog_ex, rezumat_ex = exemple[tip]
    exemplu_text = (
        "CONVERSATIE:\n" + dialog_ex + "\n"
        "REZUMAT " + tip + ":\n" + rezumat_ex + "\n\n"
    )

    return (
        "Esti un expert in sumarizarea conversatiilor telefonice din call-center.\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "SARCINA: Genereaza un rezumat de tip " + tip + " al conversatiei de mai sus.\n\n"
        "CERINTE:\n"
        "- Lungime: " + str(tip_info["min_cuv"]) + "-" + str(tip_info["max_cuv"]) + " cuvinte (" + tip_info["propozitii"] + ")\n"
        "- Limba: romana\n"
        "- Mentioneaza problema principala si rezultatul final\n"
        "- Nu adauga informatii care nu apar in conversatie\n\n"
        "EXEMPLU:\n" + exemplu_text +
        "REZUMAT " + tip + ":"
    )
def get_prompt2(dialog, satisfactie):
    """V3 varianta 2: Role + Constrained + Structured + Few-shot cu structura explicita in exemple."""
    tip_info = get_tip_rezumat(satisfactie)

    exemple = {
        "SCURT": (
            "CLIENT: Am pierdut cardul, il vreau blocat.\nOPERATOR: L-am blocat, va trimitem altul in 3 zile.",
            "Clientul a solicitat blocarea unui card pierdut. Operatorul a rezolvat imediat solicitarea si a initiat emiterea unui card de inlocuire."
        ),
        "MEDIU": (
            "CLIENT: Comanda mea nu a sosit.\nOPERATOR: Verificam, a fost o intarziere la curier. Va ajunge maine.\nCLIENT: Ok.",
            "Motivul apelului: clientul a reclamat o comanda nelivrata la termen. Actiuni operator: verificarea statusului comenzii si identificarea intarzierii la curier. Rezultat: livrarea reprogramata pentru a doua zi, clientul a acceptat solutia."
        ),
        "LUNG": (
            "CLIENT: Am sunat de trei ori pentru aceeasi problema cu factura.\nOPERATOR: Imi pare rau, investigam.\nCLIENT: Bine, astept.",
            "Motivul apelului: clientul a contactat call-center-ul pentru a treia oara in legatura cu o eroare de facturare persistenta. Context: problema nu a fost rezolvata in urma contactelor anterioare, generand frustrare acumulata. Actiuni operator: initierea unei investigatii interne si solicitarea de timp suplimentar. Reactia clientului: acceptare cu resemnare evidenta, fara satisfactie fata de procesul de rezolvare. Rezultat: problema ramane deschisa, necesita monitorizare si urmarire prioritara."
        ),
    }

    tip = tip_info["tip"]
    dialog_ex, rezumat_ex = exemple[tip]
    exemplu_text = (
        "CONVERSATIE:\n" + dialog_ex + "\n"
        "REZUMAT " + tip + ":\n" + rezumat_ex + "\n\n"
    )

    return (
        "Esti un expert in sumarizarea conversatiilor telefonice din call-center.\n\n"
        "CONVERSATIE:\n" + dialog + "\n\n"
        "SARCINA: Genereaza un rezumat de tip " + tip + " al conversatiei de mai sus.\n\n"
        "CERINTE:\n"
        "- Lungime: " + str(tip_info["min_cuv"]) + "-" + str(tip_info["max_cuv"]) + " cuvinte (" + tip_info["propozitii"] + ")\n"
        "- Limba: romana\n"
        "- Mentioneaza motivul apelului, actiunile operatorului si rezultatul final\n"
        "- Nu adauga informatii care nu apar in conversatie\n\n"
        "EXEMPLU:\n" + exemplu_text +
        "REZUMAT " + tip + ":"
    )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["romistral", "rogemma"])
    parser.add_argument("--varianta", type=int, default=1, choices=[1, 2])
    parser.add_argument("--n_per_domeniu", type=int, default=2)
    parser.add_argument("--tot_setul", action="store_true")
    args = parser.parse_args()

    func_prompt = get_prompt if args.varianta == 1 else get_prompt2
    sufix_varianta = f"_v{args.varianta}"
    versiune_completa = f"V3{sufix_varianta}"

    print(f"\n=== PROMPT V3 varianta {args.varianta} — REZUMAT — MODELE LOCALE ===")
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
        func_prompt, versiune_completa, args.model, RESULTS_DIR
    )
    metrici = calculeaza_si_afiseaza(rezultate, versiune_completa, args.model)

    output_file = os.path.join(RESULTS_DIR, f"rezumat_local_{args.model}_V3{sufix_varianta}_{descriere_set}.json")
    raport_file = os.path.join(RESULTS_DIR, f"rezumat_local_{args.model}_V3{sufix_varianta}_{descriere_set}_raport.txt")

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
