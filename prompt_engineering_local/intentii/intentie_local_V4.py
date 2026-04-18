"""
Prompt V4 - Detectare Intentie - Modele Locale
Zero-shot + Role + Constrained + Structured + Few-shot — exemple complete cu dialoguri operator-client.

Utilizare:
    python3 intentie_local_V4.py --model romistral
    python3 intentie_local_V4.py --model romistral --varianta 2
    python3 intentie_local_V4.py --model romistral --n_per_domeniu 4
    python3 intentie_local_V4.py --model romistral --tot_setul
    python3 intentie_local_V4.py --model romistral --varianta 2 --tot_setul
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

VERSIUNE = "V4"
RESULTS_DIR = "./rezultate_prompt_engineering/intentii_local"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_prompt(dialog, domeniu):
    """V4 varianta 1: Role + Constrained + Structured + Few-shot — exemple scurte (o replica client)."""
    intentii = INTENTII_DOMENII.get(domeniu, [])
    intentii_str = ", ".join(intentii)
    exemple = EXEMPLE_FEWSHOT.get(domeniu, [])
    exemple_text = ""
    for dialog_ex, intentie_ex in exemple:
        exemple_text += (
            "CONVERSATIE: " + dialog_ex + "\n"
            "INTENTIE IDENTIFICATA: " + intentie_ex + "\n\n"
        )
    return (
        "Esti un agent specializat in analiza conversatiilor telefonice din call-center-uri "
        "din domeniul " + domeniu + ". Rolul tau este sa identifici intentia clientului.\n\n"
        "SARCINA: Identifica intentia sau intentiile clientului din conversatia de mai jos.\n\n"
        "REGULI:\n"
        "1. Include DOAR ce a cerut sau intrebat clientul — ignora actiunile operatorului\n"
        "2. Alege EXCLUSIV din lista de intentii furnizata\n"
        "3. Returneaza MAXIM doua intentii, separate prin virgula\n"
        "4. Prima intentie trebuie sa fie cea principala, cu care a sunat clientul\n\n"
        "INTENTII DISPONIBILE: " + intentii_str + "\n\n"
        "EXEMPLE:\n" + exemple_text +
        "INTENTIE IDENTIFICATA:"
    )

def get_prompt2(dialog, domeniu):
    """V4 varianta 2: Role + Constrained + Structured + Few-shot — exemple lungi cu dialoguri complete."""
    intentii = INTENTII_DOMENII.get(domeniu, [])
    intentii_str = ", ".join(intentii)
    exemple_lungi = {
        "banking": [
            ("OPERATOR: Buna ziua, cu ce va pot ajuta?\nCLIENT: Buna ziua, am o problema cu cardul meu, l-am pierdut ieri si nu stiu ce sa fac.\nOPERATOR: Inteleg, va ajut imediat.", "card_pierdut"),
            ("OPERATOR: Serviciul clienti, va ascult.\nCLIENT: Buna ziua, as vrea sa stiu de ce rata mea la credit a crescut luna aceasta.\nOPERATOR: Va verific contul.", "problema_credit"),
        ],
        "medicina": [
            ("OPERATOR: Clinica MedCare, buna ziua.\nCLIENT: Buna ziua, am facut analize saptamana trecuta si vreau sa aflu rezultatele.\nOPERATOR: Va caut in sistem.", "rezultate_analize"),
            ("OPERATOR: Cu ce va pot ajuta?\nCLIENT: Am o programare maine dar nu mai pot veni, as vrea sa o anulez.\nOPERATOR: Sigur, cum va numiti?", "anulare_programare"),
        ],
        "retail": [
            ("OPERATOR: Serviciul clienti, buna ziua.\nCLIENT: Buna ziua, am primit o comanda dar produsele nu sunt cele pe care le-am comandat.\nOPERATOR: Imi pare rau, va ajut.", "comanda_gresita"),
            ("OPERATOR: Cu ce va pot ajuta?\nCLIENT: Pachetul meu nu a ajuns si a trecut termenul de livrare de trei zile.\nOPERATOR: Va verific comanda.", "problema_livrare"),
        ],
        "telecom": [
            ("OPERATOR: Buna ziua, serviciul clienti.\nCLIENT: Buna ziua, am vrut sa imi port numarul la voi dar cererea a fost respinsa.\nOPERATOR: Va verific situatia.", "portare_esuata"),
            ("OPERATOR: Cu ce va pot ajuta?\nCLIENT: As vrea sa schimb abonamentul meu dar nu reusesc sa fac asta din aplicatie.\nOPERATOR: Va ajut eu.", "problema_modificare_abonament"),
        ],
        "servicii_publice": [
            ("OPERATOR: Primaria, buna ziua.\nCLIENT: Buna ziua, dosarul meu a fost respins si nu inteleg de ce.\nOPERATOR: Va caut dosarul.", "dosar_respins"),
            ("OPERATOR: Cu ce va pot ajuta?\nCLIENT: As vrea sa fac o programare la ghiseu pentru un act de identitate.\nOPERATOR: Va programez.", "programare_ghiseu"),
        ],
    }
    exemple = exemple_lungi.get(domeniu, [])
    exemple_text = ""
    for dialog_ex, intentie_ex in exemple:
        exemple_text += (
            "CONVERSATIE:\n" + dialog_ex + "\n"
            "INTENTIE IDENTIFICATA: " + intentie_ex + "\n\n"
        )
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
        "EXEMPLE:\n" + exemple_text +
        "INTENTIE IDENTIFICATA:"
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prompt V4 - Detectare Intentie - Modele Locale")
    parser.add_argument("--model", required=True, choices=["romistral", "rogemma"])
    parser.add_argument("--varianta", type=int, default=1, choices=[1, 2],
                        help="Varianta de prompt: 1 sau 2 (default: 1)")
    parser.add_argument("--n_per_domeniu", type=int, default=2,
                        help="Conversatii per domeniu pentru subset (default: 2 = 10 total)")
    parser.add_argument("--tot_setul", action="store_true",
                        help="Ruleaza pe toate cele 100 de conversatii")
    args = parser.parse_args()

    func_prompt = get_prompt if args.varianta == 1 else get_prompt2
    sufix_varianta = f"_v{args.varianta}"

    print(f"\n=== PROMPT V4 varianta {args.varianta} — DETECTARE INTENTIE — MODELE LOCALE ===")
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

    versiune_completa = f"V4{sufix_varianta}"
    rezultate = ruleaza_evaluare(
        conversatii, tokenizer, model, device,
        func_prompt, versiune_completa, args.model, RESULTS_DIR
    )
    metrici = calculeaza_si_afiseaza(rezultate, versiune_completa, args.model)

    output_file = os.path.join(RESULTS_DIR, f"intentie_local_{args.model}_V4{sufix_varianta}_{descriere_set}.json")
    raport_file = os.path.join(RESULTS_DIR, f"intentie_local_{args.model}_V4{sufix_varianta}_{descriere_set}_raport.txt")

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