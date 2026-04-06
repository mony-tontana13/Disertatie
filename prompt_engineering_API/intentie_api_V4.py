"""
Prompt V4 - Detectare Intentie - Modele API
GPT-4.1-mini, Gemini-2.5-flash, Aya-Expanse-8b

Utilizare:
    # Subset mic (10 conversatii)
    python3 intentie_api_V4.py

    # Subset personalizat
    python3 intentie_api_V4.py --n_per_domeniu 4

    # Tot setul
    python3 intentie_api_V4.py --tot_setul
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

VERSIUNE = "V4"
RESULTS_DIR = "./rezultate_prompt_engineering"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_prompt(dialog, domeniu):
    """V4: Role + Structured + Constrained + Few-shot (Varianta B - dialoguri complete)."""
    intentii = INTENTII_DOMENII.get(domeniu, [])
    intentii_str = ", ".join(intentii)
    exemple = EXEMPLE_FEWSHOT_LUNGI.get(domeniu, [])

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

def get_prompt2(dialog, domeniu):
    """V4.3: Role + Structured + Constrained + Few-shot cu dialoguri complete."""
    intentii = INTENTII_DOMENII.get(domeniu, [])
    intentii_str = ", ".join(intentii)

    # Exemple cu dialog mai lung, mai aproape de conversatiile reale
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
    parser = argparse.ArgumentParser(description="Prompt V4 - Detectare Intentie - Modele API")
    parser.add_argument("--n_per_domeniu", type=int, default=2,
                        help="Conversatii per domeniu pentru subset (default: 2 = 10 total)")
    parser.add_argument("--tot_setul", action="store_true",
                        help="Ruleaza pe toate cele 100 de conversatii")
    args = parser.parse_args()

    print(f"\n=== PROMPT V4 — DETECTARE INTENTIE — MODELE API ===")

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
    output_file = os.path.join(RESULTS_DIR, f"intentie_api_V4_{descriere_set}.json")
    raport_file = os.path.join(RESULTS_DIR, f"intentie_api_V4_{descriere_set}_raport.txt")

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
