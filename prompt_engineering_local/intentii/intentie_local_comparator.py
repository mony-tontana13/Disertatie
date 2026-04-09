"""
Comparator rezultate - Detectare Intentie - Modele Locale
Utilizare:
    python3 intentie_local_comparator.py --model romistral
    python3 intentie_local_comparator.py --model rogemma
    python3 intentie_local_comparator.py --toate_modelele
"""
import json
import os
import argparse
import unicodedata
from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR = "./rezultate_prompt_engineering/intentii_local"

SET_DATE_PER_VERSIUNE = {
    "V1": "tot_setul_tot",
    "V2": "tot_setul",
    "V3": "subset_10_per_domeniu",
    "V4": "subset_10_per_domeniu",
}


def norm(t):
    t = unicodedata.normalize("NFD", t)
    return "".join(c for c in t if unicodedata.category(c) != "Mn").lower()


def incarca_rezultate(model, versiune, set_date):
    pattern = f"intentie_local_{model}_{versiune}_{set_date}"
    if os.path.isdir(RESULTS_DIR):
        for fisier in os.listdir(RESULTS_DIR):
            if fisier.startswith(pattern) and fisier.endswith(".json"):
                with open(os.path.join(RESULTS_DIR, fisier), encoding="utf-8") as f:
                    return json.load(f)
    return None


def calculeaza_metrici(rezultate):
    gold = [r["intentie_gold"] for r in rezultate]
    pred = [r["intentie_pred"] for r in rezultate]
    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred, average="macro", zero_division=0)
    latenta = sum(r["latenta"] for r in rezultate) / len(rezultate)
    erori = sum(1 for r in rezultate if not r["corecta"])
    fara_ordine = sum(1 for r in rezultate if norm(r["intentie_gold"]) in norm(r["raspuns_brut"]))
    return acc, f1, latenta, erori, fara_ordine, len(rezultate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["romistral", "rogemma"])
    parser.add_argument("--toate_modelele", action="store_true")
    args = parser.parse_args()

    if not args.model and not args.toate_modelele:
        parser.error("Specifica --model sau --toate_modelele")

    modele = ["romistral", "rogemma"] if args.toate_modelele else [args.model]

    print(f"\n=== COMPARATOR PROMPT ENGINEERING — DETECTARE INTENTIE — MODELE LOCALE ===")
    print(f"\n  Seturi de date folosite:")
    for v, s in SET_DATE_PER_VERSIUNE.items():
        print(f"    {v}: {s}")
    print(f"{'='*85}")

    for model in modele:
        print(f"\nModel: {model.upper()}")
        print(f"  {'Versiune':<8} {'Set date':<22} {'Acc. strict':>12} {'Acc. fara ord':>14} {'F1':>8} {'Latenta':>10} {'N':>5}")
        print(f"  {'-'*82}")

        metrici_toate = []
        for versiune in ["V1", "V2", "V3", "V4"]:
            set_date = SET_DATE_PER_VERSIUNE[versiune]
            data = incarca_rezultate(model, versiune, set_date)

            if data is None:
                print(f"  {versiune:<8} {set_date:<22} {'N/A':>12}")
                continue

            rezultate = data.get("rezultate_detaliate", [])
            if not rezultate:
                print(f"  {versiune:<8} {set_date:<22} {'Gol':>12}")
                continue

            acc, f1, latenta, erori, fara_ordine, n = calculeaza_metrici(rezultate)
            acc_fara_ordine = fara_ordine / n
            metrici_toate.append((versiune, acc, f1, latenta, erori, fara_ordine, n))
            print(f"  {versiune:<8} {set_date:<22} {acc:>12.2%} {acc_fara_ordine:>14.2%} {f1:>8.3f} {latenta:>9.1f}s {n:>5}")

        if metrici_toate:
            best_acc = max(metrici_toate, key=lambda x: x[1])
            best_f1 = max(metrici_toate, key=lambda x: x[2])
            best_fara_ord = max(metrici_toate, key=lambda x: x[5]/x[6])
            print(f"\n  Cel mai bun Accuracy strict:  {best_acc[0]} ({best_acc[1]:.2%})")
            print(f"  Cel mai bun F1 Macro:         {best_f1[0]} ({best_f1[2]:.3f})")
            print(f"  Cel mai bun fara ordine:      {best_fara_ord[0]} ({best_fara_ord[5]/best_fara_ord[6]:.2%})")

        # Tabel per domeniu
        print(f"\n  Accuracy per domeniu:")
        domenii = ["banking", "medicina", "retail", "telecom", "servicii_publice"]
        print(f"  {'Domeniu':<24}", end="")
        for versiune in ["V1", "V2", "V3", "V4"]:
            print(f" {versiune:>8}", end="")
        print()
        print(f"  {'-'*58}")

        for domeniu in domenii:
            print(f"  {domeniu:<24}", end="")
            for versiune in ["V1", "V2", "V3", "V4"]:
                set_date = SET_DATE_PER_VERSIUNE[versiune]
                data = incarca_rezultate(model, versiune, set_date)
                if data is None:
                    print(f" {'N/A':>8}", end="")
                    continue
                rez_d = [r for r in data.get("rezultate_detaliate", []) if r["domeniu"] == domeniu]
                if rez_d:
                    acc_d = sum(1 for r in rez_d if r["corecta"]) / len(rez_d)
                    print(f" {acc_d:>8.0%}", end="")
                else:
                    print(f" {'N/A':>8}", end="")
            print()
        print()


if __name__ == "__main__":
    main()