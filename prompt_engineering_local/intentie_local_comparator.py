"""
Comparator rezultate - Detectare Intentie - Modele Locale
Citeste fisierele de rezultate generate de V1-V4 si afiseaza un tabel comparativ.

Utilizare:
    python3 intentie_local_comparator.py --model romistral
    python3 intentie_local_comparator.py --model rogemma
    python3 intentie_local_comparator.py --model romistral --set_date tot_setul
"""
import json
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR = "./rezultate_prompt_engineering"
VERSIUNI = ["V1", "V2", "V3", "V4"]


def incarca_rezultate(model, versiune, set_date):
    """Cauta fisierul de rezultate pentru o versiune data."""
    # Cauta fisierul care contine versiunea si setul de date
    pattern = f"intentie_local_{model}_{versiune}_{set_date}"
    for fisier in os.listdir(RESULTS_DIR):
        if fisier.startswith(pattern) and fisier.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fisier), encoding="utf-8") as f:
                return json.load(f)
    
    # Alternativ cauta in fisierele de progres
    progress = os.path.join(RESULTS_DIR, f"intentie_{model}_{versiune}_progress.json")
    if os.path.exists(progress):
        with open(progress, encoding="utf-8") as f:
            rezultate = json.load(f)
        return {"rezultate_detaliate": rezultate, "versiune": versiune, "model": model}
    
    return None


def calculeaza_metrici(rezultate):
    gold = [r["intentie_gold"] for r in rezultate]
    pred = [r["intentie_pred"] for r in rezultate]
    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred, average="macro", zero_division=0)
    latenta = sum(r["latenta"] for r in rezultate) / len(rezultate)
    erori = sum(1 for r in rezultate if not r["corecta"])
    return acc, f1, latenta, erori, len(rezultate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["romistral", "rogemma"])
    parser.add_argument("--set_date", default="subset_2_per_domeniu",
                        help="Tipul setului de date folosit (default: subset_2_per_domeniu)")
    parser.add_argument("--toate_modelele", action="store_true",
                        help="Compara atat RoMistral cat si RoGemma")
    args = parser.parse_args()

    modele = ["romistral", "rogemma"] if args.toate_modelele else [args.model]

    print(f"\n=== COMPARATOR PROMPT ENGINEERING — DETECTARE INTENTIE ===")
    print(f"Set de date: {args.set_date}")
    print(f"{'='*70}")

    for model in modele:
        print(f"\nModel: {model.upper()}")
        print(f"  {'Versiune':<10} {'Accuracy':>10} {'F1 Macro':>10} {'Latenta':>10} {'Erori':>8} {'N':>5}")
        print(f"  {'-'*55}")

        metrici_toate = []
        for versiune in VERSIUNI:
            data = incarca_rezultate(model, versiune, args.set_date)
            if data is None:
                print(f"  {versiune:<10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>5}")
                continue

            rezultate = data.get("rezultate_detaliate", [])
            if not rezultate:
                print(f"  {versiune:<10} {'Gol':>10}")
                continue

            acc, f1, latenta, erori, n = calculeaza_metrici(rezultate)
            metrici_toate.append((versiune, acc, f1, latenta, erori, n))
            print(f"  {versiune:<10} {acc:>10.2%} {f1:>10.3f} {latenta:>9.1f}s {erori:>7}/{n:<3}")

        if metrici_toate:
            best_acc = max(metrici_toate, key=lambda x: x[1])
            best_f1 = max(metrici_toate, key=lambda x: x[2])
            print(f"\n  Cel mai bun Accuracy: {best_acc[0]} ({best_acc[1]:.2%})")
            print(f"  Cel mai bun F1:       {best_f1[0]} ({best_f1[2]:.3f})")

        # Analiza erorilor per versiune
        print(f"\n  Analiza erori per domeniu:")
        for versiune in VERSIUNI:
            data = incarca_rezultate(model, versiune, args.set_date)
            if data is None:
                continue
            rezultate = data.get("rezultate_detaliate", [])
            erori_dom = {}
            for r in rezultate:
                if not r["corecta"]:
                    dom = r["domeniu"]
                    erori_dom[dom] = erori_dom.get(dom, 0) + 1
            if erori_dom:
                print(f"  {versiune}: {dict(erori_dom)}")


if __name__ == "__main__":
    main()