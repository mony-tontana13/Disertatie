"""
Comparator rezultate - Detectare Intentie - Modele API
Citeste fisierele de rezultate si afiseaza un tabel comparativ complet.

Utilizare:
    python3 intentie_api_comparator.py
    python3 intentie_api_comparator.py --set_date tot_setul
"""
import json
import os
import argparse

RESULTS_DIR = "./rezultate_prompt_engineering"
VERSIUNI = ["V1", "V2", "V3", "V4"]
MODELE = ["GPT-4.1-mini", "Gemini-2.5-flash", "Aya-Expanse-8b"]


def incarca_rezultate(versiune, set_date):
    pattern = f"intentie_api_{versiune}_{set_date}.json"
    filepath = os.path.join(RESULTS_DIR, pattern)
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_date", default="subset_2_per_domeniu")
    args = parser.parse_args()

    print(f"\n=== COMPARATOR PROMPT ENGINEERING — DETECTARE INTENTIE — MODELE API ===")
    print(f"Set de date: {args.set_date}")

    # Tabel complet: model x versiune
    print(f"\n{'='*80}")
    print(f"  {'Model':<22} {'Versiune':<8} {'Accuracy':>10} {'F1':>8} {'TTFT':>10} {'Latenta':>10}")
    print(f"  {'-'*70}")

    toate_metrici = {}
    for versiune in VERSIUNI:
        data = incarca_rezultate(versiune, args.set_date)
        if data is None:
            print(f"  [Lipseste: {versiune}]")
            continue
        for m in data.get("metrici", []):
            key = (m["model"], versiune)
            toate_metrici[key] = m
            print(f"  {m['model']:<22} {versiune:<8} {m['accuracy']:>10.2%} {m['f1']:>8.3f} {m['ttft_medie']:>9.3f}s {m['latenta_medie']:>9.3f}s")

    # Cel mai bun prompt per model
    print(f"\n{'='*80}")
    print("CEL MAI BUN PROMPT PER MODEL (dupa Accuracy)")
    print(f"{'='*80}")
    for model in MODELE:
        rezultate_model = [(v, toate_metrici[(model, v)]) for v in VERSIUNI if (model, v) in toate_metrici]
        if rezultate_model:
            best = max(rezultate_model, key=lambda x: x[1]["accuracy"])
            print(f"  {model:<22}: {best[0]} — Accuracy={best[1]['accuracy']:.2%} | F1={best[1]['f1']:.3f}")

    # Cel mai bun model per versiune
    print(f"\n{'='*80}")
    print("CEL MAI BUN MODEL PER VERSIUNE (dupa Accuracy)")
    print(f"{'='*80}")
    for versiune in VERSIUNI:
        rezultate_v = [(m, toate_metrici[(m, versiune)]) for m in MODELE if (m, versiune) in toate_metrici]
        if rezultate_v:
            best = max(rezultate_v, key=lambda x: x[1]["accuracy"])
            print(f"  {versiune:<8}: {best[0]:<22} — Accuracy={best[1]['accuracy']:.2%} | F1={best[1]['f1']:.3f}")

    # Analiza erori per domeniu pentru cea mai buna versiune
    print(f"\n{'='*80}")
    print("ANALIZA ERORI PER DOMENIU — V4 (versiunea finala)")
    print(f"{'='*80}")
    data_v4 = incarca_rezultate("V4", args.set_date)
    if data_v4:
        for m in data_v4.get("metrici", []):
            print(f"\n  {m['model']}:")
            for domeniu, acc in m.get("per_domeniu", {}).items():
                print(f"    {domeniu:<22}: {acc:.0%}")


if __name__ == "__main__":
    main()
