"""
Comparator rezultate - Generare Rezumat - Modele API

Utilizare:
    python3 rezumat_api_comparator.py
    python3 rezumat_api_comparator.py --set_date tot_setul
"""
import json
import os
import argparse

RESULTS_DIR = "./rezultate_prompt_engineering/rezumat_api"
VERSIUNI_VARIANTE = ["V1_v1", "V1_v2", "V2_v1", "V2_v2", "V3_v1", "V3_v2"]
MODELE = ["GPT-4.1-mini", "Gemini-2.5-flash", "command-r7b-12-2024"]


def incarca_rezultate(versiune_varianta, set_date):
    pattern = f"rezumat_api_{versiune_varianta}_{set_date}"
    if os.path.isdir(RESULTS_DIR):
        for fisier in os.listdir(RESULTS_DIR):
            if fisier.startswith(pattern) and fisier.endswith(".json"):
                with open(os.path.join(RESULTS_DIR, fisier), encoding="utf-8") as f:
                    return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_date", default="subset_2_per_domeniu")
    args = parser.parse_args()

    print(f"\n=== COMPARATOR REZUMAT — MODELE API ===")
    print(f"Set de date: {args.set_date}")
    print(f"{'='*80}")

    print(f"\n  {'Model':<22} {'Versiune':<10} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BERT F1':>8} {'TTFT':>8} {'Cuv':>6}")
    print(f"  {'-'*82}")

    toate_metrici = {}
    for vv in VERSIUNI_VARIANTE:
        data = incarca_rezultate(vv, args.set_date)
        if data is None:
            continue
        for m in data.get("metrici", []):
            toate_metrici[(m["model"], vv)] = m
            print(f"  {m['model']:<22} {vv:<10} {m.get('rouge1',0):>8.4f} {m.get('rouge2',0):>8.4f} {m.get('rougeL',0):>8.4f} {m.get('bertscore_f1',0):>8.4f} {m.get('ttft_medie',0):>7.3f}s {m.get('nr_cuvinte_medii',0):>6.1f}")

    print(f"\n  Cel mai bun prompt per model (dupa ROUGE-L):")
    for model in MODELE:
        rez_model = [(vv, toate_metrici[(model, vv)]) for vv in VERSIUNI_VARIANTE if (model, vv) in toate_metrici]
        if rez_model:
            best = max(rez_model, key=lambda x: x[1].get("rougeL", 0))
            print(f"    {model:<22}: {best[0]} — ROUGE-L={best[1].get('rougeL',0):.4f} | BERT={best[1].get('bertscore_f1',0):.4f}")


if __name__ == "__main__":
    main()
