"""
Comparator rezultate - Generare Rezumat - Modele Locale

Utilizare:
    python3 rezumat_local_comparator.py --model romistral
    python3 rezumat_local_comparator.py --toate_modelele
"""
import json
import os
import argparse

RESULTS_DIR = "./rezultate_prompt_engineering/rezumat_local"
VERSIUNI_VARIANTE = ["V1_v1", "V1_v2", "V2_v1", "V2_v2", "V3_v1", "V3_v2"]


def incarca_rezultate(model, versiune_varianta, set_date):
    pattern = f"rezumat_local_{model}_{versiune_varianta}_{set_date}"
    if os.path.isdir(RESULTS_DIR):
        for fisier in os.listdir(RESULTS_DIR):
            if fisier.startswith(pattern) and fisier.endswith(".json"):
                with open(os.path.join(RESULTS_DIR, fisier), encoding="utf-8") as f:
                    return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["romistral", "rogemma"])
    parser.add_argument("--toate_modelele", action="store_true")
    parser.add_argument("--set_date", default="subset_2_per_domeniu")
    args = parser.parse_args()

    if not args.model and not args.toate_modelele:
        parser.error("Specifica --model sau --toate_modelele")

    modele = ["romistral", "rogemma"] if args.toate_modelele else [args.model]

    print(f"\n=== COMPARATOR REZUMAT — MODELE LOCALE ===")
    print(f"Set de date: {args.set_date}")
    print(f"{'='*80}")

    for model in modele:
        print(f"\nModel: {model.upper()}")
        print(f"  {'Versiune':<12} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BERT F1':>8} {'Latenta':>10} {'Cuv':>6} {'Limite':>8}")
        print(f"  {'-'*72}")

        metrici_toate = []
        for vv in VERSIUNI_VARIANTE:
            data = incarca_rezultate(model, vv, args.set_date)
            if data is None:
                print(f"  {vv:<12} {'N/A':>8}")
                continue
            m = data.get("metrici", {})
            if not m:
                continue
            metrici_toate.append((vv, m))
            n = m.get("nr_conversatii", 0)
            in_limite = m.get("in_limite", 0)
            print(f"  {vv:<12} {m.get('rouge1',0):>8.4f} {m.get('rouge2',0):>8.4f} {m.get('rougeL',0):>8.4f} {m.get('bertscore_f1',0):>8.4f} {m.get('latenta',0):>9.1f}s {m.get('nr_cuvinte_medii',0):>6.1f} {in_limite:>4}/{n:<3}")

        if metrici_toate:
            best_r1 = max(metrici_toate, key=lambda x: x[1].get("rouge1", 0))
            best_bert = max(metrici_toate, key=lambda x: x[1].get("bertscore_f1", 0))
            print(f"\n  Cel mai bun ROUGE-1:      {best_r1[0]} ({best_r1[1].get('rouge1',0):.4f})")
            print(f"  Cel mai bun BERTScore F1: {best_bert[0]} ({best_bert[1].get('bertscore_f1',0):.4f})")

        # Per tip rezumat
        print(f"\n  ROUGE-1 per tip rezumat:")
        print(f"  {'Versiune':<12} {'SCURT':>8} {'MEDIU':>8} {'LUNG':>8}")
        print(f"  {'-'*40}")
        for vv in VERSIUNI_VARIANTE:
            data = incarca_rezultate(model, vv, args.set_date)
            if data is None:
                continue
            rezultate = data.get("rezultate_detaliate", [])
            print(f"  {vv:<12}", end="")
            for tip in ["SCURT", "MEDIU", "LUNG"]:
                rez_t = [r for r in rezultate if r["tip_rezumat"] == tip]
                if rez_t:
                    try:
                        from rouge_score import rouge_scorer
                        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
                        r1_vals = [scorer.score(r["rezumat_gold"], r["rezumat_pred"])["rouge1"].fmeasure
                                   for r in rez_t if r["rezumat_gold"] and r["rezumat_pred"]]
                        avg = sum(r1_vals)/len(r1_vals) if r1_vals else 0
                        print(f" {avg:>8.4f}", end="")
                    except:
                        print(f" {'N/A':>8}", end="")
                else:
                    print(f" {'—':>8}", end="")
            print()
        print()


if __name__ == "__main__":
    main()
