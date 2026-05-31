"""
Comparator rezultate - Estimare Satisfactie - Modele Locale

Utilizare:
    python3 satisfactie_local_comparator.py --model romistral
    python3 satisfactie_local_comparator.py --toate_modelele
"""
import json
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter

RESULTS_DIR = "./rezultate_prompt_engineering/satisfactie_local"
GRAFICE_DIR = "./rezultate_prompt_engineering/grafice_satisfactie_local"
VERSIUNI_VARIANTE = ["V1_v1", "V1_v2", "V2_v1", "V2_v2", "V3_v1", "V3_v2"]
CLASE = ["pozitiv", "neutru", "negativ"]


def incarca_rezultate(model, versiune_varianta, set_date):
    pattern = f"satisfactie_local_{model}_{versiune_varianta}_{set_date}"
    if os.path.isdir(RESULTS_DIR):
        for fisier in os.listdir(RESULTS_DIR):
            if fisier.startswith(pattern) and fisier.endswith(".json"):
                with open(os.path.join(RESULTS_DIR, fisier), encoding="utf-8") as f:
                    return json.load(f)
    return None


def calculeaza_metrici(rezultate):
    gold = [r["satisfactie_gold"] for r in rezultate]
    pred = [r["satisfactie_pred"] for r in rezultate]
    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred, average="macro", zero_division=0, labels=CLASE)
    latenta = sum(r["latenta"] for r in rezultate) / len(rezultate)
    erori = sum(1 for r in rezultate if not r["corecta"])
    return acc, f1, latenta, erori, len(rezultate)


def genereaza_matrice_satisfactie_local(rezultate, model, versiune_varianta, target_dir):
    """Genereaza matricea de confuzie pentru analiza satisfactiei (modele locale)."""
    df = pd.DataFrame(rezultate)
    if df.empty:
        return

    # Toate clasele posibile + clasa de fallback
    etichete_valide = CLASE + ["necunoscut"]
    
    cm = confusion_matrix(df["satisfactie_gold"], df["satisfactie_pred"], labels=etichete_valide)
    etichete_curate = [e.capitalize() for e in etichete_valide]
    cm_df = pd.DataFrame(cm, index=etichete_curate, columns=etichete_curate)

    plt.figure(figsize=(7, 5.5))
    sns.set_theme(style="white")
    sns.heatmap(
        cm_df, annot=True, fmt="d", cmap="Blues", cbar=True,
        linewidths=0.5, linecolor="#d3d3d3", annot_kws={"size": 12, "weight": "bold"}
    )

    plt.title(
        f"Matrice de Confuzie Satisfacție — {model.upper()} ({versiune_varianta})",
        fontsize=12, fontweight="bold", pad=15
    )
    plt.ylabel("Nivel Real (Ground Truth)", fontsize=11, fontweight="bold")
    plt.xlabel("Nivel Prezistă (Predicted)", fontsize=11, fontweight="bold")
    plt.xticks(fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    # Organizare in foldere: target_dir / MODEL
    output_subdir = os.path.join(target_dir, model.lower())
    os.makedirs(output_subdir, exist_ok=True)
    
    cale_salvare = os.path.join(output_subdir, f"matrice_satisfactie_{versiune_varianta}.png")
    plt.savefig(cale_salvare, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["romistral", "rogemma"])
    parser.add_argument("--toate_modelele", action="store_true")
    parser.add_argument("--set_date", default="subset_10_per_domeniu")
    args = parser.parse_args()

    if not args.model and not args.toate_modelele:
        parser.error("Specifica --model sau --toate_modelele")

    modele = ["romistral", "rogemma"] if args.toate_modelele else [args.model]

    print(f"\n=== COMPARATOR SATISFACTIE — MODELE LOCALE ===")
    print(f"Set de date: {args.set_date}")
    print(f"{'='*75}")

    for model in modele:
        print(f"\nModel: {model.upper()}")
        print(f"  {'Versiune':<12} {'Accuracy':>10} {'F1 Macro':>10} {'Latenta':>10} {'Erori':>8} {'N':>5}")
        print(f"  {'-'*55}")

        metrici_toate = []
        for vv in VERSIUNI_VARIANTE:
            data = incarca_rezultate(model, vv, args.set_date)
            if data is None:
                print(f"  {vv:<12} {'N/A':>10}")
                continue
            rezultate = data.get("rezultate_detaliate", [])
            if not rezultate:
                continue
            acc, f1, latenta, erori, n = calculeaza_metrici(rezultate)
            metrici_toate.append((vv, acc, f1, latenta, erori, n))
            print(f"  {vv:<12} {acc:>10.2%} {f1:>10.3f} {latenta:>9.1f}s {erori:>7}/{n:<3}")

            # Generare automata matrice de confuzie
            genereaza_matrice_satisfactie_local(rezultate, model, vv, GRAFICE_DIR)

        if metrici_toate:
            best_f1 = max(metrici_toate, key=lambda x: x[2])
            best_acc = max(metrici_toate, key=lambda x: x[1])
            print(f"\n  Cel mai bun F1:       {best_f1[0]} ({best_f1[2]:.3f})")
            print(f"  Cel mai buna Accuracy: {best_acc[0]} ({best_acc[1]:.2%})")

        # Per clasa
        print(f"\n  Accuracy per clasa:")
        print(f"  {'Clasa':<12}", end="")
        for vv in VERSIUNI_VARIANTE:
            print(f" {vv:>8}", end="")
        print()
        print(f"  {'-'*65}")

        for clasa in CLASE:
            print(f"  {clasa:<12}", end="")
            for vv in VERSIUNI_VARIANTE:
                data = incarca_rezultate(model, vv, args.set_date)
                if data is None:
                    print(f" {'N/A':>8}", end="")
                    continue
                rez_c = [r for r in data.get("rezultate_detaliate", []) if r["satisfactie_gold"] == clasa]
                if rez_c:
                    acc_c = sum(1 for r in rez_c if r["corecta"]) / len(rez_c)
                    print(f" {acc_c:>8.0%}", end="")
                else:
                    print(f" {'N/A':>8}", end="")
            print()

        # Distributie predictii per varianta
        print(f"\n  Distributie predictii:")
        for vv in VERSIUNI_VARIANTE:
            data = incarca_rezultate(model, vv, args.set_date)
            if data is None:
                continue
            rezultate = data.get("rezultate_detaliate", [])
            pred = [r["satisfactie_pred"] for r in rezultate]
            dist = dict(Counter(pred))
            print(f"    {vv}: {dist}")
        print()


if __name__ == "__main__":
    main()