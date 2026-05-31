"""
Comparator rezultate - Estimare Satisfactie - Modele API

Utilizare:
    python3 satisfactie_api_comparator.py
    python3 satisfactie_api_comparator.py --set_date tot_setul
"""
import json
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter

RESULTS_DIR = "./rezultate_prompt_engineering/satisfactie_api"
GRAFICE_DIR = "./rezultate_prompt_engineering/grafice_satisfactie_api"
VERSIUNI_VARIANTE = ["V1_v1", "V1_v2", "V2_v1", "V2_v2", "V3_v1", "V3_v2"]
MODELE = ["GPT-4.1-mini", "Gemini-2.5-flash", "command-r7b-12-2024"]
CLASE = ["pozitiv", "neutru", "negativ"]


def incarca_rezultate(versiune_varianta, set_date):
    pattern = f"satisfactie_api_{versiune_varianta}_{set_date}"
    if os.path.isdir(RESULTS_DIR):
        for fisier in os.listdir(RESULTS_DIR):
            if fisier.startswith(pattern) and fisier.endswith(".json"):
                with open(os.path.join(RESULTS_DIR, fisier), encoding="utf-8") as f:
                    return json.load(f)
    return None


def genereaza_matrice_satisfactie_api(data_json, versiune_varianta, target_dir):
    """Parcurge rezultatele detaliate dintr-un fisier API si genereaza matricea de confuzie separata per model."""
    rezultate_toate = data_json.get("rezultate_detaliate", [])
    if not rezultate_toate:
        return

    df_complet = pd.DataFrame(rezultate_toate)
    etichete_valide = CLASE + ["necunoscut"]
    etichete_curate = [e.capitalize() for e in etichete_valide]

    for model in df_complet["model"].unique():
        df_model = df_complet[df_complet["model"] == model]

        cm = confusion_matrix(df_model["satisfactie_gold"], df_model["satisfactie_pred"], labels=etichete_valide)
        cm_df = pd.DataFrame(cm, index=etichete_curate, columns=etichete_curate)

        plt.figure(figsize=(7, 5.5))
        sns.set_theme(style="white")
        
        # Paleta Purples pentru modelele API
        sns.heatmap(
            cm_df, annot=True, fmt="d", cmap="Purples", cbar=True,
            linewidths=0.5, linecolor="#d3d3d3", annot_kws={"size": 12, "weight": "bold"}
        )

        model_safe_name = model.replace("/", "_").replace(":", "_")

        plt.title(
            f"Matrice de Confuzie Satisfacție — {model.upper()} ({versiune_varianta})",
            fontsize=12, fontweight="bold", pad=15
        )
        plt.ylabel("Nivel Real (Ground Truth)", fontsize=11, fontweight="bold")
        plt.xlabel("Nivel Prezistă (Predicted)", fontsize=11, fontweight="bold")
        plt.xticks(fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        # Structura directoarelor: target_dir / MODEL / versiune
        output_subdir = os.path.join(target_dir, model_safe_name, versiune_varianta)
        os.makedirs(output_subdir, exist_ok=True)
        
        cale_salvare = os.path.join(output_subdir, "matrice_satisfactie.png")
        plt.savefig(cale_salvare, dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_date", default="subset_2_per_domeniu")
    args = parser.parse_args()

    print(f"\n=== COMPARATOR SATISFACTIE — MODELE API ===")
    print(f"Set de date: {args.set_date}")
    print(f"{'='*80}")

    # Tabel complet model x versiune
    print(f"\n  {'Model':<22} {'Versiune':<10} {'Accuracy':>10} {'F1':>8} {'TTFT':>10}")
    print(f"  {'-'*62}")

    toate_metrici = {}
    for vv in VERSIUNI_VARIANTE:
        data = incarca_rezultate(vv, args.set_date)
        if data is None:
            continue
            
        # Generare automata matrice de confuzie pentru versiunea curenta (per model)
        genereaza_matrice_satisfactie_api(data, vv, GRAFICE_DIR)
        
        for m in data.get("metrici", []):
            key = (m["model"], vv)
            toate_metrici[key] = m
            print(f"  {m['model']:<22} {vv:<10} {m['accuracy']:>10.2%} {m['f1']:>8.3f} {m['ttft_medie']:>9.3f}s")

    # Cel mai bun per model
    print(f"\n  Cel mai bun prompt per model (dupa F1):")
    for model in MODELE:
        rez_model = [(vv, toate_metrici[(model, vv)]) for vv in VERSIUNI_VARIANTE if (model, vv) in toate_metrici]
        if rez_model:
            best = max(rez_model, key=lambda x: x[1]["f1"])
            print(f"    {model:<22}: {best[0]} — F1={best[1]['f1']:.3f} | Acc={best[1]['accuracy']:.2%}")

    # Per clasa pentru fiecare versiune
    print(f"\n  Accuracy per clasa:")
    print(f"  {'Model':<22} {'Versiune':<10}", end="")
    for clasa in CLASE:
        print(f" {clasa:>10}", end="")
    print()
    print(f"  {'-'*55}")

    for vv in VERSIUNI_VARIANTE:
        data = incarca_rezultate(vv, args.set_date)
        if data is None:
            continue
        for m in data.get("metrici", []):
            print(f"  {m['model']:<22} {vv:<10}", end="")
            for clasa in CLASE:
                acc_c = m.get("per_clasa", {}).get(clasa, None)
                if acc_c is not None:
                    print(f" {acc_c:>10.0%}", end="")
                else:
                    print(f" {'N/A':>10}", end="")
            print()

    # Distributie predictii — util pentru a detecta bias spre negativ
    print(f"\n  Distributie predictii (atentie la bias spre negativ):")
    for vv in VERSIUNI_VARIANTE:
        data = incarca_rezultate(vv, args.set_date)
        if data is None:
            continue
        for m in data.get("metrici", []):
            rezultate = [r for r in data.get("rezultate_detaliate", []) if r["model"] == m["model"]]
            pred = [r["satisfactie_pred"] for r in rezultate]
            dist = dict(Counter(pred))
            print(f"    {m['model']:<22} {vv}: {dist}")


if __name__ == "__main__":
    main()