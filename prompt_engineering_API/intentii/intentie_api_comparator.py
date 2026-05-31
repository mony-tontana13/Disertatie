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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

RESULTS_DIR = "./rezultate_prompt_engineering/intentii_api"
GRAFICE_DIR = "./rezultate_prompt_engineering/grafice_intentii_api"
VERSIUNI = ["V1_v2", "V2_v1", "V3_v1", "V4_v1"]
MODELE = ["GPT-4.1-mini", "Gemini-2.5-flash", "command-r7b-12-2024", "Aya-Expanse-8b"]

INTENTII_DOMENII = {
    "banking": ["problema_credit","tranzactie_gresita","card_blocat","tranzactie_suspecta","problema_transfer","problema_schimb_valutar","problema_sold","card_pierdut"],
    "medicina": ["rezultate_analize","problema_reteta","problema_asigurare","reclamatie_personal","consultatie_anulata","problema_facturare","problema_programare","anulare_programare"],
    "retail": ["produs_lipsa_stoc","comanda_gresita","problema_livrare","problema_garantie","reclamatie_produs","anulare_comanda","comanda_intarziata","retur_produs"],
    "telecom": ["problema_modificare_abonament","portare_esuata","problema_internet","problema_roaming","factura_gresita","reziliere_contract","activare_esuata","problema_semnal"],
    "servicii_publice": ["dosar_respins","contestatie_decizie","informatii_program","reclamatie_serviciu","sesizare_problema","problema_plata_taxa","acte_incomplete","programare_ghiseu"]
}


def incarca_rezultate(versiune, set_date):
    pattern = f"intentie_api_{versiune}_{set_date}.json"
    filepath = os.path.join(RESULTS_DIR, pattern)
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    return None


def genereaza_matrice_confuzie_api(data_json, versiune, target_dir):
    """
    Extrage rezultatele detaliate per model dintr-un fisier de evaluare API 
    si genereaza matricele de confuzie aferente fiecarui domeniu.
    """
    # Verificam daca fisierul are structura detaliata salvata
    rezultate_toate = data_json.get("rezultate_detaliate", [])
    if not rezultate_toate:
        return

    df_complet = pd.DataFrame(rezultate_toate)
    
    # Adaugam si eticheta de fallback in liste
    liste_intentii = {domeniu: liste + ["alta_solicitare"] for domeniu, liste in INTENTII_DOMENII.items()}

    # Pentru fiecare model gasit in rularea respectiva
    for model in df_complet["model"].unique():
        df_model = df_complet[df_complet["model"] == model]

        for domeniu, etichete_valide in liste_intentii.items():
            df_domeniu = df_model[df_model["domeniu"] == domeniu]
            if df_domeniu.empty:
                continue

            # Calculul efectiv al matricei
            cm = confusion_matrix(df_domeniu["intentie_gold"], df_domeniu["intentie_pred"], labels=etichete_valide)
            etichete_curate = [e.replace("_", " ") for e in etichete_valide]
            cm_df = pd.DataFrame(cm, index=etichete_curate, columns=etichete_curate)

            # Design-ul graficului
            plt.figure(figsize=(9, 7))
            sns.set_theme(style="white")
            sns.heatmap(
                cm_df, annot=True, fmt="d", cmap="Purples", cbar=True,
                linewidths=0.5, linecolor="#d3d3d3", annot_kws={"size": 11, "weight": "bold"}
            )

            plt.title(
                f"Matrice de Confuzie — {model.upper()} ({versiune})\nDomeniul: {domeniu.replace('_', ' ').upper()}",
                fontsize=12, fontweight="bold", pad=15
            )
            plt.ylabel("Intenție Reală (Ground Truth)", fontsize=11, fontweight="bold")
            plt.xlabel("Intenție Prezistă (Predicted)", fontsize=11, fontweight="bold")
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()

            # Organizare in foldere: grafice_intentii_api / MODEL / VERSIUNE
            output_subdir = os.path.join(target_dir, model.replace("/", "_"), versiune)
            os.makedirs(output_subdir, exist_ok=True)
            
            cale_salvare = os.path.join(output_subdir, f"matrice_{domeniu}.png")
            plt.savefig(cale_salvare, dpi=300)
            plt.close()


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
            
        # --- GENERARE MATRICE CONFUIZIE PENTRU ACEASTA VERSIUNE ---
        genereaza_matrice_confuzie_api(data, versiune, GRAFICE_DIR)
        
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
    data_v4 = incarca_rezultate("V4_v1", args.set_date)
    if data_v4:
        for m in data_v4.get("metrici", []):
            print(f"\n  {m['model']}:")
            for domeniu, acc in m.get("per_domeniu", {}).items():
                print(f"    {domeniu:<22}: {acc:.0%}")


if __name__ == "__main__":
    main()