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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RESULTS_DIR = "./rezultate_prompt_engineering/intentii_local"
GRAFICE_DIR = "./rezultate_prompt_engineering/grafice_intentii_local"

SET_DATE_PER_VERSIUNE = {
    "V1": "tot_setul_tot",
    "V2": "v2_subset_10_per_domeniu",
    "V3": "subset_10_per_domeniu",
    "V4": "subset_10_per_domeniu",
}

INTENTII_DOMENII = {
    "banking": ["problema_credit","tranzactie_gresita","card_blocat","tranzactie_suspecta","problema_transfer","problema_schimb_valutar","problema_sold","card_pierdut"],
    "medicina": ["rezultate_analize","problema_reteta","problema_asigurare","reclamatie_personal","consultatie_anulata","problema_facturare","problema_programare","anulare_programare"],
    "retail": ["produs_lipsa_stoc","comanda_gresita","problema_livrare","problema_garantie","reclamatie_produs","anulare_comanda","comanda_intarziata","retur_produs"],
    "telecom": ["problema_modificare_abonament","portare_esuata","problema_internet","problema_roaming","factura_gresita","reziliere_contract","activare_esuata","problema_semnal"],
    "servicii_publice": ["dosar_respins","contestatie_decizie","informatii_program","reclamatie_serviciu","sesizare_problema","problema_plata_taxa","acte_incomplete","programare_ghiseu"]
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


def genereaza_matrice_confuzie_local(rezultate, model, versiune, target_dir):
    """Genereaza matrice de confuzie per domeniu pentru un model si o versiune specifice."""
    df = pd.DataFrame(rezultate)
    if df.empty:
        return

    # Adaugam si eticheta de fallback in liste
    liste_intentii = {domeniu: liste + ["alta_solicitare"] for domeniu, liste in INTENTII_DOMENII.items()}

    for domeniu, etichete_valide in liste_intentii.items():
        df_domeniu = df[df["domeniu"] == domeniu]
        if df_domeniu.empty:
            continue

        # Calculul matricei
        cm = confusion_matrix(df_domeniu["intentie_gold"], df_domeniu["intentie_pred"], labels=etichete_valide)
        etichete_curate = [e.replace("_", " ") for e in etichete_valide]
        cm_df = pd.DataFrame(cm, index=etichete_curate, columns=etichete_curate)

        # Plot structural academic
        plt.figure(figsize=(9, 7))
        sns.set_theme(style="white")
        sns.heatmap(
            cm_df, annot=True, fmt="d", cmap="Blues", cbar=True,
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

        # Salvare organizata in subfoldere per model si versiune
        output_subdir = os.path.join(target_dir, model, versiune)
        os.makedirs(output_subdir, exist_ok=True)
        
        cale_salvare = os.path.join(output_subdir, f"matrice_{domeniu}.png")
        plt.savefig(cale_salvare, dpi=300)
        plt.close()


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

            # --- AICI INTEGRAM GENERAREA AUTOMATA DE MATRICE ---
            genereaza_matrice_confuzie_local(rezultate, model, versiune, GRAFICE_DIR)

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