import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def genereaza_matrice_confuzie_domenii(rezultate, results_dir="grafice_disertatie"):
    """Genereaza matrice de confuzie per domeniu si una globala pe baza listei de rezultate."""
    # Cream directorul de salvare daca nu exista
    os.makedirs(results_dir, exist_ok=True)

    # Convertim lista de rezultate intr-un DataFrame pentru filtrare usoara
    df = pd.DataFrame(rezultate)

    # Dictionary-ul cu etichetele complete preluat din structura ta
    INTENTII_DOMENII = {
        "banking": [
            "problema_credit",
            "tranzactie_gresita",
            "card_blocat",
            "tranzactie_suspecta",
            "problema_transfer",
            "problema_schimb_valutar",
            "problema_sold",
            "card_pierdut",
        ],
        "medicina": [
            "rezultate_analize",
            "problema_reteta",
            "problema_asigurare",
            "reclamatie_personal",
            "consultatie_anulata",
            "problema_facturare",
            "problema_programare",
            "anulare_programare",
        ],
        "retail": [
            "produs_lipsa_stoc",
            "comanda_gresita",
            "problema_livrare",
            "problema_garantie",
            "reclamatie_produs",
            "anulare_comanda",
            "comanda_intarziata",
            "retur_produs",
        ],
        "telecom": [
            "problema_modificare_abonament",
            "portare_esuata",
            "problema_internet",
            "problema_roaming",
            "factura_gresita",
            "reziliere_contract",
            "activare_esuata",
            "problema_semnal",
        ],
        "servicii_publice": [
            "dosar_respins",
            "contestatie_decizie",
            "informatii_program",
            "reclamatie_serviciu",
            "sesizare_problema",
            "problema_plata_taxa",
            "acte_incomplete",
            "programare_ghiseu",
        ],
    }

    # Adaugam si eticheta de fallback "alta_solicitare" in cazul in care modelul a esuat complet la extragere
    for domeniu in INTENTII_DOMENII:
        INTENTII_DOMENII[domeniu].append("alta_solicitare")

    # --- 1. GENERARE MATRICE PER DOMENIU ---
    for domeniu, etichete_valide in INTENTII_DOMENII.items():
        df_domeniu = df[df["domeniu"] == domeniu]

        if df_domeniu.empty:
            print(f"  [Info] Nu exista date pentru domeniul: {domeniu}")
            continue

        # Generam matricea folosind DOAR etichetele specifice acelui domeniu
        cm = confusion_matrix(
            df_domeniu["intentie_gold"],
            df_domeniu["intentie_pred"],
            labels=etichete_valide,
        )

        # Curatam denumirile etichetelor pentru grafic (scoatem underscore-urile pentru aspect academic)
        etichete_curate = [e.replace("_", " ") for e in etichete_valide]

        cm_df = pd.DataFrame(
            cm, index=etichete_curate, columns=etichete_curate
        )

        plt.figure(figsize=(9, 7))
        sns.set_theme(style="white")

        # Heatmap configurat curat
        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            linewidths=0.5,
            linecolor="#d3d3d3",
            annot_kws={"size": 11, "weight": "bold"},
        )

        # Preluam numele modelului si versiunea pentru titlu (daca exista in date)
        nume_model = df_domeniu["model"].iloc[0].upper()
        versiune = df_domeniu["versiune_prompt"].iloc[0]

        plt.title(
            f"Matrice de Confuzie — {nume_model} ({versiune})\nDomeniul: {domeniu.replace('_', ' ').upper()}",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.ylabel("Intenție Reală (Ground Truth)", fontsize=11, fontweight="bold")
        plt.xlabel("Intenție Prezistă (Predicted)", fontsize=11, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        # Salvare
        cale_salvare = os.path.join(
            results_dir, f"matrice_{nume_model.lower()}_{versiune}_{domeniu}.png"
        )
        plt.savefig(cale_salvare, dpi=300)
        plt.close()
        print(f"-> Matricea salvata cu succes: {cale_salvare}")

    # --- 2. GENERARE MATRICE GLOBALA (TOATE DOMENIILE COMBINATE) ---
    all_labels = []
    for labs in INTENTII_DOMENII.values():
        all_labels.extend(labs)
    all_labels = sorted(list(set(all_labels)))  # Lista unica cu toate intentiile

    cm_global = confusion_matrix(
        df["intentie_gold"], df["intentie_pred"], labels=all_labels
    )
    all_labels_curate = [e.replace("_", " ") for e in all_labels]
    cm_global_df = pd.DataFrame(
        cm_global, index=all_labels_curate, columns=all_labels_curate
    )

    # Matricea globala e mare (41 de clase), deci setam un figsize generos
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        cm_global_df,
        annot=True,
        fmt="d",
        cmap="Purples",
        cbar=True,
        linewidths=0.2,
        annot_kws={"size": 8},
    )

    nume_model_glob = df["model"].iloc[0].upper()
    versiune_glob = df["versiune_prompt"].iloc[0]

    plt.title(
        f"Matrice de Confuzie Globala — {nume_model_glob} ({versiune_glob})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.ylabel("Toate Intențiile Reale", fontsize=14, fontweight="bold")
    plt.xlabel("Toate Intențiile Prezise", fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cale_salvare_global = os.path.join(
        results_dir, f"matrice_{nume_model_glob.lower()}_{versiune_glob}_GLOBAL.png"
    )
    plt.savefig(cale_salvare_global, dpi=300)
    plt.close()
    print(f"-> Matricea GLOBALA salvata cu succes: {cale_salvare_global}")