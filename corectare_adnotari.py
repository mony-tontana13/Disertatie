import anthropic
import json
import os
import argparse
import time

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
INPUT_BASE_DIR = "./conversatii_adnotate"
OUTPUT_BASE_DIR = "./conversatii_adnotate_corectate"  # Suprascrie fisierele existente

# ============================================================
# INTENTIILE DEFINITE INITIAL PER DOMENIU (din intentii.txt)
# ============================================================

INTENTII_DOMENII = {
    "banking": [
        "problema_credit", "tranzactie_gresita", "card_blocat", "tranzactie_suspecta",
        "problema_transfer", "problema_schimb_valutar", "problema_sold", "card_pierdut",
        "problema_credit", "tranzactie_gresita", "card_blocat", "tranzactie_suspecta",
        "problema_transfer", "problema_schimb_valutar", "problema_sold", "card_pierdut",
        "problema_credit", "tranzactie_gresita", "card_blocat", "tranzactie_suspecta"
    ],
    "medicina": [
        "rezultate_analize", "problema_reteta", "problema_asigurare", "reclamatie_personal",
        "consultatie_anulata", "problema_facturare", "problema_programare", "anulare_programare",
        "rezultate_analize", "problema_reteta", "problema_asigurare", "reclamatie_personal",
        "consultatie_anulata", "problema_facturare", "problema_programare", "anulare_programare",
        "rezultate_analize", "problema_reteta", "problema_asigurare", "reclamatie_personal"
    ],
    "retail": [
        "produs_lipsa_stoc", "comanda_gresita", "problema_livrare", "problema_garantie",
        "reclamatie_produs", "anulare_comanda", "comanda_intarziata", "retur_produs",
        "produs_lipsa_stoc", "comanda_gresita", "problema_livrare", "problema_garantie",
        "reclamatie_produs", "anulare_comanda", "comanda_intarziata", "retur_produs",
        "produs_lipsa_stoc", "comanda_gresita", "problema_livrare", "problema_garantie"
    ],
    "telecom": [
        "problema_modificare_abonament", "portare_esuata", "problema_internet", "problema_roaming",
        "factura_gresita", "reziliere_contract", "activare_esuata", "problema_semnal",
        "problema_modificare_abonament", "portare_esuata", "problema_internet", "problema_roaming",
        "factura_gresita", "reziliere_contract", "activare_esuata", "problema_semnal",
        "problema_modificare_abonament", "portare_esuata", "problema_internet", "problema_roaming"
    ],
    "servicii_publice": [
        "dosar_respins", "contestatie_decizie", "informatii_program", "reclamatie_serviciu",
        "sesizare_problema", "problema_plata_taxa", "acte_incomplete", "programare_ghiseu",
        "dosar_respins", "contestatie_decizie", "informatii_program", "reclamatie_serviciu",
        "sesizare_problema", "problema_plata_taxa", "acte_incomplete", "programare_ghiseu",
        "dosar_respins", "contestatie_decizie", "informatii_program", "reclamatie_serviciu"
    ]
}

# ============================================================
# DISTRIBUTIA — ordinea exacta a celor 20 de conversatii
# ============================================================

DISTRIBUTIE = [
    {"complexitate": "simpla",  "satisfactie": "pozitiv", "tip": None},
    {"complexitate": "simpla",  "satisfactie": "pozitiv", "tip": None},
    {"complexitate": "simpla",  "satisfactie": "pozitiv", "tip": None},
    {"complexitate": "simpla",  "satisfactie": "neutru",  "tip": None},
    {"complexitate": "simpla",  "satisfactie": "neutru",  "tip": None},
    {"complexitate": "simpla",  "satisfactie": "neutru",  "tip": None},
    {"complexitate": "complexa", "satisfactie": "neutru",  "tip": "multi_intentie"},
    {"complexitate": "complexa", "satisfactie": "neutru",  "tip": "ezitanta"},
    {"complexitate": "complexa", "satisfactie": "negativ", "tip": "problema_nerezolvata"},
    {"complexitate": "complexa", "satisfactie": "negativ", "tip": "pasat_intre_departamente"},
    {"complexitate": "complexa", "satisfactie": "negativ", "tip": "dialect_familiar"},
    {"complexitate": "complexa", "satisfactie": "negativ", "tip": "context_implicit"},
    {"complexitate": "complexa", "satisfactie": "negativ", "tip": "negata_conditionata"},
    {"complexitate": "complexa", "satisfactie": "negativ", "tip": "reclamatie_implicita"},
    {"complexitate": "complexa", "satisfactie": "negativ", "tip": "ambigua"},
    {"complexitate": "complexa", "satisfactie": "negativ", "tip": "ezitanta"},
    {"complexitate": "complexa", "satisfactie": "negativ", "tip": "informatii_contradictorii"},
    {"complexitate": "complexa", "satisfactie": "negativ", "tip": "dialect_familiar"},
    {"complexitate": "complexa", "satisfactie": "negativ", "tip": "multi_intentie"},
    {"complexitate": "complexa", "satisfactie": "negativ", "tip": "context_implicit"},
]


def detecteaza_intentie_secundara(client, conversatie, domeniu, intentie_primara, intentii_disponibile):
    """Detecteaza a doua intentie din dialogul unei conversatii multi_intentie."""
    dialog_text = "\n".join([f"{r['rol'].upper()}: {r['text']}" for r in conversatie])
    intentii_str = ", ".join([i for i in intentii_disponibile if i != intentie_primara])

    prompt = f"""Analizeaza conversatia telefonica de mai jos.
Intentia principala a clientului este deja cunoscuta: {intentie_primara}

Identifica daca clientul a avut si o A DOUA intentie sau problema in aceeasi conversatie.
Alege DOAR din lista: [{intentii_str}]
Daca nu exista o a doua intentie clara, raspunde cu: niciuna

CONVERSATIE:
{dialog_text}

Raspunde DOAR cu numele intentiei secundare sau "niciuna":"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )

    raspuns = response.content[0].text.strip().lower()

    # Verifica daca raspunsul e o intentie valida
    for intentie in intentii_disponibile:
        if intentie in raspuns and intentie != intentie_primara:
            return intentie

    return None


def get_prefix(domeniu):
    prefixuri = {
        "banking": "BNK",
        "medicina": "MED",
        "retail": "RET",
        "telecom": "TEL",
        "servicii_publice": "SP"
    }
    return prefixuri[domeniu]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domeniu", required=True,
                        choices=["banking", "medicina", "retail", "telecom", "servicii_publice"])
    args = parser.parse_args()

    domeniu = args.domeniu
    prefix = get_prefix(domeniu)
    intentii = INTENTII_DOMENII[domeniu]
    intentii_unice = list(dict.fromkeys(intentii))  # Lista unica pentru detectare secundara

    input_dir = os.path.join(INPUT_BASE_DIR, domeniu)
    output_dir = os.path.join(OUTPUT_BASE_DIR, domeniu)
    os.makedirs(output_dir, exist_ok=True)

    client = anthropic.Anthropic(api_key=API_KEY)

    print(f"\nCorectez adnotarile pentru: {domeniu}")
    print(f"Folder: {input_dir}\n")

    stats = {"corectat": 0, "ok": 0, "eroare": 0}

    for idx, spec in enumerate(DISTRIBUTIE):
        nr = idx + 1
        conv_id = f"{prefix}_{nr:03d}"
        fisier_name = f"{conv_id}.json"
        input_path = os.path.join(input_dir, fisier_name)
        output_path = os.path.join(output_dir, fisier_name)

        if not os.path.exists(input_path):
            print(f"  [{nr:02d}/20] {conv_id} — fisier lipsa, skip")
            continue

        print(f"  [{nr:02d}/20] {conv_id}", end="", flush=True)

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                conv = json.load(f)

            intentie_corecta = intentii[idx]  # Intentia exacta pentru pozitia aceasta
            satisfactie_corecta = spec["satisfactie"]
            complexitate_corecta = spec["complexitate"]
            tip_corectat = spec["tip"]

            # Construieste intentie_gold
            if tip_corectat == "multi_intentie":
                # Detecteaza a doua intentie din dialog
                intentie_secundara = detecteaza_intentie_secundara(
                    client, conv["conversatie"], domeniu,
                    intentie_corecta, intentii_unice
                )
                if intentie_secundara:
                    intentie_gold = [intentie_corecta, intentie_secundara]
                else:
                    intentie_gold = [intentie_corecta]
                time.sleep(1)
            else:
                intentie_gold = [intentie_corecta]

            # Verifica daca ceva s-a schimbat
            vechi_intentie = conv.get("intentie_gold", [])
            vechi_satisfactie = conv.get("satisfactie", "")
            vechi_complexitate = conv.get("complexitate", "")

            modificari = []
            if vechi_intentie != intentie_gold:
                modificari.append(f"intentie: {vechi_intentie} -> {intentie_gold}")
            if vechi_satisfactie != satisfactie_corecta:
                modificari.append(f"satisfactie: {vechi_satisfactie} -> {satisfactie_corecta}")
            if vechi_complexitate != complexitate_corecta:
                modificari.append(f"complexitate: {vechi_complexitate} -> {complexitate_corecta}")

            # Actualizeaza campurile
            conv["intentie_gold"] = intentie_gold
            conv["satisfactie"] = satisfactie_corecta
            conv["complexitate"] = complexitate_corecta
            if tip_corectat:
                conv["tip_complexitate"] = tip_corectat

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(conv, f, ensure_ascii=False, indent=2)

            if modificari:
                print(f" CORECTAT: {" | ".join(modificari)}")
                stats["corectat"] += 1
            else:
                print(f" ok ({intentie_gold[0]} / {satisfactie_corecta})")
                stats["ok"] += 1

        except Exception as e:
            print(f" EROARE: {e}")
            stats["eroare"] += 1

    print(f"\n--- SUMAR CORECTARE {domeniu.upper()} ---")
    print(f"  Corectate: {stats['corectat']}")
    print(f"  Deja corecte: {stats['ok']}")
    print(f"  Erori: {stats['eroare']}")


if __name__ == "__main__":
    main()