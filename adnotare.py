import anthropic
import json
import os
import argparse
import time

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

INPUT_BASE_DIR = "./conversatii_corectate"
OUTPUT_BASE_DIR = "./conversatii_adnotate"

DOMENII_CONFIG = {
    "banking": { 
        "descriere": "serviciul clienți al unei bănci",
        "intentii_posibile": [
            "problema_sold",
            "card_pierdut",
            "problema_transfer",
            "problema_credit",
            "tranzactie_gresita",
            "problema_schimb_valutar",
            "card_blocat",
            "tranzactie_suspecta"
        ]
    },
    "medicina": {
        "descriere": "serviciul de programări al unei clinici medicale",
        "intentii_posibile": [
            "problema_programare",
            "anulare_programare",
            "consultatie_anulata",
            "rezultate_analize",
            "problema_reteta",
            "problema_facturare",
            "problema_asigurare",
            "reclamatie_personal"
        ]
    },
    "retail": {
        "descriere": "serviciul clienți al unui magazin online",
        "intentii_posibile": [
            "comanda_intarziata",
            "retur_produs",
            "reclamatie_produs",
            "produs_lipsa_stoc",
            "comanda_gresita",
            "anulare_comanda",
            "problema_livrare",
            "problema_garantie"
        ]
    },
    "telecom": {
        "descriere": "serviciul clienți al unui operator de telecomunicații",
        "intentii_posibile": [
            "activare_esuata",
            "problema_semnal",
            "factura_gresita",
            "problema_modificare_abonament",
            "portare_esuata",
            "reziliere_contract",
            "problema_internet",
            "problema_roaming"
        ]
    },
    "servicii_publice": {
        "descriere": "serviciul de relații cu publicul al unei instituții publice",
        "intentii_posibile": [
            "acte_incomplete",
            "programare_ghiseu",
            "sesizare_problema",
            "dosar_respins",
            "contestatie_decizie",
            "problema_plata_taxa",
            "informatii_program",
            "reclamatie_serviciu"
        ]
    }
}


def get_tip_rezumat(satisfactie):
    if satisfactie == "pozitiv":
        return "SCURT", "1-2 propozitii, 20-40 cuvinte, doar ideile principale"
    elif satisfactie == "neutru":
        return "MEDIU", "3-4 propozitii, 40-70 cuvinte, ideile principale plus contextul si actiunile cheie"
    else:
        return "LUNG", "5-7 propozitii, 60-100 cuvinte, toate detaliile relevante si evolutia completa a conversatiei"


def trunchieaza_rezumat(text, tip):
    limite = {"SCURT": 40, "MEDIU": 70, "LUNG": 100}
    maxim = limite.get(tip, 100)
    cuvinte = text.strip().split()
    if len(cuvinte) <= maxim:
        return text.strip()
    cuvinte_taiate = cuvinte[:maxim]
    text_taiat = " ".join(cuvinte_taiate)
    last_dot = text_taiat.rfind(".")
    if last_dot > len(text_taiat) // 2:
        return text_taiat[:last_dot + 1]
    return text_taiat + "."


def build_annotation_prompt(conversatie, domeniu, intentii_posibile, intentie_primara=None, intentie_secundara=None):
    dialog_text = "\n".join(
        [f"{r['rol'].upper()}: {r['text']}" for r in conversatie]
    )

    intentii_str = ", ".join(intentii_posibile)

    ghid_intentie = ""
    if intentie_primara:
        ghid_intentie = f"""
CONTEXT INTENTII (din procesul de generare):
- Intentia principala cu care a sunat clientul: {intentie_primara}
- Intentia secundara (daca exista): {intentie_secundara or "niciuna"}
Verifica daca acestea reies din conversatie si confirma-le sau corecteaza-le daca e necesar.
Nu confunda intentiile cu care suna clientul cu actiunile pe care acesta le confirma de-a lungul cnversatiei. 
"""

    return f"""Esti un expert in analiza conversatiilor telefonice din domeniul {domeniu}.
Analizeaza conversatia de mai jos si returneaza DOAR un obiect JSON valid, fara text suplimentar.

CONVERSATIE:
{dialog_text}
{ghid_intentie}
INSTRUCTIUNI DE ADNOTARE:

1. COMPLEXITATE: Determina daca conversatia este "simpla" (directa, o intentie clara, 10-15 replici)
   sau "complexa" (ambigua, multiple intentii, limbaj colocvial/regional, 20+ replici).

2. INTENTIE_GOLD: Identifica EXCLUSIV intentiile CLIENTULUI din lista: [{intentii_str}].
   REGULI STRICTE:
   - Include DOAR ce a cerut sau intrebat clientul, nu actiunile operatorului
   - De obicei conversatiile au maxim doua intentii
   - Daca operatorul a blocat cardul din proprie initiativa, NU include blocare_card ca intentie
   - Daca clientul a sunat pentru reclamatie_tranzactie si ulterior a intrebat de card nou, include ambele
   - Returneaza o lista cu una sau mai multe intentii
   - ACTIUNILE ULTERIOARE INTENTIEI INITAILE NU INSEAMNA INTENTII MULTIPLE
   - Intentiile multiple sunt atunci cand clientul pune mai multe intrebari separate/care nu au legatura intre ele

3. SATISFACTIE: Evalueaza nivelul de satisfactie al clientului la finalul conversatiei:
   - "pozitiv": problema rezolvata complet, clientul multumit si o exprima clar
   - "neutru": problema rezolvata tehnic dar clientul nu prezinta nicio emotie si este neutru.
   - "negativ": clientul pleaca frustrat sau nemultumit, chiar daca implicit
     (ironie, resemnare, replici taioase, inchide brusc conversatia)
   IMPORTANT: un singur comentariu negativ urmat de acceptare NU inseamna automat negativ.
   Uita-te la TONUL GENERAL si REZULTATUL FINAL.

4. MOTIV_SATISFACTIE: Un string scurt care descrie motivul principal al nivelului de satisfactie.
   Exemple: "problema_rezolvata_rapid", "pasat_intre_departamente", "informatii_contradictorii",
   "timp_asteptare_mare", "promisiune_nerespectata", "problema_nerezolvata", "rezolvat_mecanic_fara_empatie"

5. REZUMAT: Genereaza un rezumat al conversatiei in limba romana.
   Lungimea depinde de satisfactie:
   - pozitiv: SCURT — 1-2 propozitii, 20-40 cuvinte, ideile principale
   - neutru: MEDIU — 3-4 propozitii, 40-70 cuvinte, ideile principale plus contextul
   - negativ: LUNG — 5-7 propozitii, 60-100 cuvinte, detalii relevante si evolutia conversatiei

FORMAT RASPUNS (DOAR JSON, fara markdown, fara explicatii):
{{
  "complexitate": "simpla" sau "complexa",
  "intentie_gold": ["intentie1"] sau ["intentie1", "intentie2"],
  "satisfactie": "pozitiv" sau "neutru" sau "negativ",
  "motiv_satisfactie": "descriere_scurta",
  "rezumat": "Rezumatul conversatiei..."
}}"""


def curata_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end != 0:
        text = text[start:end]
    return text.strip()


def annotate_conversation(client, conversatie, domeniu, intentii_posibile, intentie_primara=None, intentie_secundara=None):
    prompt = build_annotation_prompt(conversatie, domeniu, intentii_posibile, intentie_primara, intentie_secundara)

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = curata_json(response.content[0].text)
    adnotari = json.loads(text)

    tip_rezumat, _ = get_tip_rezumat(adnotari["satisfactie"])
    adnotari["rezumat"] = trunchieaza_rezumat(adnotari["rezumat"], tip_rezumat)

    return adnotari


def main():
    parser = argparse.ArgumentParser(description="Adnoteaza conversatiile pentru un domeniu dat.")
    parser.add_argument("--domeniu", required=True, choices=list(DOMENII_CONFIG.keys()),
                        help="Domeniul pentru care se adnoteaza conversatiile")
    args = parser.parse_args()

    domeniu = args.domeniu
    config = DOMENII_CONFIG[domeniu]
    intentii = config["intentii_posibile"]

    input_dir = os.path.join(INPUT_BASE_DIR, domeniu)
    output_dir = os.path.join(OUTPUT_BASE_DIR, domeniu)
    os.makedirs(output_dir, exist_ok=True)

    fisiere = sorted([f for f in os.listdir(input_dir) if f.endswith(".json")])
    if not fisiere:
        print(f"Nu am gasit fisiere JSON in: {input_dir}")
        return

    client = anthropic.Anthropic(api_key=API_KEY)

    print(f"\nAdnotez {len(fisiere)} conversatii pentru domeniul: {domeniu}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}\n")

    for i, fisier_name in enumerate(fisiere, 1):
        input_path = os.path.join(input_dir, fisier_name)
        output_path = os.path.join(output_dir, fisier_name)

        if os.path.exists(output_path):
            print(f"  [{i:02d}/{len(fisiere)}] {fisier_name} — deja adnotat, skip")
            continue

        print(f"  [{i:02d}/{len(fisiere)}] {fisier_name}", end="", flush=True)

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                date_originale = json.load(f)

            conversatie = date_originale["conversatie"]

            # Preia intentiile din fisierul generat de Script 1 daca exista
            intentie_primara = date_originale.get("intentie_primara")
            intentie_secundara = date_originale.get("intentie_secundara")

            adnotari = annotate_conversation(
                client, conversatie, domeniu, intentii,
                intentie_primara, intentie_secundara
            )

            fisier_adnotat = {
                "id": date_originale["id"],
                "domeniu": domeniu,
                "complexitate": adnotari["complexitate"],
                "intentie_gold": adnotari["intentie_gold"],
                "satisfactie": adnotari["satisfactie"],
                "motiv_satisfactie": adnotari["motiv_satisfactie"],
                "rezumat": adnotari["rezumat"],
                "conversatie": conversatie
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(fisier_adnotat, f, ensure_ascii=False, indent=2)

            tip_rez, _ = get_tip_rezumat(adnotari["satisfactie"])
            nr_cuvinte = len(adnotari["rezumat"].split())
            print(f" [{adnotari['satisfactie']}] {adnotari['intentie_gold']} | rezumat {tip_rez}: {nr_cuvinte} cuv")

        except json.JSONDecodeError as e:
            print(f" EROARE JSON: {e}")
        except Exception as e:
            print(f" EROARE: {e}")

        time.sleep(1)

    print(f"\nFinalizat! Fisierele adnotate sunt in: {output_dir}")

    print("\n--- SUMAR ADNOTARE ---")
    counts = {"pozitiv": 0, "neutru": 0, "negativ": 0}
    for fisier_name in os.listdir(output_dir):
        if fisier_name.endswith(".json"):
            with open(os.path.join(output_dir, fisier_name), "r", encoding="utf-8") as f:
                d = json.load(f)
            if "satisfactie" in d:
                counts[d["satisfactie"]] = counts.get(d["satisfactie"], 0) + 1
    for nivel, nr in counts.items():
        print(f"  {nivel}: {nr} conversatii")


if __name__ == "__main__":
    main()