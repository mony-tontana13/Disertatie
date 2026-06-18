import anthropic
import json
import os
import argparse
import time
import re

API_KEY = "PLACEHOLDER"

DOMENII = {
    "banking": {
        "prefix": "BNK",
        "intentii": [
            "problema_sold",
            "card_pierdut",
            "problema_transfer",
            "problema_credit",
            "tranzactie_gresita",
            "problema_schimb_valutar",
            "card_blocat",
            "tranzactie_suspecta"
        ],
        "descriere": "serviciul clienți al unei bănci comerciale"
    },
    "medicina": {
        "prefix": "MED",
        "intentii": [
            "problema_programare",
            "anulare_programare",
            "consultatie_anulata",
            "rezultate_analize",
            "problema_reteta",
            "problema_facturare",
            "problema_asigurare",
            "reclamatie_personal"
        ],
        "descriere": "serviciul de programări al unei clinici medicale private"
    },
    "retail": {
        "prefix": "RET",
        "intentii": [
            "comanda_intarziata",
            "retur_produs",
            "reclamatie_produs",
            "produs_lipsa_stoc",
            "comanda_gresita",
            "anulare_comanda",
            "problema_livrare",
            "problema_garantie"
        ],
        "descriere": "serviciul clienți al unui magazin online"
    },
    "telecom": {
        "prefix": "TEL",
        "intentii": [
            "activare_esuata",
            "problema_semnal",
            "factura_gresita",
            "problema_modificare_abonament",
            "portare_esuata",
            "reziliere_contract",
            "problema_internet",
            "problema_roaming"
        ],
        "descriere": "serviciul clienți al unui operator de telecomunicații"
    },
    "servicii_publice": {
        "prefix": "SP",
        "intentii": [
            "acte_incomplete",
            "programare_ghiseu",
            "sesizare_problema",
            "dosar_respins",
            "contestatie_decizie",
            "problema_plata_taxa",
            "informatii_program",
            "reclamatie_serviciu"
        ],
        "descriere": "serviciul de relații cu publicul al unei instituții publice"
    }
}

# Distributia conversatiilor: 6 simple, 14 complexe
# Satisfactie: 3 pozitive, 5 neutre, 12 negative
DISTRIBUTIE = {
    "simple": [
        {"complexitate": "simpla", "satisfactie": "pozitiv"},
        {"complexitate": "simpla", "satisfactie": "pozitiv"},
        {"complexitate": "simpla", "satisfactie": "pozitiv"},
        {"complexitate": "simpla", "satisfactie": "neutru"},
        {"complexitate": "simpla", "satisfactie": "neutru"},
        {"complexitate": "simpla", "satisfactie": "neutru"},
    ],
    "complexe": [
        {"complexitate": "complexa", "satisfactie": "neutru", "tip": "multi_intentie"},
        {"complexitate": "complexa", "satisfactie": "neutru", "tip": "ezitanta"},
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
}


def aloca_intentii(intentii_disponibile, nr_conversatii=20):
    import random
    random.seed(42)
    
    intentii_shuffled = intentii_disponibile.copy()
    random.shuffle(intentii_shuffled)
    
    rezultat = []
    for i in range(nr_conversatii):
        rezultat.append(intentii_shuffled[i % len(intentii_shuffled)])
    
    return rezultat


def build_prompt(domeniu_info, complexitate, satisfactie, intentie_primara, tip_complexitate=None, intentie_secundara=None):
    tip_desc = ""
    if tip_complexitate:
        tipuri = {
            "multi_intentie": f"clientul suna cu intentia principala '{intentie_primara}', iar ulterior in conversatie ridica o a doua problema legata de '{intentie_secundara or intentie_primara}'",
            "ezitanta": "clientul vorbeste ezitant, cu pauze si reformulari, dar intenția lui este clara",
            "problema_nerezolvata": "clientul pleaca cu problema nerezolvata, operatorul nu poate ajuta din motive procedurale sau tehnice",
            "pasat_intre_departamente": "clientul este redirectionat intre departamente fara a-si rezolva problema",
            "dialect_familiar": "clientul foloseste un registru mai informal si expresii regionale, dar intentia este clara",
            "context_implicit": "clientul presupune ca operatorul stie deja contextul unui apel anterior si face referire la el",
            "negata_conditionata": "clientul formuleaza cereri conditionate sau negatii — stie ce nu vrea si incearca sa obtina alternativa",
            "reclamatie_implicita": "clientul este nemultumit dar nu exprima explicit — nemultumirea reiese din ton, remarci scurte si din ce nu spune",
            "ambigua": "clientul descrie o situatie neclara din care intentia nu reiese imediat, operatorul trebuie sa clarifice",
            "informatii_contradictorii": "clientul a primit informatii contradictorii anterior si suna pentru clarificare",
        }
        tip_desc = f"\nTipul de complexitate: {tipuri.get(tip_complexitate, tip_complexitate)}"

    satisfactie_desc = {
        "pozitiv": "problema se rezolva complet, clientul pleaca multumit si o exprima la final in mod clar",
        "neutru": "problema se rezolva tehnic dar interactiunea este rece sau birocrativa, clientul pleaca indiferent",
        "negativ": (
            "clientul pleaca frustrat sau nemultumit — problema nu se rezolva sau este pasat. "
            "IMPORTANT: frustrarea trebuie sa fie atat implicita, cat si explicita. "
            "Clientul nu tipa si nu insulta in fiecare conversatie. Frustrarea se poate manifesta prin: "
            "remarci ironice, "
            "suparare, "
            "raspunsuri scurte si taioase, sau prin faptul ca pune punct brusc conversatiei."
        )
    }

    complexitate_desc = {
        "simpla": f"Conversatie directa si eficienta, 10-15 replici totale. Clientul suna cu intentia: '{intentie_primara}'. Conversatia trebuie sa fie centrata exclusiv pe aceasta intentie.",
        "complexa": f"Conversatie mai lunga, 20-30 replici totale. Clientul suna cu intentia principala: '{intentie_primara}'.{tip_desc}"
    }

    return f"""Genereaza o conversatie telefonica in limba romana intre un client si un operator uman de la {domeniu_info["descriere"]}.

CERINTE OBLIGATORII:
- Operatorul incepe prin a declara numele sau si compania la care a sunat clientul si sa intrebe cu ce poate ajuta.
- Evita replicile care sa denote incantare excesiva sau sa para false. 
- Limbajul trebuie sa fie PROFESIONAL si NATURAL — ca intr-un call-center real. 
  Operatorul vorbeste clar, politicos, vorbeste la persoana a doua plural, foloseste formule de politete adecvate.
  Clientul vorbeste natural, dar nu colocvial sau informal excesiv.
- {complexitate_desc[complexitate]}
- Nivel satisfactie client: {satisfactie_desc[satisfactie]}
- Conversatia TREBUIE sa se incheie cu o formula de inchidere (la revedere, o zi buna etc.)
- Operatorul este UMAN — are empatie, dar este si profesionist.
- Intentia clientului trebuie sa reiasa din ceea ce CERE sau INTREABA clientul.
  NU include ca intentie actiunile pe care le face operatorul din proprie initiativa.

FORMAT RASPUNS — returneaza DOAR JSON valid, fara text suplimentar:
{{
  "conversatie": [
    {{"rol": "operator", "text": "..."}},
    {{"rol": "client", "text": "..."}},
    ...
  ]
}}"""


def generate_conversation(client, domeniu_info, complexitate, satisfactie, intentie_primara, tip_complexitate=None, intentie_secundara=None):
    prompt = build_prompt(domeniu_info, complexitate, satisfactie, intentie_primara, tip_complexitate, intentie_secundara)

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            if part.startswith("json"):
                text = part[4:].strip()
                break
            elif "{" in part:
                text = part.strip()
                break

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end != 0:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r'(?<!\\)"(?=[^:,\}\]\s])', '\\"'  , text)
        return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Genereaza conversatii pentru un domeniu dat.")
    parser.add_argument("--domeniu", required=True, choices=list(DOMENII.keys()),
                        help="Domeniul pentru care se genereaza conversatiile")
    parser.add_argument("--output_dir", default="./conversatii_generate",
                        help="Directorul de output (default: ./conversatii_generate)")
    args = parser.parse_args()

    domeniu = args.domeniu
    domeniu_info = DOMENII[domeniu]
    prefix = domeniu_info["prefix"]
    output_dir = os.path.join(args.output_dir, domeniu)
    os.makedirs(output_dir, exist_ok=True)

    client = anthropic.Anthropic(api_key=API_KEY)
    toate_conversatiile = []
    for conv in DISTRIBUTIE["simple"]:
        toate_conversatiile.append({
            "complexitate": conv["complexitate"],
            "satisfactie": conv["satisfactie"],
            "tip_complexitate": None
        })
    for conv in DISTRIBUTIE["complexe"]:
        toate_conversatiile.append({
            "complexitate": conv["complexitate"],
            "satisfactie": conv["satisfactie"],
            "tip_complexitate": conv.get("tip")
        })
    intentii_alocate = aloca_intentii(domeniu_info["intentii"], len(toate_conversatiile))

    print(f"\nGenerez 20 conversatii pentru domeniul: {domeniu}")
    print(f"Output: {output_dir}\n")
    print("Intentii alocate:")
    for i, intentie in enumerate(intentii_alocate, 1):
        print(f"  {i:02d}. {intentie}")
    print()

    for i, (conv_spec, intentie) in enumerate(zip(toate_conversatiile, intentii_alocate), 1):
        conv_id = f"{prefix}_{i:03d}"
        output_path = os.path.join(output_dir, f"{conv_id}.json")

        if os.path.exists(output_path):
            print(f"  [{i:02d}/20] {conv_id} — deja exista, skip")
            continue

        # Pentru multi_intentie, aloca o intentie secundara diferita
        intentie_secundara = None
        if conv_spec["tip_complexitate"] == "multi_intentie":
            intentii_disponibile = [x for x in domeniu_info["intentii"] if x != intentie]
            import random
            intentie_secundara = random.choice(intentii_disponibile)

        print(f"  [{i:02d}/20] {conv_id} — {conv_spec['complexitate']} / {conv_spec['satisfactie']} / {intentie}", end="", flush=True)

        try:
            result = generate_conversation(
                client,
                domeniu_info,
                conv_spec["complexitate"],
                conv_spec["satisfactie"],
                intentie,
                conv_spec["tip_complexitate"],
                intentie_secundara
            )

            fisier = {
                "id": conv_id,
                "intentie_primara": intentie,
                "intentie_secundara": intentie_secundara,
                "conversatie": result["conversatie"]
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(fisier, f, ensure_ascii=False, indent=2)

            print(f" ({len(result['conversatie'])} replici)")

        except Exception as e:
            print(f" EROARE: {e}")

        time.sleep(3)

    print(f"\nFinalizat! Fisierele sunt in: {output_dir}")


if __name__ == "__main__":
    main()