"""
Date si configuratii pentru robotul telefonic call-center.
Importat de robot_telefonic_v2.py
"""

INTENTII_DOMENII = {
    "banking": [
        "problema_credit", "tranzactie_gresita", "card_blocat", "tranzactie_suspecta",
        "problema_transfer", "problema_schimb_valutar", "problema_sold", "card_pierdut"
    ],
    "medicina": [
        "rezultate_analize", "problema_reteta", "problema_asigurare", "reclamatie_personal",
        "consultatie_anulata", "problema_facturare", "problema_programare", "anulare_programare"
    ],
    "retail": [
        "produs_lipsa_stoc", "comanda_gresita", "problema_livrare", "problema_garantie",
        "reclamatie_produs", "anulare_comanda", "comanda_intarziata", "retur_produs"
    ],
    "telecom": [
        "problema_modificare_abonament", "portare_esuata", "problema_internet",
        "problema_roaming", "factura_gresita", "reziliere_contract",
        "activare_esuata", "problema_semnal"
    ],
    "servicii_publice": [
        "dosar_respins", "contestatie_decizie", "informatii_program", "reclamatie_serviciu",
        "sesizare_problema", "problema_plata_taxa", "acte_incomplete", "programare_ghiseu"
    ],
}

SALUT_DOMENII = {
    "banking":          "Bună ziua, ați sunat la serviciul clienți al băncii Transilvania.",
    "medicina":         "Bună ziua, ați sunat la clinica Romed Plus.",
    "retail":           "Bună ziua, ați sunat la serviciul clienți E-shop.",
    "telecom":          "Bună ziua, ați contactat serviciul clienți al Telenet.",
    "servicii_publice": "Bună ziua, ați sunat la serviciile publice de asistența agricolă.",
}

# Replici predefinite de identificare — cerute in ordine, fara LLM
REPLICI_IDENTIFICARE = {
    "banking": [
        "Vă rog să îmi spuneți numele complet.",
        "Vă mulțumesc. Și ultimele 4 cifre ale cardului dumneavoastră, vă rog.",
        "Vă mulțumesc pentru identificare. Cu ce vă pot ajuta astăzi?",
    ],
    "medicina": [
        "Vă rog să îmi spuneți numele complet.",
        "Vă mulțumesc. Și data nașterii dumneavoastră, vă rog.",
        "Vă mulțumesc pentru identificare. Cu ce vă pot ajuta astăzi?",
    ],
    "retail": [
        "Vă rog să îmi spuneți numele complet.",
        "Vă mulțumesc. Și adresa de email asociată contului dumneavoastră, vă rog.",
        "Vă mulțumesc pentru identificare. Cu ce vă pot ajuta astăzi?",
    ],
    "telecom": [
        "Vă rog să îmi spuneți numele complet.",
        "Vă mulțumesc. Și numărul de telefon al abonamentului dumneavoastră, vă rog.",
        "Vă mulțumesc pentru identificare. Cu ce vă pot ajuta astăzi?",
    ],
    "servicii_publice": [
        "Vă rog să îmi spuneți numele complet.",
        "Vă mulțumesc. Și numărul CNP sau al documentului de identitate, vă rog.",
        "Vă mulțumesc pentru identificare. Cu ce vă pot ajuta astăzi?",
    ],
}

REPLICA_INCHEIERE = "Vă mulțumim că ați contactat serviciul nostru. O zi bună!"
REPLICA_OPERATOR  = (
    "Solicitarea dumneavoastră depășește competențele sistemului automat. "
    "Vă voi transfera către un operator uman care vă poate ajuta. "
    "Vă rugăm să rămâneți în linie. O zi bună!"
)

NUME_DOMENII = {
    "banking": "bancar", "medicina": "medical", "retail": "retail",
    "telecom": "telecomunicatii", "servicii_publice": "servicii publice",
}

TIP_REZUMAT = {
    "pozitiv": {"tip": "SCURT",  "min_cuv": 20, "max_cuv": 40,  "propozitii": "1-2 propozitii"},
    "neutru":  {"tip": "MEDIU",  "min_cuv": 40, "max_cuv": 70,  "propozitii": "3-4 propozitii"},
    "negativ": {"tip": "LUNG",   "min_cuv": 60, "max_cuv": 100, "propozitii": "5-7 propozitii"},
}

# ─── DIFICULTATE ALEATOARE ────────────────────────────────────────────────────

NIVELURI_DIFICULTATE = {
    "simpla": {
        "pondere": 0.30,
        "instructiune": (
            "Problema clientului are o solutie simpla si imediata. "
            "Rezolv-o complet in aceasta conversatie. "
            "Ofera o solutie concreta si rapida. Tonul tau este eficient si pozitiv."
        ),
    },
    "medie": {
        "pondere": 0.25,
        "instructiune": (
            "Problema clientului are o solutie partiala. "
            "Poti rezolva doar o parte acum, restul necesita timp sau alt departament. "
            "Explica limitarile clar dar empatic. Tonul tau este profesionist."
        ),
    },
    "complexa": {
        "pondere": 0.45,
        "instructiune": (
            "Problema clientului nu are o solutie imediata completa. "
            "Insa ofera INTOTDEAUNA cel putin un pas concret pe care clientul il poate face acum "
            "(ex: trimitere email, numar de referinta, departament de contact, termen clar). "
            "Nu lasa clientul fara nicio actiune concreta. "
            "Tonul tau este empatic dar limitele sunt explicate clar."
        ),
    },
}


DETALII_NECESARE = {
    "problema_credit":               ["numărul contractului de credit sau CNP-ul"],
    "tranzactie_gresita":            ["data tranzacției", "suma tranzacției"],
    "card_blocat":                   ["ultimele 4 cifre ale cardului"],
    "tranzactie_suspecta":           ["data și suma tranzacției suspecte"],
    "problema_transfer":             ["data transferului", "suma și IBAN-ul destinatarului"],
    "problema_schimb_valutar":       ["data tranzacției valutare", "suma convertită"],
    "problema_sold":                 ["data pentru care verificați soldul"],
    "card_pierdut":                  ["ultimele 4 cifre ale cardului pierdut"],
    "rezultate_analize":             ["data recoltării analizelor"],
    "problema_reteta":               ["numele medicamentului", "data emiterii rețetei"],
    "problema_asigurare":            ["numărul poliței de asigurare"],
    "reclamatie_personal":           ["numele medicului sau asistentului", "data incidentului"],
    "consultatie_anulata":           ["data și ora consultației anulate"],
    "problema_facturare":            ["numărul facturii", "suma contestată"],
    "problema_programare":           ["data programării existente"],
    "anulare_programare":            ["data și ora programării de anulat"],
    "produs_lipsa_stoc":             ["numele produsului sau codul de produs"],
    "comanda_gresita":               ["numărul comenzii"],
    "problema_livrare":              ["numărul comenzii"],
    "problema_garantie":             ["numărul comenzii sau seria produsului"],
    "reclamatie_produs":             ["numărul comenzii", "descrierea problemei"],
    "anulare_comanda":               ["numărul comenzii"],
    "comanda_intarziata":            ["numărul comenzii"],
    "retur_produs":                  ["numărul comenzii", "motivul returului"],
    "problema_modificare_abonament": ["pachetul dorit"],
    "portare_esuata":                [],
    "problema_internet":             ["adresa unde se înregistrează problema"],
    "problema_roaming":              ["țara în care vă aflați"],
    "factura_gresita":               ["numărul facturii", "suma contestată"],
    "reziliere_contract":            [],
    "activare_esuata":               [],
    "problema_semnal":               ["adresa sau zona cu problemă de semnal"],
    "dosar_respins":                 ["numărul dosarului"],
    "contestatie_decizie":           ["numărul deciziei contestate"],
    "informatii_program":            [],
    "reclamatie_serviciu":           ["descrierea problemei", "data incidentului"],
    "sesizare_problema":             ["descrierea problemei"],
    "problema_plata_taxa":           ["data plății", "suma plătită"],
    "acte_incomplete":               ["numărul dosarului"],
    "programare_ghiseu":             ["motivul programării"],
    "alta_solicitare":               [],
}