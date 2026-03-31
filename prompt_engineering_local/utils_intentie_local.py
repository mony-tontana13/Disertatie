import json
import os
import random
import time
import unicodedata
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODELE = {
    "romistral": "OpenLLM-Ro/RoMistral-7b-Instruct",
    "rogemma": "OpenLLM-Ro/RoGemma-7b-Instruct"
}

INTENTII_DOMENII = {
    "banking": ["problema_credit","tranzactie_gresita","card_blocat","tranzactie_suspecta","problema_transfer","problema_schimb_valutar","problema_sold","card_pierdut"],
    "medicina": ["rezultate_analize","problema_reteta","problema_asigurare","reclamatie_personal","consultatie_anulata","problema_facturare","problema_programare","anulare_programare"],
    "retail": ["produs_lipsa_stoc","comanda_gresita","problema_livrare","problema_garantie","reclamatie_produs","anulare_comanda","comanda_intarziata","retur_produs"],
    "telecom": ["problema_modificare_abonament","portare_esuata","problema_internet","problema_roaming","factura_gresita","reziliere_contract","activare_esuata","problema_semnal"],
    "servicii_publice": ["dosar_respins","contestatie_decizie","informatii_program","reclamatie_serviciu","sesizare_problema","problema_plata_taxa","acte_incomplete","programare_ghiseu"]
}

EXEMPLE_FEWSHOT = {
    "banking": [
        ("CLIENT: Buna ziua, am o problema cu cardul meu, l-am pierdut ieri.", "card_pierdut"),
        ("CLIENT: Vreau sa stiu de ce mi-a aparut o tranzactie pe care nu am facut-o.", "tranzactie_suspecta"),
    ],
    "medicina": [
        ("CLIENT: Buna ziua, am facut analize saptamana trecuta si vreau sa stiu rezultatele.", "rezultate_analize"),
        ("CLIENT: Am o programare joi dar nu mai pot veni, vreau sa o anulez.", "anulare_programare"),
    ],
    "retail": [
        ("CLIENT: Am primit o comanda gresita, mi-au venit alte produse decat ce am comandat.", "comanda_gresita"),
        ("CLIENT: Pachetul meu nu a ajuns inca si a trecut termenul de livrare.", "problema_livrare"),
    ],
    "telecom": [
        ("CLIENT: Vreau sa imi schimb abonamentul dar nu reusesc, am incercat de mai multe ori.", "problema_modificare_abonament"),
        ("CLIENT: Am o factura mult mai mare decat de obicei si nu inteleg de ce.", "factura_gresita"),
    ],
    "servicii_publice": [
        ("CLIENT: Dosarul meu a fost respins si nu inteleg motivul.", "dosar_respins"),
        ("CLIENT: Vreau sa fac o programare la ghiseu pentru un act.", "programare_ghiseu"),
    ]
}


def normalizeaza(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().strip()


def extrage_intentie(raspuns, domeniu):
    intentii_valide = INTENTII_DOMENII.get(domeniu, [])
    raspuns_norm = normalizeaza(raspuns)
    intentii_norm = {normalizeaza(i): i for i in intentii_valide}
    for intentie_norm, intentie_orig in intentii_norm.items():
        if intentie_norm in raspuns_norm:
            return intentie_orig
    return "alta_solicitare"


def selecteaza_subset(folder, n_per_domeniu=2):
    """Selecteaza n conversatii per domeniu (1 simpla + 1 complexa)."""
    random.seed(42)
    subset = []
    domenii = ["banking", "medicina", "retail", "telecom", "servicii_publice"]
    for domeniu in domenii:
        domeniu_path = os.path.join(folder, domeniu)
        if not os.path.isdir(domeniu_path):
            print(f"  ATENTIE: folder lipsa pentru {domeniu}")
            continue
        fisiere = sorted([f for f in os.listdir(domeniu_path) if f.endswith(".json")])
        simple, complexe = [], []
        for fisier in fisiere:
            with open(os.path.join(domeniu_path, fisier), encoding="utf-8") as f:
                conv = json.load(f)
            if conv.get("complexitate") == "simpla":
                simple.append(conv)
            else:
                complexe.append(conv)
        selectate = []
        if simple:
            selectate += random.sample(simple, min(n_per_domeniu // 2 + n_per_domeniu % 2, len(simple)))
        if complexe:
            selectate += random.sample(complexe, min(n_per_domeniu // 2, len(complexe)))
        subset.extend(selectate[:n_per_domeniu])
        print(f"  {domeniu}: {len(selectate[:n_per_domeniu])} conversatii")
    return subset


def incarca_toate_conversatiile(folder):
    """Incarca toate conversatiile din toate domeniile."""
    conversatii = []
    domenii = ["banking", "medicina", "retail", "telecom", "servicii_publice"]
    for domeniu in domenii:
        domeniu_path = os.path.join(folder, domeniu)
        if not os.path.isdir(domeniu_path):
            continue
        fisiere = sorted([f for f in os.listdir(domeniu_path) if f.endswith(".json")])
        for fisier in fisiere:
            with open(os.path.join(domeniu_path, fisier), encoding="utf-8") as f:
                conversatii.append(json.load(f))
        print(f"  {domeniu}: {len(fisiere)} conversatii")
    return conversatii


def incarca_model(nume_model):
    model_path = MODELE[nume_model]
    print(f"Se incarca {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)
    print(f"Model incarcat pe {device}")
    return tokenizer, model, device


def genereaza_raspuns(tokenizer, model, device, prompt, max_new_tokens=40):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    latenta = time.time() - start
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_length:]
    raspuns = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raspuns, latenta


def ruleaza_evaluare(conversatii, tokenizer, model, device, get_prompt_fn, versiune, nume_model, results_dir):
    """Ruleaza evaluarea si returneaza rezultatele finale."""
    rezultate = []

    for i, conv in enumerate(conversatii):
        conv_id = conv["id"]
        domeniu = conv.get("domeniu", "banking")
        complexitate = conv.get("complexitate", "?")
        dialog = "\n".join([r["rol"].upper() + ": " + r["text"] for r in conv["conversatie"]])

        intentie_gold = conv.get("intentie_gold", ["alta_solicitare"])
        if isinstance(intentie_gold, list):
            intentie_gold = intentie_gold[0]

        prompt = get_prompt_fn(dialog, domeniu)
        raspuns_brut, latenta = genereaza_raspuns(tokenizer, model, device, prompt)
        intentie_pred = extrage_intentie(raspuns_brut, domeniu)
        corecta = intentie_pred == intentie_gold
        status = "OK" if corecta else "GRESIT"

        print(f"  [{i+1:03d}] {conv_id} [{complexitate}] | gold: {intentie_gold} | pred: {intentie_pred} | {status} | {latenta:.0f}s")
        if not corecta:
            print(f"    brut: {raspuns_brut[:80]}")

        rezultate.append({
            "id": conv_id, "domeniu": domeniu, "complexitate": complexitate,
            "intentie_gold": intentie_gold, "intentie_pred": intentie_pred,
            "corecta": corecta, "raspuns_brut": raspuns_brut,
            "latenta": round(latenta, 2), "versiune_prompt": versiune, "model": nume_model
        })

    return rezultate


def calculeaza_si_afiseaza(rezultate, versiune, nume_model):
    gold = [r["intentie_gold"] for r in rezultate]
    pred = [r["intentie_pred"] for r in rezultate]
    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred, average="macro", zero_division=0)
    latenta_medie = sum(r["latenta"] for r in rezultate) / len(rezultate)

    linii = []
    linii.append(f"\n=== {nume_model} | Prompt {versiune} | {len(rezultate)} conversatii ===")
    linii.append(f"  Accuracy:      {acc:.2%}")
    linii.append(f"  F1 Macro:      {f1:.3f}")
    linii.append(f"  Latenta medie: {latenta_medie:.1f}s")

    erori = [r for r in rezultate if not r["corecta"]]
    if erori:
        linii.append(f"  Erori ({len(erori)}/{len(rezultate)}):")
        for e in erori:
            linii.append(f"    {e['id']} [{e['complexitate']}]: gold={e['intentie_gold']} pred={e['intentie_pred']}")

    linii.append(f"  Per domeniu:")
    per_domeniu = {}
    for domeniu in ["banking", "medicina", "retail", "telecom", "servicii_publice"]:
        rez_d = [r for r in rezultate if r["domeniu"] == domeniu]
        if rez_d:
            acc_d = sum(1 for r in rez_d if r["corecta"]) / len(rez_d)
            per_domeniu[domeniu] = round(acc_d, 4)
            linii.append(f"    {domeniu:<22}: {acc_d:.0%} ({sum(1 for r in rez_d if r['corecta'])}/{len(rez_d)})")

    raport_text = "\n".join(linii)
    print(raport_text)

    return {
        "model": nume_model, "versiune": versiune,
        "accuracy": round(acc, 4), "f1": round(f1, 4),
        "latenta": round(latenta_medie, 2),
        "nr_conversatii": len(rezultate),
        "per_domeniu": per_domeniu,
        "raport_text": raport_text
    }