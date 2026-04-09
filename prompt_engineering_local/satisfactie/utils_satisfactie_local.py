"""
Utilitare comune pentru toate versiunile de prompt - satisfactie modele locale.
Fiecare versiune are doua variante de prompt: get_prompt si get_prompt2.
"""
import json
import os
import random
import time
import unicodedata
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

MODELE = {
    "romistral": "OpenLLM-Ro/RoMistral-7b-Instruct",
    "rogemma": "OpenLLM-Ro/RoGemma-7b-Instruct"
}

CLASE = ["pozitiv", "neutru", "negativ"]

# Exemple scurte — o replica client
EXEMPLE_SCURTE = {
    "pozitiv": (
        "CLIENT: Buna ziua, am o problema cu cardul meu blocat.\n"
        "OPERATOR: Va deblochez imediat. Gata, cardul este activ.\n"
        "CLIENT: Minunat, multumesc mult, chiar ma ajutati!",
        "pozitiv"
    ),
    "neutru": (
        "CLIENT: Buna ziua, vreau sa stiu statusul comenzii mele.\n"
        "OPERATOR: Comanda va ajunge maine intre 10 si 14.\n"
        "CLIENT: Ok, am inteles. Pa.",
        "neutru"
    ),
    "negativ": (
        "CLIENT: Sunt la a treia tentativa sa rezolv aceasta problema.\n"
        "OPERATOR: Inteleg, va transferam la un specialist.\n"
        "CLIENT: Bine, ce sa fac... transferati.",
        "negativ"
    ),
}

# Exemple lungi — ilustreaza granita neutru/negativ mai explicit
EXEMPLE_LUNGI = {
    "pozitiv": (
        "CLIENT: Buna ziua, am observat o taxa dubla pe factura mea.\n"
        "OPERATOR: Va verific factura. Da, aveti dreptate, a fost o eroare. Va restitui suma in 24 de ore.\n"
        "CLIENT: Perfect, exact asta aveam nevoie. Multumesc foarte mult, sunteti promti!\n"
        "OPERATOR: Cu placere, o zi buna!\n"
        "CLIENT: Si dumneavoastra, la revedere!",
        "pozitiv"
    ),
    "neutru": (
        "CLIENT: Buna ziua, comanda mea nu a ajuns la timp.\n"
        "OPERATOR: Imi pare rau, am verificat si va fi livrata maine.\n"
        "CLIENT: Bine, maine merge si asa.\n"
        "OPERATOR: Va multumim pentru intelegere.\n"
        "CLIENT: Da, la revedere.",
        "neutru"
    ),
    "negativ": (
        "CLIENT: Buna ziua, am sunat a doua oara pentru aceeasi problema cu internetul.\n"
        "OPERATOR: Imi pare rau, va pot programa un tehnician pentru joi.\n"
        "CLIENT: Joi... bine, ce sa fac, nu am de ales.\n"
        "OPERATOR: Va multumim pentru rabdare.\n"
        "CLIENT: Da... la revedere.",
        "negativ"
    ),
}


def normalizeaza(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().strip()


def extrage_satisfactie(raspuns):
    raspuns_norm = normalizeaza(raspuns)
    for clasa in CLASE:
        if clasa in raspuns_norm[:50]:
            return clasa
    for clasa in CLASE:
        if clasa in raspuns_norm:
            return clasa
    return "necunoscut"


def selecteaza_subset(folder, n_per_domeniu=2):
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
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
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


def genereaza_raspuns(tokenizer, model, device, prompt, max_new_tokens=20):
    import torch
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
    rezultate = []
    for i, conv in enumerate(conversatii):
        conv_id = conv["id"]
        domeniu = conv.get("domeniu", "banking")
        complexitate = conv.get("complexitate", "?")
        satisfactie_gold = conv.get("satisfactie", "necunoscut")
        dialog = "\n".join([r["rol"].upper() + ": " + r["text"] for r in conv["conversatie"]])

        prompt = get_prompt_fn(dialog)
        raspuns_brut, latenta = genereaza_raspuns(tokenizer, model, device, prompt)
        satisfactie_pred = extrage_satisfactie(raspuns_brut)
        corecta = satisfactie_pred == satisfactie_gold
        status = "OK" if corecta else "GRESIT"

        print(f"  [{i+1:03d}] {conv_id} [{complexitate}] | gold: {satisfactie_gold} | pred: {satisfactie_pred} | {status} | {latenta:.1f}s")
        if not corecta:
            print(f"    brut: {raspuns_brut[:80]}")

        rezultate.append({
            "id": conv_id, "domeniu": domeniu, "complexitate": complexitate,
            "satisfactie_gold": satisfactie_gold, "satisfactie_pred": satisfactie_pred,
            "corecta": corecta, "raspuns_brut": raspuns_brut,
            "latenta": round(latenta, 2), "versiune_prompt": versiune, "model": nume_model
        })
    return rezultate


def calculeaza_si_afiseaza(rezultate, versiune, nume_model):
    gold = [r["satisfactie_gold"] for r in rezultate]
    pred = [r["satisfactie_pred"] for r in rezultate]
    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred, average="macro", zero_division=0, labels=CLASE)
    latenta_medie = sum(r["latenta"] for r in rezultate) / len(rezultate)

    linii = []
    linii.append(f"\n=== {nume_model} | Prompt {versiune} | {len(rezultate)} conversatii ===")
    linii.append(f"  Accuracy:         {acc:.2%}")
    linii.append(f"  F1 Macro:         {f1:.3f}")
    linii.append(f"  Latenta medie:    {latenta_medie:.1f}s")
    linii.append(f"  Distributie gold: {dict(Counter(gold))}")
    linii.append(f"  Distributie pred: {dict(Counter(pred))}")

    erori = [r for r in rezultate if not r["corecta"]]
    if erori:
        linii.append(f"  Erori ({len(erori)}/{len(rezultate)}):")
        for e in erori:
            linii.append(f"    {e['id']} [{e['complexitate']}]: gold={e['satisfactie_gold']} pred={e['satisfactie_pred']}")

    linii.append(f"  Per clasa:")
    for clasa in CLASE:
        rez_c = [r for r in rezultate if r["satisfactie_gold"] == clasa]
        if rez_c:
            acc_c = sum(1 for r in rez_c if r["corecta"]) / len(rez_c)
            linii.append(f"    {clasa:<10}: {acc_c:.0%} ({sum(1 for r in rez_c if r['corecta'])}/{len(rez_c)})")

    raport_text = "\n".join(linii)
    print(raport_text)

    return {
        "model": nume_model, "versiune": versiune,
        "accuracy": round(acc, 4), "f1": round(f1, 4),
        "latenta": round(latenta_medie, 2),
        "nr_conversatii": len(rezultate),
        "raport_text": raport_text
    }
