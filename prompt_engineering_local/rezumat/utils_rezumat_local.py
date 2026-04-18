"""
Utilitare comune pentru toate versiunile de prompt - rezumat modele locale.
"""
import json
import os
import random
import time
import unicodedata
from collections import Counter

MODELE = {
    "romistral": "OpenLLM-Ro/RoMistral-7b-Instruct",
    "rogemma": "OpenLLM-Ro/RoGemma-7b-Instruct"
}

TIP_REZUMAT = {
    "pozitiv": {"tip": "SCURT",  "min_cuv": 20, "max_cuv": 40,  "propozitii": "1-2 propozitii"},
    "neutru":  {"tip": "MEDIU",  "min_cuv": 40, "max_cuv": 70,  "propozitii": "3-4 propozitii"},
    "negativ": {"tip": "LUNG",   "min_cuv": 60, "max_cuv": 100, "propozitii": "5-7 propozitii"},
}


def get_tip_rezumat(satisfactie):
    return TIP_REZUMAT.get(satisfactie, TIP_REZUMAT["neutru"])


def normalizeaza(text):
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower().strip()


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


def genereaza_raspuns(tokenizer, model, device, prompt, max_new_tokens=150):
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


def calculeaza_rouge(predictii, referinte):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        r1, r2, rl = [], [], []
        for pred, ref in zip(predictii, referinte):
            if pred and ref:
                s = scorer.score(ref, pred)
                r1.append(s["rouge1"].fmeasure)
                r2.append(s["rouge2"].fmeasure)
                rl.append(s["rougeL"].fmeasure)
        if not r1:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        return {
            "rouge1": round(sum(r1)/len(r1), 4),
            "rouge2": round(sum(r2)/len(r2), 4),
            "rougeL": round(sum(rl)/len(rl), 4)
        }
    except ImportError:
        print("  ATENTIE: rouge_score nu e instalat. Ruleaza: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def calculeaza_bertscore(predictii, referinte):
    try:
        from bert_score import score as bert_score
        perechi = [(p, r) for p, r in zip(predictii, referinte) if p and r]
        if not perechi:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        pred_list, ref_list = zip(*perechi)
        print("    Calculez BERTScore (prima rulare poate dura mai mult)...")
        P, R, F1 = bert_score(list(pred_list), list(ref_list), lang="ro", verbose=False)
        return {
            "precision": round(P.mean().item(), 4),
            "recall": round(R.mean().item(), 4),
            "f1": round(F1.mean().item(), 4)
        }
    except ImportError:
        print("  ATENTIE: bert_score nu e instalat. Ruleaza: pip install bert-score")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def ruleaza_evaluare(conversatii, tokenizer, model, device, get_prompt_fn, versiune, nume_model, results_dir):
    rezultate = []

    for i, conv in enumerate(conversatii):
        conv_id = conv["id"]
        domeniu = conv.get("domeniu", "banking")
        complexitate = conv.get("complexitate", "?")
        satisfactie = conv.get("satisfactie", "neutru")
        rezumat_gold = conv.get("rezumat", "")
        dialog = "\n".join([r["rol"].upper() + ": " + r["text"] for r in conv["conversatie"]])

        tip_info = get_tip_rezumat(satisfactie)
        prompt = get_prompt_fn(dialog, satisfactie)
        raspuns_brut, latenta = genereaza_raspuns(tokenizer, model, device, prompt)

        nr_cuvinte = len(raspuns_brut.split())
        in_limite = tip_info["min_cuv"] <= nr_cuvinte <= tip_info["max_cuv"]
        status_lungime = "OK" if in_limite else f"LUNGIME ({nr_cuvinte} cuv)"

        print(f"  [{i+1:03d}] {conv_id} [{complexitate}/{satisfactie}] | {tip_info['tip']} | {nr_cuvinte} cuv | {status_lungime} | {latenta:.1f}s")

        rezultate.append({
            "id": conv_id, "domeniu": domeniu, "complexitate": complexitate,
            "satisfactie": satisfactie, "tip_rezumat": tip_info["tip"],
            "rezumat_gold": rezumat_gold, "rezumat_pred": raspuns_brut,
            "nr_cuvinte_gold": len(rezumat_gold.split()),
            "nr_cuvinte_pred": nr_cuvinte,
            "in_limite_lungime": in_limite,
            "latenta": round(latenta, 2),
            "versiune_prompt": versiune, "model": nume_model
        })

    return rezultate


def calculeaza_si_afiseaza(rezultate, versiune, nume_model):
    predictii = [r["rezumat_pred"] for r in rezultate]
    referinte = [r["rezumat_gold"] for r in rezultate]

    rouge = calculeaza_rouge(predictii, referinte)
    bertscore = calculeaza_bertscore(predictii, referinte)
    latenta_medie = sum(r["latenta"] for r in rezultate) / len(rezultate)
    nr_cuv_medie = sum(r["nr_cuvinte_pred"] for r in rezultate) / len(rezultate)
    in_limite = sum(1 for r in rezultate if r["in_limite_lungime"])

    linii = []
    linii.append(f"\n=== {nume_model} | Prompt {versiune} | {len(rezultate)} conversatii ===")
    linii.append(f"  ROUGE-1:          {rouge['rouge1']:.4f}")
    linii.append(f"  ROUGE-2:          {rouge['rouge2']:.4f}")
    linii.append(f"  ROUGE-L:          {rouge['rougeL']:.4f}")
    linii.append(f"  BERTScore F1:     {bertscore['f1']:.4f}")
    linii.append(f"  Latenta medie:    {latenta_medie:.1f}s")
    linii.append(f"  Cuvinte medii:    {nr_cuv_medie:.1f}")
    linii.append(f"  In limite:        {in_limite}/{len(rezultate)}")

    linii.append(f"  Per tip rezumat:")
    for tip in ["SCURT", "MEDIU", "LUNG"]:
        rez_t = [r for r in rezultate if r["tip_rezumat"] == tip]
        if rez_t:
            pred_t = [r["rezumat_pred"] for r in rez_t]
            ref_t = [r["rezumat_gold"] for r in rez_t]
            rouge_t = calculeaza_rouge(pred_t, ref_t)
            cuv_t = sum(r["nr_cuvinte_pred"] for r in rez_t) / len(rez_t)
            linii.append(f"    {tip:<8}: ROUGE-1={rouge_t['rouge1']:.3f} | ROUGE-L={rouge_t['rougeL']:.3f} | {cuv_t:.0f} cuv medii | n={len(rez_t)}")

    raport_text = "\n".join(linii)
    print(raport_text)

    return {
        "model": nume_model, "versiune": versiune,
        "rouge1": rouge["rouge1"], "rouge2": rouge["rouge2"], "rougeL": rouge["rougeL"],
        "bertscore_f1": bertscore["f1"],
        "latenta": round(latenta_medie, 2),
        "nr_cuvinte_medii": round(nr_cuv_medie, 1),
        "in_limite": in_limite,
        "nr_conversatii": len(rezultate),
        "raport_text": raport_text
    }
