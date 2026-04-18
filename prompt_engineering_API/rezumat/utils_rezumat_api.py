"""
Utilitare comune pentru toate versiunile de prompt - rezumat modele API.
"""
import json
import os
import random
import time
import unicodedata

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

TIP_REZUMAT = {
    "pozitiv": {"tip": "SCURT",  "min_cuv": 20, "max_cuv": 40,  "propozitii": "1-2 propozitii"},
    "neutru":  {"tip": "MEDIU",  "min_cuv": 40, "max_cuv": 70,  "propozitii": "3-4 propozitii"},
    "negativ": {"tip": "LUNG",   "min_cuv": 60, "max_cuv": 100, "propozitii": "5-7 propozitii"},
}


def get_tip_rezumat(satisfactie):
    return TIP_REZUMAT.get(satisfactie, TIP_REZUMAT["neutru"])


def selecteaza_subset(folder, n_per_domeniu=2):
    random.seed(42)
    subset = []
    domenii = ["banking", "medicina", "retail", "telecom", "servicii_publice"]
    for domeniu in domenii:
        domeniu_path = os.path.join(folder, domeniu)
        if not os.path.isdir(domeniu_path):
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


def call_gpt(prompt):
    if not OPENAI_API_KEY:
        return "", 0.0, 0.0
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    start_total = time.time()
    ttft = None
    raspuns_complet = ""
    stream = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        stream=True
    )
    for chunk in stream:
        if ttft is None and chunk.choices[0].delta.content:
            ttft = time.time() - start_total
        if chunk.choices[0].delta.content:
            raspuns_complet += chunk.choices[0].delta.content
    latenta_totala = time.time() - start_total
    return raspuns_complet.strip(), round(ttft or latenta_totala, 3), round(latenta_totala, 3)


def call_gemini(prompt):
    if not GEMINI_API_KEY:
        return "", 0.0, 0.0
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    start_total = time.time()
    ttft = None
    raspuns_complet = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt
    ):
        if ttft is None and chunk.text:
            ttft = time.time() - start_total
        if chunk.text:
            raspuns_complet += chunk.text
    latenta_totala = time.time() - start_total
    return raspuns_complet.strip(), round(ttft or latenta_totala, 3), round(latenta_totala, 3)


def call_cohere(prompt):
    if not COHERE_API_KEY:
        return "", 0.0, 0.0
    import cohere
    co = cohere.ClientV2(api_key=COHERE_API_KEY)
    start_total = time.time()
    response = co.chat(
        model="command-r7b-12-2024",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    ttft = time.time() - start_total
    return response.message.content[0].text.strip(), round(ttft, 3), round(ttft, 3)


MODELE_API = {
    "GPT-4.1-mini": call_gpt,
    "Gemini-2.5-flash": call_gemini,
    "command-r7b-12-2024": call_cohere
}


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
        return {"rouge1": round(sum(r1)/len(r1), 4), "rouge2": round(sum(r2)/len(r2), 4), "rougeL": round(sum(rl)/len(rl), 4)}
    except ImportError:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def calculeaza_bertscore(predictii, referinte):
    try:
        from bert_score import score as bert_score
        perechi = [(p, r) for p, r in zip(predictii, referinte) if p and r]
        if not perechi:
            return {"f1": 0.0}
        pred_list, ref_list = zip(*perechi)
        print("    Calculez BERTScore...")
        P, R, F1 = bert_score(list(pred_list), list(ref_list), lang="ro", verbose=False)
        return {"precision": round(P.mean().item(), 4), "recall": round(R.mean().item(), 4), "f1": round(F1.mean().item(), 4)}
    except ImportError:
        return {"f1": 0.0}


def ruleaza_evaluare_api(conversatii, get_prompt_fn, versiune, results_dir):
    disponibile = []
    if OPENAI_API_KEY: disponibile.append("GPT-4.1-mini")
    if GEMINI_API_KEY: disponibile.append("Gemini-2.5-flash")
    if COHERE_API_KEY: disponibile.append("command-r7b-12-2024")

    if not disponibile:
        print("Nicio cheie API setata.")
        return {}

    print(f"Modele disponibile: {', '.join(disponibile)}")
    toate_rezultatele = {}

    for nume_model in disponibile:
        func_model = MODELE_API[nume_model]
        print(f"\n--- {nume_model} | {versiune} ---")
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
            raspuns_brut, ttft, latenta_totala = func_model(prompt)

            nr_cuvinte = len(raspuns_brut.split())
            in_limite = tip_info["min_cuv"] <= nr_cuvinte <= tip_info["max_cuv"]
            status = "OK" if in_limite else f"LUNGIME ({nr_cuvinte} cuv)"

            print(f"  [{i+1:03d}] {conv_id} [{complexitate}/{satisfactie}] | {tip_info['tip']} | {nr_cuvinte} cuv | {status} | TTFT: {ttft:.2f}s")

            rezultate.append({
                "id": conv_id, "domeniu": domeniu, "complexitate": complexitate,
                "satisfactie": satisfactie, "tip_rezumat": tip_info["tip"],
                "rezumat_gold": rezumat_gold, "rezumat_pred": raspuns_brut,
                "nr_cuvinte_gold": len(rezumat_gold.split()),
                "nr_cuvinte_pred": nr_cuvinte,
                "in_limite_lungime": in_limite,
                "ttft": ttft, "latenta_totala": latenta_totala,
                "versiune_prompt": versiune, "model": nume_model
            })

        toate_rezultatele[nume_model] = rezultate

    return toate_rezultatele


def calculeaza_si_afiseaza_api(toate_rezultatele, versiune):
    toate_metrici = []

    for nume_model, rezultate in toate_rezultatele.items():
        predictii = [r["rezumat_pred"] for r in rezultate]
        referinte = [r["rezumat_gold"] for r in rezultate]

        rouge = calculeaza_rouge(predictii, referinte)
        bertscore = calculeaza_bertscore(predictii, referinte)
        ttft_medie = sum(r["ttft"] for r in rezultate) / len(rezultate)
        latenta_medie = sum(r["latenta_totala"] for r in rezultate) / len(rezultate)
        nr_cuv_medie = sum(r["nr_cuvinte_pred"] for r in rezultate) / len(rezultate)
        in_limite = sum(1 for r in rezultate if r["in_limite_lungime"])

        linii = []
        linii.append(f"\n=== {nume_model} | Prompt {versiune} | {len(rezultate)} conversatii ===")
        linii.append(f"  ROUGE-1:          {rouge['rouge1']:.4f}")
        linii.append(f"  ROUGE-2:          {rouge['rouge2']:.4f}")
        linii.append(f"  ROUGE-L:          {rouge['rougeL']:.4f}")
        linii.append(f"  BERTScore F1:     {bertscore['f1']:.4f}")
        linii.append(f"  TTFT medie:       {ttft_medie:.3f}s")
        linii.append(f"  Latenta medie:    {latenta_medie:.3f}s")
        linii.append(f"  Cuvinte medii:    {nr_cuv_medie:.1f}")
        linii.append(f"  In limite:        {in_limite}/{len(rezultate)}")

        raport_text = "\n".join(linii)
        print(raport_text)

        toate_metrici.append({
            "model": nume_model, "versiune": versiune,
            "rouge1": rouge["rouge1"], "rouge2": rouge["rouge2"], "rougeL": rouge["rougeL"],
            "bertscore_f1": bertscore["f1"],
            "ttft_medie": round(ttft_medie, 3),
            "latenta_medie": round(latenta_medie, 3),
            "nr_cuvinte_medii": round(nr_cuv_medie, 1),
            "in_limite": in_limite,
            "nr_conversatii": len(rezultate),
            "raport_text": raport_text
        })

    return toate_metrici
