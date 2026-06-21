"""
Aplica punctuatie (via GPT-4.1-mini) pe un cache STT existent, fara a re-rula transcrierea audio.
Util cand ai deja cache_stt.json si vrei doar varianta cu punctuatie, pentru comparatie.

Utilizare:
    export OPENAI_API_KEY="sk-..."
    python3 proceseaza_cache_punctuatie.py --input cache_stt_brut.json --output cache_stt_punctuatie.json
"""

import os
import json
import time
import argparse
from openai import OpenAI

client_gpt = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def adauga_punctuatie(text_brut):
    """Adauga punctuatie/majuscule pe transcrierea bruta, fara a modifica cuvintele."""
    if not text_brut or len(text_brut.split()) < 4:
        return text_brut, 0.0

    prompt = (
        "Urmatorul text este o transcriere automata (STT) continua, fara punctuatie "
        "si fara delimitare intre vorbitori. Adauga semne de punctuatie si majuscule "
        "corecte, pastrand EXACT aceleasi cuvinte. Nu adauga, elimina sau modifica "
        "informatii. Nu adauga etichete de tip CLIENT/OPERATOR daca nu poti distinge "
        "cu certitudine vorbitorii — daca nu poti, lasa textul ca un singur bloc "
        "continuu, doar cu punctuatie adaugata.\n\n"
        "TEXT BRUT:\n" + text_brut + "\n\n"
        "TEXT CU PUNCTUATIE:"
    )
    start = time.time()
    raspuns = ""
    stream = client_gpt.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=min(len(text_brut.split()) * 2, 1200),
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            raspuns += chunk.choices[0].delta.content
    lat = round(time.time() - start, 2)
    return (raspuns.strip() if raspuns.strip() else text_brut), lat


def main():
    parser = argparse.ArgumentParser(description="Adauga punctuatie pe un cache STT existent")
    parser.add_argument("--input", required=True, help="Cache STT brut (JSON)")
    parser.add_argument("--output", required=True, help="Unde se salveaza cache-ul procesat")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("EROARE: seteaza variabila de mediu OPENAI_API_KEY inainte de a rula.")
        return

    with open(args.input, encoding="utf-8") as f:
        cache = json.load(f)

    print(f"Incarcate {len(cache)} conversatii din {args.input}\n")

    rezultat = {}
    for i, (cale, info) in enumerate(cache.items(), 1):
        print(f"[{i}/{len(cache)}] {cale}...", end=" ", flush=True)
        text_brut = info["text"]
        text_procesat, lat_punct = adauga_punctuatie(text_brut)
        print(f"OK ({lat_punct}s)")

        rezultat[cale] = {
            "text": text_procesat,
            "text_brut": text_brut,
            "latenta_stt": info.get("latenta_stt", 0),
            "latenta_punctuatie": lat_punct,
            "punctuatie_aplicata": True,
            "domeniu": info.get("domeniu", "")
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rezultat, f, ensure_ascii=False, indent=2)

    lat_totala_punct = sum(v["latenta_punctuatie"] for v in rezultat.values())
    print(f"\nGata. Salvat in: {args.output}")
    print(f"Latenta totala adaugata de punctuatie: {round(lat_totala_punct, 2)}s "
          f"(medie {round(lat_totala_punct/len(rezultat), 2)}s/conversatie)")


if __name__ == "__main__":
    main()