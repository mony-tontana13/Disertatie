"""
Elimina duplicarea aproximativa din replicile clientului, cauzata de un bug in
zevo_stt_transcrie (text/text_pp adaugate ambele in loc ca text_pp sa il
inlocuiasca pe text). Proceseaza toate fisierele JSON dintr-un folder.

Tiparul observat: replica clientului contine fraza spusa de doua ori la rand,
uneori identic, uneori cu numerele scrise diferit intre cele doua aparitii
(prima oara cu litere, a doua oara cu cifre - varianta normalizata de Zevo).

Strategie: impartim cuvintele replicii in doua jumatati aproximativ egale si
verificam cat de similare sunt structural (ignorand reprezentarea cifrelor).
Daca similaritatea e suficient de mare, pastram doar a doua jumatate (de
obicei varianta finala/normalizata). Altfel, lasam replica neschimbata.

Utilizare:
    python3 elimina_duplicari.py --folder ./rezultate_robot/
    python3 elimina_duplicari.py --folder ./rezultate_robot/ --output ./rezultate_robot_curatate/
    python3 elimina_duplicari.py --fisier conversatie_X.json --dry_run
"""

import os
import re
import json
import argparse
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

CUVANT_LA_CIFRA = {
    "zero": "0", "unu": "1", "doi": "2", "două": "2", "trei": "3", "patru": "4",
    "cinci": "5", "șase": "6", "sase": "6", "șapte": "7", "sapte": "7",
    "opt": "8", "nouă": "9", "noua": "9",
}


def normalizeaza(text):
    """Lowercase, fara diacritice, pentru comparatie structurala."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().strip()


def cuvinte_la_cifre(text):
    """Inlocuieste cuvinte numerice cu cifre, pentru a compara 'doua' cu '2'."""
    cuvinte = normalizeaza(text).split()
    rezultat = [CUVANT_LA_CIFRA.get(c, c) for c in cuvinte]
    return rezultat


def similaritate(a_cuvinte, b_cuvinte):
    """Similaritate intre doua liste de cuvinte (0..1), invariant la cifre vs litere."""
    if not a_cuvinte or not b_cuvinte:
        return 0.0
    sm = SequenceMatcher(a=a_cuvinte, b=b_cuvinte)
    return sm.ratio()


def elimina_duplicare(text, prag_similaritate=0.65, min_cuvinte=2):
    """
    Detecteaza si elimina duplicarea aproximativa intr-o singura replica.
    Returneaza (text_curatat, a_fost_modificat).
    """
    cuvinte_originale = text.split()
    n = len(cuvinte_originale)

    if n < min_cuvinte * 2:
        return text, False

    candidati = []
    # Verificam toate impartirile posibile intr-o plaja larga din jurul mijlocului,
    # fiindca cele doua "jumatati" pot avea lungimi diferite (numere scrise cu
    # litere vs cifre, cuvinte de umplutura variabile intre text si text_pp)
    centru = n // 2
    plaja = max(5, n // 4)
    for offset in range(-plaja, plaja + 1):
        punct = centru + offset
        if punct < min_cuvinte or punct > n - min_cuvinte:
            continue
        prima = cuvinte_originale[:punct]
        a_doua = cuvinte_originale[punct:]

        prima_norm = cuvinte_la_cifre(" ".join(prima))
        a_doua_norm = cuvinte_la_cifre(" ".join(a_doua))

        sim = similaritate(prima_norm, a_doua_norm)
        echilibru = 1 - abs(len(prima) - len(a_doua)) / n
        scor = sim * 0.85 + echilibru * 0.15
        candidati.append((scor, sim, punct, prima, a_doua))

    if not candidati:
        return text, False

    candidati.sort(key=lambda c: c[0], reverse=True)
    best_scor, best_sim, best_punct, prima, a_doua = candidati[0]

    if best_sim >= prag_similaritate:
        # Pastram a doua jumatate (de obicei varianta finala/text_pp normalizata)
        text_curatat = " ".join(a_doua)
        return text_curatat, True

    return text, False


def proceseaza_conversatie(date, prag_similaritate=0.65):
    """Proceseaza toate replicile client dintr-o conversatie incarcata din JSON."""
    modificari = []
    for replica in date.get("conversatie", []):
        if replica.get("rol") == "client":
            text_original = replica["text"]
            text_curatat, modificat = elimina_duplicare(
                text_original, prag_similaritate=prag_similaritate
            )
            if modificat:
                modificari.append({
                    "original": text_original,
                    "curatat": text_curatat
                })
                replica["text"] = text_curatat
                replica["text_original_brut"] = text_original  # pastram pentru audit
    return date, modificari


def main():
    parser = argparse.ArgumentParser(
        description="Elimina duplicarea aproximativa din replicile client ale transcrierilor STT"
    )
    grup = parser.add_mutually_exclusive_group(required=True)
    grup.add_argument("--folder", help="Folder cu fisiere JSON de conversatii")
    grup.add_argument("--fisier", help="Un singur fisier JSON")
    parser.add_argument("--output", default=None,
                        help="Folder de output (default: suprascrie in acelasi loc, cu backup .bak)")
    parser.add_argument("--prag", type=float, default=0.65,
                        help="Prag de similaritate pentru a considera o replica duplicata (default 0.65)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Doar afiseaza ce s-ar modifica, fara a scrie fisiere")
    args = parser.parse_args()

    fisiere = (
        [Path(args.fisier)] if args.fisier
        else sorted(Path(args.folder).glob("*.json"))
    )

    if not fisiere:
        print("Niciun fisier JSON gasit.")
        return

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    total_modificari = 0
    total_fisiere_modificate = 0

    for fisier in fisiere:
        try:
            with open(fisier, encoding="utf-8") as f:
                date = json.load(f)
        except Exception as e:
            print(f"  [SKIP] {fisier.name} — eroare la citire: {e}")
            continue

        date_curatate, modificari = proceseaza_conversatie(date, prag_similaritate=args.prag)

        if modificari:
            total_fisiere_modificate += 1
            total_modificari += len(modificari)
            print(f"\n[{fisier.name}] — {len(modificari)} replici curatate:")
            for m in modificari:
                print(f"  INAINTE: {m['original'][:90]}...")
                print(f"  DUPA:    {m['curatat'][:90]}...")

            if not args.dry_run:
                cale_output = (
                    Path(args.output) / fisier.name if args.output
                    else fisier
                )
                if not args.output:
                    # backup inainte de suprascriere
                    backup = fisier.with_suffix(".json.bak")
                    if not backup.exists():
                        with open(backup, "w", encoding="utf-8") as f:
                            json.dump(date, f, ensure_ascii=False, indent=2)
                with open(cale_output, "w", encoding="utf-8") as f:
                    json.dump(date_curatate, f, ensure_ascii=False, indent=2)
        else:
            if args.output:
                # copiem neschimbat, ca sa avem tot setul complet in folderul de output
                cale_output = Path(args.output) / fisier.name
                if not args.dry_run:
                    with open(cale_output, "w", encoding="utf-8") as f:
                        json.dump(date, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMAR")
    print(f"{'='*60}")
    print(f"  Fisiere procesate:   {len(fisiere)}")
    print(f"  Fisiere modificate:  {total_fisiere_modificate}")
    print(f"  Replici curatate:    {total_modificari}")
    if args.dry_run:
        print(f"  (DRY RUN — nu s-a scris niciun fisier)")
    elif not args.output:
        print(f"  (Fisierele originale au backup cu extensia .json.bak)")


if __name__ == "__main__":
    main()