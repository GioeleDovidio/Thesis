"""
split_date_gu.py
================
Per ogni riga con più date nel campo Date_GU (separate da ' - '),
crea una riga per ogni data mantenendo tutti gli altri campi identici.

Esempio:
  PRIMA:  84 | 2026 | 16/04/2025 - 30/04/2025 | ...
  DOPO:   84 | 2026 | 16/04/2025              | ...   ← riga originale
          84 | 2026 | 30/04/2025              | ...   ← riga nuova

UTILIZZO:
  python split_date_gu.py --csv PRONUNCE_COMPLETE.csv
  python split_date_gu.py --csv PRONUNCE_COMPLETE.csv --output split.csv
"""

import argparse
import copy
import csv
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    required=True)
    parser.add_argument("--output", default=None,
                        help="File di output (default: <nome>_split.csv)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[!] File non trovato: {args.csv}"); sys.exit(1)

    # Determina file di output
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.csv)
        out_path  = base + "_split" + ext

    # Leggi CSV
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        prima      = f.readline()
        sep        = ";" if prima.count(";") >= prima.count(",") else ","
        f.seek(0)
        reader     = csv.DictReader(f, delimiter=sep)
        fieldnames = list(reader.fieldnames or [])
        righe      = list(reader)

    print(f"[i] Righe originali: {len(righe)}")
    print(f"[i] Separatore: '{sep}'")

    # Costruisce le righe di output
    righe_output  = []
    righe_sdoppiate = 0
    righe_nuove     = 0

    for row in righe:
        campo_gu = row.get("Date_GU", "")

        # Controlla se ci sono più date
        if " - " not in campo_gu:
            # Riga singola: passa invariata
            righe_output.append(row)
            continue

        # Separa le date
        date = [d.strip() for d in campo_gu.split(" - ") if d.strip()]

        # Riga originale: tiene solo la prima data
        riga_originale = copy.copy(row)
        riga_originale["Date_GU"] = date[0]
        righe_output.append(riga_originale)

        # Righe nuove: una per ogni data successiva
        for data_extra in date[1:]:
            nuova_riga = copy.copy(row)
            nuova_riga["Date_GU"] = data_extra
            righe_output.append(nuova_riga)
            righe_nuove += 1

        righe_sdoppiate += 1

    print(f"[i] Righe con più date sdoppiate: {righe_sdoppiate}")
    print(f"[i] Righe nuove create: {righe_nuove}")
    print(f"[i] Righe totali output: {len(righe_output)}")

    # Scrivi output
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=sep,
                                quoting=csv.QUOTE_MINIMAL, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(righe_output)

    print(f"[OK] Salvato: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()