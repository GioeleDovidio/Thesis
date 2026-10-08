"""
aggiungi_giorni_gu.py
=====================
- Elimina la colonna 'Giorni_GU_Deposito' se presente
- Aggiunge 'Giorni_GU_Decisione' subito dopo 'Decisione_del'
  Valore = Decisione_del - prima_data_GU (in giorni)

Se Date_GU contiene più date (es. "16/04/2025 - 30/04/2025"),
usa la più vecchia (inizio effettivo del procedimento).
Se una delle date manca, il campo resta vuoto.

UTILIZZO:
  python aggiungi_giorni_gu.py --csv PRONUNCE_COMPLETE.csv
"""

import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime


def parse_data(s):
    try:
        return datetime.strptime(s.strip(), "%d/%m/%Y")
    except Exception:
        return None


def prima_data_gu(campo):
    """Estrae tutte le date dal campo Date_GU e restituisce la più vecchia."""
    date = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', campo or "")
    valide = [parse_data(d) for d in date]
    valide = [d for d in valide if d]
    return min(valide) if valide else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[!] File non trovato: {args.csv}"); sys.exit(1)

    # Leggi CSV
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        prima = f.readline()
        sep   = ";" if prima.count(";") >= prima.count(",") else ","
        f.seek(0)
        reader     = csv.DictReader(f, delimiter=sep)
        fieldnames = list(reader.fieldnames or [])
        righe      = list(reader)

    print(f"[i] Righe: {len(righe)} | Colonne: {len(fieldnames)}")

    # 1. Rimuovi Giorni_GU_Deposito se esiste
    if "Giorni_GU_Deposito" in fieldnames:
        fieldnames.remove("Giorni_GU_Deposito")
        for row in righe:
            row.pop("Giorni_GU_Deposito", None)
        print("[i] Colonna 'Giorni_GU_Deposito' rimossa")

    # 2. Aggiungi Giorni_GU_Decisione dopo Decisione_del
    col = "Giorni_GU_Decisione"
    if col in fieldnames:
        fieldnames.remove(col)
        print(f"[i] Colonna '{col}' già presente — ricalcolo")

    try:
        pos = fieldnames.index("Decisione_del") + 1
    except ValueError:
        pos = len(fieldnames)
    fieldnames.insert(pos, col)
    print(f"[i] Colonna '{col}' inserita in posizione {pos}")

    # 3. Calcola per ogni riga
    calcolati = 0
    vuoti     = 0
    for row in righe:
        decisione = parse_data(row.get("Decisione_del", ""))
        gu        = prima_data_gu(row.get("Date_GU", ""))

        if decisione and gu:
            row[col] = str((decisione - gu).days)
            calcolati += 1
        else:
            row[col] = ""
            vuoti += 1

    # 4. Salva con file temporaneo (sicuro su Windows)
    path_assoluto = os.path.abspath(args.csv)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path_assoluto), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=sep,
                                    quoting=csv.QUOTE_MINIMAL, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(righe)
        shutil.move(tmp, path_assoluto)
    except Exception as e:
        try: os.remove(tmp)
        except Exception: pass
        print(f"[!] Errore scrittura: {e}"); sys.exit(1)

    print(f"[OK] Calcolati: {calcolati} | Vuoti: {vuoti}")
    print(f"[OK] CSV salvato: {path_assoluto}")


if __name__ == "__main__":
    main()