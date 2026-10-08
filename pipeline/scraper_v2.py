"""
Scraper Corte Costituzionale - Giudizio di legittimità in via incidentale
=========================================================================

UTILIZZO:
  python scraper_v2.py                # scarica tutti gli anni (1956-2026)
  python scraper_v2.py 2024           # scarica solo l'anno 2024
  python scraper_v2.py 2020 2024      # scarica dal 2020 al 2024 inclusi

PREREQUISITO:
  Esegui prima build_gu.py per costruire gu_index.csv.
  Lo scraper funziona anche senza (Date_GU sarà vuota per i casi senza
  data esplicita nel testo), ma con l'indice è completo.

OUTPUT (pronunce_raw.jsonl), per ogni pronuncia:
  metadati + Date_GU + Note_GU + testo integrale della pronuncia (campo "testo",
  dalla formula "ha pronunciato la seguente SENTENZA/ORDINANZA" fino alla fine).

LOGICA DATE GU (nessuna richiesta al sito GU durante lo scraping):
  0. Il testo viene ripulito dagli spazi e tagliato all'EPIGRAFE (dall'inizio del
     giudizio al primo "Visto/Udito/Ritenuto/Considerato/per questi motivi"):
     le GU delle ordinanze di rimessione stanno tutte lì.
  1. Data già scritta nel testo → usala direttamente
     Es: "n. 60 del 1 marzo 1978"  oppure  "n. 146 del 14/06/1956"
  2. Solo numero GU + anno nel testo → lookup in gu_index.csv (O(1))
     Es: "n. 72 dell'anno 1976"
     Es: "n. 15, prima serie speciale, dell'anno 2025"
  I fascicoli citati ma assenti dall'indice finiscono in Note_GU.
"""

import csv
import json
import math
import os
import random
import re
import sys
import time
from datetime import date

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
import undetected_chromedriver as uc

# ── Configurazione ────────────────────────────────────────────────────────────
SEARCH_URL        = "https://www.cortecostituzionale.it/ricerca-pronunce"
LIST_BASE_URL     = "https://www.cortecostituzionale.it/elenco-pronunce"
OUTPUT_JSONL      = "pronunce_raw.jsonl"
GU_INDEX_CSV      = "gu_index.csv"
TIPO_GIUDIZIO_VAL = "7"
RISULTATI_PER_PAG = 10

FIELDNAMES = [
    "Numero_Pronuncia", "Tipo", "Anno", "Date_GU", "Note_GU",
    "Camera_Consiglio", "Decisione_del", "Deposito_del",
    "Esito", "Comunicato", "URL_Pronuncia",
    "testo",
]

DATE = r"(\d{1,2}[/\.]\d{1,2}[/\.]\d{4})"

MESI = {
    "gennaio":1, "febbraio":2, "marzo":3, "aprile":4,
    "maggio":5, "giugno":6, "luglio":7, "agosto":8,
    "settembre":9, "ottobre":10, "novembre":11, "dicembre":12,
    # Varianti/refusi OCR presenti nei testi storici
    "gugno":6, "guigno":6, "giugon":6,
    "genaio":1, "gennnaio":1, "febraio":2, "aprle":4,
    "setembre":9, "settebre":9, "setsembre":9, "settmbre":9,
    "otobre":10, "ottbre":10, "otiobre":10,
    "novebre":11, "novmbre":11, "novembr":11,
    "dicebre":12, "dicmbre":12, "dicembr":12,
}
_MESI_ALT = "|".join(sorted(MESI.keys(), key=len, reverse=True))

# Cifre spesso corrotte dall'OCR: I/l al posto di 1, O/o al posto di 0
_GG   = r"[0-9Il]"
_AAAA = r"[0-9IlOo]"

ESITI_PATTERNS = [
    (r"illegittimit[àa]\s+costituzionale",                                            "illegittimità costituzionale"),
    (r"\binammissibil",                                                                "inammissibile"),
    (r"manifest[ae]\s+infondatezz[ae]|manifestamente\s+infondat[ae]",                "manifestamente infondata"),
    (r"\bnon\s+fondat[ae]",                                                            "non fondata"),
    (r"\bfondat[ae]\b",                                                                "fondata"),
    (r"\bestint[ao]\b",                                                                "estinto"),
    (r"restituisce\s+gli\s+atti|ordina\s+la\s+restituzione",                          "restituzione atti"),
    (r"\bsospesa\s+ogni\s+altra\s+decisione|\bsospende\b|\bordina\s+la\s+sospensione","sospesa - istruttoria"),
    (r"\brespinge\b|\brigetta\b",                                                      "rigetto eccezioni"),
    (r"\baccoglie\b",                                                                  "accoglimento"),
]


# ── Indice GU locale ──────────────────────────────────────────────────────────

def carica_indice_gu(csv_path: str) -> dict:
    """
    Carica gu_index.csv in un dizionario per lookup O(1).
    Chiave: (numero_gu, anno_gu)  →  Valore: "GG/MM/AAAA"
    Se il file non esiste restituisce dizionario vuoto con avviso.
    """
    indice = {}
    if not os.path.exists(csv_path):
        print(f"[!] {csv_path} non trovato.")
        print(f"[!] Le Date_GU saranno vuote per i casi senza data esplicita.")
        print(f"[!] Esegui prima: python build_gu.py")
        return indice

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                chiave = (int(row["numero"]), int(row["anno"]))
                indice[chiave] = row["data"]
            except Exception:
                pass

    print(f"[i] Indice GU caricato: {len(indice)} voci da {csv_path}")
    return indice


# ── Estrazione info GU dal testo ──────────────────────────────────────────────

# Inizio del corpo del giudizio e fine dell'epigrafe
MARKER_INIZIO_GIUDIZIO = re.compile(
    r"n(?:el|ei)\s+giudizi[oa]?\s+(?:riuniti\s+)?di\s+legittimit"
    r"|nel\s+giudizio\s+promosso",
    re.I)
MARKER_FINE_EPIGRAFE = re.compile(
    r"\bVist[oi]\b"            # Visto / Visti
    r"|\bUdit[oaie]\b"         # Udito / Udita / Uditi / Udite
    r"|\bRitenut[oi]\b"        # Ritenuto
    r"|\bConsiderat[oi]\b"     # Considerato
    r"|\bper\s+questi\s+motivi\b",
    re.I)


def taglia_epigrafe(text: str) -> str:
    """
    Normalizza gli spazi e ritaglia la sola epigrafe:
      INIZIO → "nel/nei giudizio/i ... di legittimità" (o "nel giudizio promosso")
      FINE   → primo marcatore procedurale (Visto/Udito/Ritenuto/Considerato/...)
    Le GU citate più avanti (leggi impugnate, precedenti) restano escluse.
    """
    text = re.sub(r"\s+", " ", text).strip()
    m_start = MARKER_INIZIO_GIUDIZIO.search(text)
    resto = text[m_start.start():] if m_start else text
    m_end = MARKER_FINE_EPIGRAFE.search(resto)
    return resto[:m_end.start()].strip() if m_end else resto


def _fix_ocr_int(s: str) -> int:
    """Converte le cifre OCR (I,l->1 ; O,o->0) e restituisce int."""
    return int(s.translate(str.maketrans("IlOo", "1100")))


def estrai_info_gu_da_testo(text: str) -> list:
    """
    Per ogni occorrenza di "Gazzetta Ufficiale":
      {"data": "GG/MM/AAAA"}           — data già nel testo -> usala
      {"numeri": [X], "anno": AAAA}    — solo numero GU + anno -> lookup CSV
    Logica portata identica dallo scraper (casi 1a..3k).
    """
    text = re.sub(
        r"G\s*a\s*z\s*z\s*e\s*t\s*t\s*a\s+U\s*f\s*f\s*i\s*c\s*i\s*a\s*l\s*e",
        "Gazzetta Ufficiale", text, flags=re.I)
    text = re.sub(
        r"Gazzetta\s+Uff+if+iciale",
        "Gazzetta Ufficiale", text, flags=re.I)
    # OCR: "Gazzetta"/"Ufficiale" corrotte (Gazzetfa, Ufliciale, Uffciale,
    # Ufficiate, Uticiale, Udiciale, ...) -> forma canonica.
    text = re.sub(
        r"\bGazz[a-z]{2,5}\s+U[a-z]{2,6}(?:ale|ate)\b",
        "Gazzetta Ufficiale", text, flags=re.I)
    # PLURALE: "Gazzette Ufficiali" (più ordinanze in gazzette diverse) -> singolare.
    text = re.sub(
        r"\bGazzett[ae]\s+Ufficial[ei]\b",
        "Gazzetta Ufficiale", text, flags=re.I)
    # Abbreviazioni: "Gazz. Uff.", "G. U." -> forma canonica.
    text = re.sub(r"\bGazz\.?\s*Uff\.?", "Gazzetta Ufficiale", text, flags=re.I)
    text = re.sub(r"\bG\.\s*U\.", "Gazzetta Ufficiale", text)
    # Giorno "1°" scritto come lo/l°/1°/1o/primo prima di un mese -> "1".
    text = re.sub(r"\b(?:lo|l°|1°|1o|primo)\s+(" + _MESI_ALT + r")\b",
                  r"1 \1", text, flags=re.I)
    text = re.sub(r"\([^)]{0,80}\)", "", text)

    risultati = []

    for m_gu in re.finditer(r"Gazzetta\s+Ufficiale\b", text, re.I):
        pos = m_gu.start()
        ctx_200 = text[pos:pos + 200]
        ctx_400 = text[pos:pos + 400]

        # ── CASO 1a: edizione straordinaria ───────────────────────────────────
        m_extra = re.search(
            r"edizione\s+straordinaria[^.\d]{0,40}?del\s+(" + _GG + r"{1,2})\s+(\w+)\s+(" + _AAAA + r"{4})",
            ctx_200, re.I)
        if m_extra:
            g, ms, a = (_fix_ocr_int(m_extra.group(1)), m_extra.group(2).lower(), _fix_ocr_int(m_extra.group(3)))
            if ms in MESI:
                try:
                    d = date(a, MESI[ms], g).strftime("%d/%m/%Y")
                except ValueError:
                    d = None          # data impossibile (es. 32 marzo): ignora
                if d:
                    risultati.append({"data": d})
                continue

        # ── CASO 1b: data in lettere ──────────────────────────────────────────
        m_lett = re.search(
            r"n\.?\s*\d+\.?(?:[^.'\u2019]{0,50}?)(?:del(?:l['\u2019])?|dei|de)\s*"
            r"(" + _GG + r"{1,2})\s+(\w+)\s+(" + _AAAA + r"{4})",
            ctx_200, re.I)
        if m_lett:
            g, ms, a = (_fix_ocr_int(m_lett.group(1)), m_lett.group(2).lower(), _fix_ocr_int(m_lett.group(3)))
            if ms in MESI:
                try:
                    d = date(a, MESI[ms], g).strftime("%d/%m/%Y")
                except ValueError:
                    d = None          # data impossibile (es. 32 marzo): ignora
                if d:
                    risultati.append({"data": d})
                continue

        # ── CASO 2: data numerica ─────────────────────────────────────────────
        m_num = re.search(
            r"n\.?\s*\d+\s+del\s+(\d{1,2}[/\.]\d{1,2}[/\.]\d{4})",
            ctx_200, re.I)
        if m_num:
            risultati.append({"data": m_num.group(1).replace(".", "/")})
            continue

        # ── CASO 1c: formato INVERTITO (data PRIMA del numero) — pronunce pre-1980
        #   "Gazzetta Ufficiale [della Repubblica] del <gg mese aaaa>[, n. NN]"
        #   Ancorato alla GU per non catturare la data dell'ordinanza vicina.
        m_inv = re.match(
            r"Gazzetta\s+Ufficiale(?:\s+della\s+Repubblica)?\s+"
            r"(?:(?:del(?:l['\u2019])?|dei|de|il|l['\u2019]|in\s+data)\s*)?"
            r"(" + _GG + r"{1,2})\s+(\w+)\s+(" + _AAAA + r"{4})",
            ctx_200, re.I)
        if m_inv:
            g, ms, a = _fix_ocr_int(m_inv.group(1)), m_inv.group(2).lower(), _fix_ocr_int(m_inv.group(3))
            if ms in MESI:
                try:
                    d = date(a, MESI[ms], g).strftime("%d/%m/%Y")
                except ValueError:
                    d = None
                if d:
                    risultati.append({"data": d})
                    continue

        # ── CASO 1d: "numero NNN del <gg mese aaaa>" (numero scritto per esteso)
        m_numero = re.search(
            r"\bnumero\s+\d+\.?\s*(?:del(?:l['\u2019])?|dei)\s+(" + _GG + r"{1,2})\s+(\w+)\s+(" + _AAAA + r"{4})",
            ctx_200, re.I)
        if m_numero:
            g, ms, a = _fix_ocr_int(m_numero.group(1)), m_numero.group(2).lower(), _fix_ocr_int(m_numero.group(3))
            if ms in MESI:
                try:
                    d = date(a, MESI[ms], g).strftime("%d/%m/%Y")
                except ValueError:
                    d = None
                if d:
                    risultati.append({"data": d})
                    continue

        # ── CASO 1e: due date che condividono mese+anno ──────────────────────
        #   "rispettivamente dell'8 e del 22 giugno 1963" / "il 12 e il 15 gennaio 1957"
        m_due = re.search(
            r"(?:dell['\u2019]|del\s+|dei\s+|il\s+|l['\u2019])(" + _GG + r"{1,2})\s+e\s+"
            r"(?:dell['\u2019]|del\s+|dei\s+|il\s+|l['\u2019])(" + _GG + r"{1,2})\s+"
            r"(\w+)\s+(" + _AAAA + r"{4})",
            ctx_200, re.I)
        if m_due:
            ms = m_due.group(3).lower()
            if ms in MESI:
                a = _fix_ocr_int(m_due.group(4))
                trovate = False
                for gg in (m_due.group(1), m_due.group(2)):
                    try:
                        d = date(a, MESI[ms], _fix_ocr_int(gg)).strftime("%d/%m/%Y")
                    except ValueError:
                        d = None
                    if d:
                        risultati.append({"data": d})
                        trovate = True
                if trovate:
                    continue

        # ── CASO 1f: lista multi-ordinanza con anno condiviso una volta sola ──
        #   "del 1986, prima serie speciale, n. 12 del 26 marzo, n. 13 del 2 aprile, ..."
        #   L'anno è dato una volta; ogni voce ha solo giorno+mese (senza anno).
        m_anno_rif = re.search(r"(?:del|dell['\u2019]\s*anno)\s+(\d{4})\b", ctx_400, re.I)
        if m_anno_rif:
            anno_rif = int(m_anno_rif.group(1))
            voci = re.findall(
                r"del(?:l['\u2019])?\s+(\d{1,2})\s+(" + _MESI_ALT + r")(?!\s+\d{4})",
                ctx_400, re.I)
            date_f = []
            for gg, ms in voci:
                ms = ms.lower()
                if ms in MESI:
                    try:
                        date_f.append(date(anno_rif, MESI[ms], int(gg)).strftime("%d/%m/%Y"))
                    except ValueError:
                        pass
            if len(date_f) >= 2:      # solo vere liste (>=2 voci senza anno proprio)
                for d in date_f:
                    if {"data": d} not in risultati:
                        risultati.append({"data": d})
                continue

        # ── Limite numeri GU ──────────────────────────────────────────────────
        is_serie_speciale = bool(re.search(r"prima\s+serie\s+speciale", ctx_400, re.I))
        max_num_gu = 60 if is_serie_speciale else 999

        # ── CASO 3a: ordine invertito ─────────────────────────────────────────
        m_inv = re.search(
            r"prima\s+serie\s+speciale[,\s]+numeri\s+([\d,\s]+(?:e\s+\d+)?)\s+"
            r"dell['\u2019]anno\s+(\d{4})", ctx_400, re.I)
        if m_inv:
            for n in [int(n) for n in re.findall(r"\d+", m_inv.group(1)) if int(n) <= max_num_gu]:
                risultati.append({"numeri": [n], "anno": int(m_inv.group(2))})
            continue

        # ── CASO 3b: 'numeri X, Y e Z ... dell'anno AAAA' ────────────────────
        m_multi = re.search(
            r"numeri\s+([\d,\s]+(?:e\s+\d+)?)[^'\u2019\d]{0,60}"
            r"dell['\u2019]anno\s+(\d{4})", ctx_400, re.I)
        if m_multi:
            for n in [int(n) for n in re.findall(r"\d+", m_multi.group(1)) if int(n) <= max_num_gu]:
                risultati.append({"numeri": [n], "anno": int(m_multi.group(2))})
            continue

        # ── CASO 3c: plurale 'nn. X, Y ... dell'anno AAAA' ───────────────────
        matches_nn = list(re.finditer(
            r"nn\.\s*(\d+(?:[,\s]+(?:e\s+)?\d+)*)[,\s]*"
            r"(?:prima\s+serie\s+speciale[,\s]*)?dell['\u2019]anno\s+(\d{4})",
            ctx_400, re.I))
        if matches_nn:
            for m_nn in matches_nn:
                for n in [int(n) for n in re.findall(r"\d+", m_nn.group(1)) if int(n) <= max_num_gu]:
                    risultati.append({"numeri": [n], "anno": int(m_nn.group(2))})
            continue

        # ── CASO 3d: 'n. X e Y, prima serie speciale, dell'anno AAAA' ─────────
        m_n_e = re.search(
            r"n\.?\s*(\d+)\s+e\s+(\d+)[,\s]*prima\s+serie\s+speciale[,\s]+"
            r"dell['\u2019]anno\s+(\d{4})", ctx_400, re.I)
        if m_n_e:
            anno_gu = int(m_n_e.group(3))
            for n in [int(m_n_e.group(1)), int(m_n_e.group(2))]:
                if 1 <= n <= max_num_gu:
                    risultati.append({"numeri": [n], "anno": anno_gu})
            continue

        # ── CASO 3g: 'n. X 1a/1ª serie speciale ... dell'anno AAAA' ──────────
        m_1a = re.search(
            r"n\.?\s*(\d+)[\s,\-]+1[a\xaa]\s+serie\s+speciale[\s,\-]+"
            r"dell['\u2019]anno\s+(\d{4})", ctx_400, re.I)
        if m_1a:
            n = int(m_1a.group(1))
            if 1 <= n <= max_num_gu:
                risultati.append({"numeri": [n], "anno": int(m_1a.group(2))})
            continue

        # ── CASO 3i: formato con slash (anni 1986-1987) ───────────────────────
        matches_slash = list(re.finditer(
            r"n\.?\s*(\d+)\s*/[^'\u2019]{0,40}dell['\u2019]anno\s+(\d{4})",
            ctx_400, re.I))
        if matches_slash:
            for m_s in matches_slash:
                n = int(m_s.group(1))
                if 1 <= n <= max_num_gu:
                    risultati.append({"numeri": [n], "anno": int(m_s.group(2))})
            continue

        # ── CASO 3j: varianti 'nn.' anni 1985-1987 ────────────────────────────
        trovati_3j = []
        for m in re.finditer(
                r"nn\.\s*(\d+(?:[,\s]+(?:e\s+)?\d+)*)\s+della\s+"
                r"(?:1[a\xaa°]|prima)\s+(?:serie\s+speciale|s\.s\.)\s+"
                r"del(?:l['\u2019]anno)?\s+(\d{4})\b", ctx_400, re.I):
            anno_gu = int(m.group(2))
            for n in [int(n) for n in re.findall(r"\d+", m.group(1)) if int(n) <= max_num_gu]:
                trovati_3j.append({"numeri": [n], "anno": anno_gu})
        for m in re.finditer(
                r"nn\.\s*(\d+(?:[,\s]+(?:e\s+)?\d+)*)[,\s]+"
                r"prima\s+serie\s+speciale[,\s]+anno\s+(\d{4})\b", ctx_400, re.I):
            for n in [int(n) for n in re.findall(r"\d+", m.group(1)) if int(n) <= max_num_gu]:
                trovati_3j.append({"numeri": [n], "anno": int(m.group(2))})
        for m in re.finditer(
                r"nn\.\s*(\d+(?:[,\s]+(?:e\s+)?\d+)*)\s+"
                r"1[a\xaa°]\s+serie\s+speciale\s+dell['\u2019]anno\s+(\d{4})", ctx_400, re.I):
            for n in [int(n) for n in re.findall(r"\d+", m.group(1)) if int(n) <= max_num_gu]:
                trovati_3j.append({"numeri": [n], "anno": int(m.group(2))})
        for m in re.finditer(
                r"nn\.\s*((?:\d+[\s\-]+bis)(?:[,\s]+(?:e\s+)?\d+[\s\-]+bis)*)\s+"
                r"dell['\u2019]anno\s+(\d{4})", ctx_400, re.I):
            for n in [int(n) for n in re.findall(r"(\d+)[\s\-]+bis", m.group(1), re.I) if int(n) <= max_num_gu]:
                trovati_3j.append({"numeri": [n], "anno": int(m.group(2))})
        for m in re.finditer(
                r"nn\.\s*(\d+(?:[,\s]+(?:e\s+)?\d+)*)[,\s]+"
                r"I\s+Serie\s+Speciale[,\s]+dell['\u2019]anno\s+(\d{4})", ctx_400, re.I):
            for n in [int(n) for n in re.findall(r"\d+", m.group(1)) if int(n) <= max_num_gu]:
                trovati_3j.append({"numeri": [n], "anno": int(m.group(2))})
        for m in re.finditer(
                r"nn\.\s*(\d+(?:[,\s]+(?:e\s+)?\d+)*)[,\s]+"
                r"prima\s+serie\s+speciale[,\s]+del\s+(\d{4})\b", ctx_400, re.I):
            for n in [int(n) for n in re.findall(r"\d+", m.group(1)) if int(n) <= max_num_gu]:
                trovati_3j.append({"numeri": [n], "anno": int(m.group(2))})
        for m in re.finditer(
                r"nn\.\s*(\d+(?:[,\s]+(?:e\s+)?\d+)*)\s+"
                r"1\s+s\.s\.\s+dell['\u2019]anno\s+(\d{4})", ctx_400, re.I):
            for n in [int(n) for n in re.findall(r"\d+", m.group(1)) if int(n) <= max_num_gu]:
                trovati_3j.append({"numeri": [n], "anno": int(m.group(2))})
        for m in re.finditer(
                r"n\.?\s*(\d+)[,\s]+prima\s+serie\s+speciale\s+"
                r"dell['\u2019]anno\s+(\d{4})", ctx_400, re.I):
            n = int(m.group(1))
            if 1 <= n <= max_num_gu:
                trovati_3j.append({"numeri": [n], "anno": int(m.group(2))})
        for m in re.finditer(
                r"n\.?\s*(\d+)[\s\-]+bis\s+dell['\u2019]anno\s+(\d{4})", ctx_400, re.I):
            n = int(m.group(1))
            if 1 <= n <= max_num_gu:
                trovati_3j.append({"numeri": [n], "anno": int(m.group(2))})
        if trovati_3j:
            risultati.extend(trovati_3j)
            continue

        # ── CASO 3k: 1ª serie speciale anni 2000+ ─────────────────────────────
        PAT_3K = re.compile(
            r'(?:numeri|nn\.|n\.?)\s+([\d,\s]+(?:e\s+\d+)?)[\s,\-]+'
            r'1[a\xaa°]\s*serie?\s*speciale[\s,\-]+dell[\u2019\']anno\s+(\d{4})', re.I)
        PAT_3K_NUDO = re.compile(
            r'(?:[\s,]e\s+)(\d+(?:[,\s]+(?:e\s+)?\d+)*)[\s,\-]+'
            r'1[a\xaa°]\s*serie?\s*speciale[\s,\-]+dell[\u2019\']anno\s+(\d{4})', re.I)
        trovati_3k, seen_3k = [], set()
        for m in list(PAT_3K.finditer(ctx_400)) + list(PAT_3K_NUDO.finditer(ctx_400)):
            anno_k = int(m.group(2))
            for n in [int(x) for x in re.findall(r'\d+', m.group(1)) if 1 <= int(x) <= 60]:
                if (n, anno_k) not in seen_3k:
                    seen_3k.add((n, anno_k))
                    trovati_3k.append({"numeri": [n], "anno": anno_k})
        if trovati_3k:
            risultati.extend(trovati_3k)
            continue

        # ── CASO 3e: singolo/multi con 'dell'anno' (fallback generale) ───────
        coppie = re.findall(
            r"n[.,]?\s*(\d+)[^'\u2019\d]{0,60}dell['\u2019]anno\s+(\d{4})", ctx_400, re.I)
        if coppie:
            app_3e = False
            for num_str, anno_str in coppie:
                n = int(num_str)
                if 1 <= n <= max_num_gu:
                    risultati.append({"numeri": [n], "anno": int(anno_str)})
                    app_3e = True
            if app_3e:      # se ha scartato tutto (num > cap), NON blocca il 3M
                continue

        # ── CASO 3h: formato 'bis' (edizioni straordinarie pre-1987) ─────────
        matches_nn_bis = list(re.finditer(
            r"nn\.\s*((?:\d+\s+bis)(?:[,\s]+(?:e\s+)?\d+\s+bis)*)\s+"
            r"dell['\u2019]anno\s+(\d{4})", ctx_400, re.I))
        if matches_nn_bis:
            for m_bis in matches_nn_bis:
                anno_gu = int(m_bis.group(2))
                for n in [int(n) for n in re.findall(r"(\d+)\s+bis", m_bis.group(1), re.I) if int(n) <= max_num_gu]:
                    risultati.append({"numeri": [n], "anno": anno_gu})
            continue
        m_n_bis = re.search(r"n\.?\s*(\d+)\s+bis\s+del\s+(\d{4})\b", ctx_400, re.I)
        if m_n_bis:
            n, anno_gu = int(m_n_bis.group(1)), int(m_n_bis.group(2))
            if 1956 <= anno_gu <= 2026 and 1 <= n <= max_num_gu:
                risultati.append({"numeri": [n], "anno": anno_gu})
            continue

        # ── CASO 3f: formato old-style 'del AAAA' senza 'dell'' ──────────────
        trovati_3f = []
        for m in re.finditer(
                r"nn\.\s*(\d+(?:[,\s]+(?:e\s+)?\d+)*)\s+del\s+(\d{4})\b", ctx_400, re.I):
            anno = int(m.group(2))
            if 1956 <= anno <= 2026:
                for n in [int(x) for x in re.findall(r"\d+", m.group(1)) if 1 <= int(x) <= max_num_gu]:
                    trovati_3f.append({"numeri": [n], "anno": anno})
        for m in re.finditer(
                r"n\.?\s*(\d+)\s+e\s+(\d+)\s+del\s+(\d{4})\b", ctx_400, re.I):
            anno = int(m.group(3))
            if 1956 <= anno <= 2026:
                for n in [int(m.group(1)), int(m.group(2))]:
                    if 1 <= n <= max_num_gu:
                        trovati_3f.append({"numeri": [n], "anno": anno})
        for m in re.finditer(
                r"(?<![n\d])n\.\s*(\d+)\s+del\s+(\d{4})\b", ctx_400, re.I):
            n, anno = int(m.group(1)), int(m.group(2))
            if 1956 <= anno <= 2026 and 1 <= n <= max_num_gu:
                trovati_3f.append({"numeri": [n], "anno": anno})
        risultati.extend(trovati_3f)

        # ── CASO 3M: serie speciale moderna con varianti (fallback robusto) ───
        #   Copre: numero/numeri per esteso, dall'anno, anno nudo, trattini,
        #   numero senza n., "1a/1o/1ª/1.a [serie] speciale", "prime speciali", "s.s.".
        if not trovati_3f:
            m_ss = re.search(
                r"(?:1[\s.]*[ªa°o8]?|prim[ae])\s*(?:a\s+)?(?:seri[ae]?\s+)?"
                r"(?:spec[a-z]{2,8}|s\.?\s*s\.?)"
                r"[\s,\-\u2013.]*"
                r"(?:(?:dell|dall)['\u2019]\s*anno\s+|del\s+(?:l['\u2019]\s*anno\s+)?)?(\d{4})",
                ctx_400, re.I)
            if m_ss:
                anno_ss = int(m_ss.group(1))
                if 1985 <= anno_ss <= 2026:
                    pre = ctx_400[:m_ss.start()]
                    tagli = list(re.finditer(r"Repubblica|Ufficiale", pre, re.I))
                    blocco = pre[tagli[-1].end():] if tagli else pre
                    nums = [int(x) for x in re.findall(r"\b(\d{1,3})\b", blocco)
                            if 1 <= int(x) <= 999]
                    visti_ss = []
                    for n in nums:
                        if n not in visti_ss:
                            visti_ss.append(n)
                            risultati.append({"numeri": [n], "anno": anno_ss})
                    if visti_ss:
                        continue

    return risultati

# ── Lookup date GU ────────────────────────────────────────────────────────────

def estrai_tutte_date_gu(testo: str, text_pagina: str, indice_gu: dict) -> tuple:
    """
    Estrae le date GU dall'epigrafe della pronuncia. Nessuna richiesta di rete.

    STEP 1 — Data già nel testo (lettere o numerica): usala direttamente.
    STEP 2 — Solo numero GU + anno: lookup O(1) su indice_gu (da gu_index.csv).

    L'epigrafe si ricava dal testo integrale; se questo manca si usa il testo della
    pagina a partire dal corpo del giudizio. Senza corpo si rinuncia: i metadati in
    cima alla pagina contengono la GU della pronuncia stessa → falso positivo.

    Restituisce (date separate da " - ", note sui fascicoli non presenti nell'indice).
    """
    if testo:
        epigrafe = taglia_epigrafe(testo)
    elif MARKER_INIZIO_GIUDIZIO.search(text_pagina):
        epigrafe = taglia_epigrafe(text_pagina)
    else:
        return "", ""

    date_trovate  = []
    non_in_indice = []

    for info in estrai_info_gu_da_testo(epigrafe):
        if "data" in info:
            # Step 1: data già nel testo
            if info["data"] not in date_trovate:
                date_trovate.append(info["data"])

        elif "numeri" in info:
            # Step 2: lookup nel CSV
            for n_gu in info["numeri"]:
                chiave = (n_gu, info["anno"])
                d = indice_gu.get(chiave, "")
                if d and d not in date_trovate:
                    date_trovate.append(d)
                    print(f"    [GU] n.{n_gu}/{info['anno']} → {d}")
                elif not d:
                    print(f"    [GU] n.{n_gu}/{info['anno']} → non in indice")
                    ref = f"n.{n_gu}/{info['anno']}"
                    if ref not in non_in_indice:
                        non_in_indice.append(ref)

    note = ("GU non in indice: " + ", ".join(non_in_indice)) if non_in_indice else ""
    return " - ".join(date_trovate), note


# ── Esito della pronuncia ─────────────────────────────────────────────────────

def estrai_esito(text: str) -> str:
    # Due step per evitare backtracking catastrofico su testi grandi
    m_fine = re.search(r"Cos[ìíi]\s+deciso|cosi['\s]+deciso", text, re.I)
    if not m_fine:
        return ""
    testo_prima = text[:m_fine.start()]
    m = re.search(
        r"per\s+questi\s+motivi\s+"
        r"LA\s+CORTE\s+COSTITUZIONALE\s+"
        r"(.+)$",
        testo_prima, re.I | re.S)
    if not m:
        return ""
    dispositivo = m.group(1).strip()

    trovati = []
    for pattern, categoria in ESITI_PATTERNS:
        if re.search(pattern, dispositivo, re.I) and categoria not in trovati:
            trovati.append(categoria)

    if len(trovati) > 1 and "rigetto eccezioni" in trovati:
        trovati.remove("rigetto eccezioni")

    if "fondata" in trovati:
        reale = False
        for fm in re.finditer(r"\bfondat[ae]\b", dispositivo, re.I):
            prima = dispositivo[max(0, fm.start()-15):fm.start()].lower()
            if "non " not in prima and not prima.rstrip().endswith("in"):
                reale = True
                break
        if not reale:
            trovati.remove("fondata")

    if trovati:
        return " - ".join(trovati)

    disp_clean = re.sub(
        r"^\s*riuniti\s+i\s+giudizi[,;]?\s*", "", dispositivo, flags=re.I).strip()
    fb = re.match(r"([^,;.]{3,80})(?:[,;.]|$)", disp_clean, re.I)
    if fb:
        return fb.group(1).strip().rstrip(".,;")
    return " ".join(disp_clean.split()[:10])


# ── Testo integrale della pronuncia ───────────────────────────────────────────

# Inizio del testo, in ordine di priorità:
#  1. "ha [verbo, fino a 6 parole] la seguente TITOLO": copre "ha pronunciato",
#     "ha deliberato in camera di consiglio", "ha emesso", "ha adottato", ...
#  2. TITOLO su una riga isolata (pronunce storiche senza formula di enunciazione)
MARKER_INIZIO_TESTO = [
    re.compile(r"ha\s+\w+(?:\s+\w+){0,6}\s+la\s+seguente\s+"
               r"(?:SENTENZA|ORDINANZA|DECISIONE|DECRETO)\b", re.I),
    re.compile(r"\n\s*\n\s*(?:SENTENZA|ORDINANZA|DECISIONE|DECRETO)\s*\n", re.I),
]


def estrai_testo_integrale(testo_pagina: str) -> str:
    """
    Testo della pronuncia dalla formula di enunciazione fino alla fine della
    pagina. testo_pagina deve conservare gli a capo (get_text con separator="\n").
    """
    for pat in MARKER_INIZIO_TESTO:
        m = pat.search(testo_pagina)
        if m:
            testo = testo_pagina[m.start():].strip()
            return re.sub(r"\n{3,}", "\n\n", testo)
    return ""


# ── Parsing metadati ──────────────────────────────────────────────────────────

def parse_metadata(html: str, url: str, indice_gu: dict) -> dict:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    meta = {k: "" for k in FIELDNAMES}
    meta["URL_Pronuncia"] = url

    m = re.search(r"/(\d{4})/(\d+)", url)
    if m:
        meta["Anno"]             = m.group(1)
        meta["Numero_Pronuncia"] = m.group(2)

    if re.search(r"\bSENTENZA\b", text, re.I):
        meta["Tipo"] = "Sentenza"
    elif re.search(r"\bORDINANZA\b", text, re.I):
        meta["Tipo"] = "Ordinanza"

    def first_date_after(label: str) -> str:
        fm = re.search(rf"{label}[^0-9]{{0,80}}{DATE}", text, re.I | re.S)
        return fm.group(1).replace(".", "/") if fm else ""

    meta["Camera_Consiglio"] = (
        first_date_after(r"camera\s+di\s+consiglio")
        or first_date_after(r"udienza\s+pubblica")
    )
    meta["Decisione_del"]   = first_date_after(r"decisione")
    meta["Deposito_del"]    = first_date_after(r"deposito")
    meta["Esito"]           = estrai_esito(text)
    meta["Comunicato"]      = "SI" if any(
        a.get_text(strip=True) == "Comunicato" for a in soup.find_all("a")
    ) else "NO"

    # Testo integrale: si tolgono gli elementi di navigazione della pagina e si
    # conservano gli a capo (servono al riconoscimento del titolo isolato)
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    meta["testo"] = estrai_testo_integrale(soup.get_text(separator="\n"))

    meta["Date_GU"], meta["Note_GU"] = estrai_tutte_date_gu(meta["testo"], text, indice_gu)

    print(
        f"    [META] n.{meta['Numero_Pronuncia']}/{meta['Anno']} | "
        f"{meta['Tipo']} | Dep: {meta['Deposito_del']} | "
        f"GU: {meta['Date_GU'][:30] or '—'} | "
        f"Esito: {meta['Esito'][:20]} | "
        f"Testo: {len(meta['testo'])} char"
    )
    return meta

# ── Raccolta link ─────────────────────────────────────────────────────────────

def collect_links_from_page(page_source: str) -> list:
    soup  = BeautifulSoup(page_source, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "scheda-pronuncia" in href or "visualizzaScheda" in href:
            if href.startswith("/"):
                href = "https://www.cortecostituzionale.it" + href
            links.append(href)
    for a in soup.find_all("a", onclick=True):
        fm = re.search(r"visualizzaScheda\('([^']+)'\)", a["onclick"])
        if fm:
            href = fm.group(1)
            if href.startswith("/"):
                href = "https://www.cortecostituzionale.it" + href
            links.append(href)
    seen = set()
    uniq = []
    for l in links:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    return uniq


def get_total_results(page_source: str) -> int:
    m = re.search(r"pronunce\s+INDIVIDUATE[:\s]+(\d[\d\s]*)", page_source, re.I)
    return int(m.group(1).replace(" ", "")) if m else 0


# ── Captcha ───────────────────────────────────────────────────────────────────

def attendi_captcha(driver):
    """
    Se il sito ha reindirizzato alla pagina anti-bot (Radware/ShieldSquare su
    validate.perfdrive.com, oppure hCaptcha), attende che l'utente la risolva
    nel browser e prema INVIO. Restituisce True se c'era un captcha.
    """
    def in_captcha():
        return ("perfdrive.com" in driver.current_url
                or "captcha" in driver.title.lower()
                or "hcaptcha" in driver.page_source.lower())

    trovato = False
    while in_captcha():
        trovato = True
        print("    [!] CAPTCHA – risolvilo nel browser e premi INVIO")
        input()
        time.sleep(3)
    return trovato


# ── Ricerca per anno ──────────────────────────────────────────────────────────

def pagina_filtrata(page_source: str, anno: str) -> bool:
    """True se la pagina contiene link e sono TUTTI dell'anno cercato."""
    links = collect_links_from_page(page_source)
    return bool(links) and all(f"/{anno}/" in l for l in links)


def esegui_ricerca(driver, wait, anno: str) -> bool:
    """
    Compila e invia il form di ricerca (POST: i filtri restano nella sessione
    del server, le pagine /elenco-pronunce/k dipendono da essa).
    Il captcha Radware fa perdere i filtri → dopo averlo risolto si ripete
    la ricerca finché il risultato è davvero filtrato per anno.
    Restituisce False se non ci riesce (o se non ci sono risultati).
    """
    for _ in range(3):
        driver.get(SEARCH_URL)
        time.sleep(3)
        attendi_captcha(driver)
        try:
            wait.until(EC.presence_of_element_located((By.NAME, "tipo_giudizio")))
        except Exception:
            print(f"  [!] Timeout form per {anno}"); return False

        Select(driver.find_element(By.NAME, "tipo_giudizio")).select_by_value(TIPO_GIUDIZIO_VAL)
        time.sleep(0.8)
        campo = driver.find_element(By.NAME, "anno")
        campo.clear()
        campo.send_keys(anno)
        time.sleep(0.5)

        try:
            btn = driver.find_element(By.CSS_SELECTOR, "button[onclick*='ricerca']")
        except Exception:
            btn = driver.find_element(
                By.CSS_SELECTOR, "input[type='submit'],button[type='submit']")
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(5)
        attendi_captcha(driver)

        src = driver.page_source
        if "nessun risultato" in src.lower():
            print(f"  [=] Nessun risultato per {anno}"); return False
        if pagina_filtrata(src, anno):
            return True
        print(f"  [i] Risultati non filtrati per {anno} (captcha?) — ripeto la ricerca")

    print(f"  [!] Impossibile applicare il filtro anno {anno}"); return False


def search_anno(driver, wait, anno: str, seen_urls: set,
                jsonl_path: str, indice_gu: dict,
                num_da: int = None, num_a: int = None):
    """
    num_da: numero pronuncia da cui partire (incluso)
    num_a:  numero pronuncia fino a cui arrivare (incluso)
    Se entrambi None → scarica tutte. Solo num_da → da num_da in poi.
    num_da == num_a → solo quella pronuncia.
    """
    print(f"\n{'='*60}\n  ANNO: {anno}\n{'='*60}")

    if not esegui_ricerca(driver, wait, anno):
        return

    src = driver.page_source
    totale = get_total_results(src)
    if totale == 0:
        totale = len(collect_links_from_page(src))

    n_pagine = math.ceil(totale / RISULTATI_PER_PAG)
    print(f"  [i] Pronunce: {totale} → {n_pagine} pagine")

    all_links = []
    for k in range(n_pagine):
        url_p = LIST_BASE_URL if k == 0 else f"{LIST_BASE_URL}/{k}"
        print(f"  [Pagina {k+1}/{n_pagine}]")
        driver.get(url_p)
        time.sleep(random.uniform(2.0, 3.5))
        attendi_captcha(driver)
        # Se il captcha ha fatto perdere il filtro, rifai la ricerca e ricarica la pagina k
        if not pagina_filtrata(driver.page_source, anno):
            print("    [i] Filtro perso — ripeto la ricerca e ricarico la pagina")
            if not esegui_ricerca(driver, wait, anno):
                break
            driver.get(url_p)
            time.sleep(random.uniform(2.0, 3.5))
            attendi_captcha(driver)
        pl = collect_links_from_page(driver.page_source)
        print(f"    {len(pl)} link")
        if not pl or not pagina_filtrata(driver.page_source, anno):
            print("    [!] Pagina vuota o non filtrata — interrompo"); break
        all_links.extend(pl)

    seen_run = set()
    unique   = []
    for l in all_links:
        if l not in seen_run:
            seen_run.add(l)
            unique.append(l)

    # Filtra solo i link che appartengono all'anno cercato
    unique = [l for l in unique if f"/{anno}/" in l]

    # Filtro per range numero pronuncia (--da, --a, --solo)
    if num_da is not None or num_a is not None:
        def estrai_numero(url: str) -> int:
            m = re.search(r"/(\d+)$", url)
            return int(m.group(1)) if m else -1
        if num_da is not None and num_a is not None:
            unique = [l for l in unique if num_da <= estrai_numero(l) <= num_a]
            print(f"  [i] Filtro numeri: {num_da}\u2013{num_a}")
        elif num_da is not None:
            unique = [l for l in unique if estrai_numero(l) >= num_da]
            print(f"  [i] Filtro numeri: da {num_da} in poi")
        elif num_a is not None:
            unique = [l for l in unique if estrai_numero(l) <= num_a]
            print(f"  [i] Filtro numeri: fino a {num_a}")

    nuovi = [l for l in unique if l not in seen_urls]
    print(f"\n  [i] {len(unique)} totali, {len(nuovi)} nuovi")
    if not nuovi:
        print("  [=] Nessuna novità"); return

    for idx, url in enumerate(nuovi, 1):
        print(f"    [{idx}/{len(nuovi)}] {url}")
        try:
            driver.get(url)
            time.sleep(random.uniform(2.0, 3.5))
            attendi_captcha(driver)
            data = parse_metadata(driver.page_source, url, indice_gu)
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            seen_urls.add(url)
            print(f"      [OK] n.{data['Numero_Pronuncia']}/{data['Anno']}")
        except KeyboardInterrupt:
            raise  # propaga per fermarsi pulitamente
        except Exception as e:
            print(f"      [ERR] {type(e).__name__}: {str(e)[:100]}")
            # Verifica se Chrome è ancora vivo dopo l'errore
            try:
                driver.current_url
            except Exception:
                print("      [!] Chrome non risponde — interrompo anno")
                break
        time.sleep(random.uniform(0.8, 1.8))


# ── Driver ────────────────────────────────────────────────────────────────────

def create_driver():
    opts = uc.ChromeOptions()
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    # pageLoadStrategy 'eager': non aspetta CSS/immagini, solo DOM pronto
    # Evita blocchi infiniti su risorse lente o non caricate
    opts.page_load_strategy = "eager"
    driver = uc.Chrome(options=opts, version_main=153)
    # Timeout massimo per il caricamento di una pagina: 30 secondi
    driver.set_page_load_timeout(30)
    return driver


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Estrai opzioni --da, --a, --solo PRIMA di valutare gli anni ──────
    # Così "1985 --solo 293" diventa args=["1985"] e num_da=num_a=293
    num_da = None
    num_a  = None
    args   = []
    raw    = sys.argv[1:]
    i = 0
    while i < len(raw):
        if raw[i] in ("--da", "--a", "--solo") and i + 1 < len(raw):
            try:
                n = int(raw[i + 1])
            except ValueError:
                print(f"[!] {raw[i]} richiede un numero intero."); sys.exit(1)
            if raw[i] == "--da":   num_da = n
            elif raw[i] == "--a":  num_a  = n
            elif raw[i] == "--solo": num_da = num_a = n
            i += 2
        else:
            args.append(raw[i]); i += 1

    # ── Parsing anni ──────────────────────────────────────────────────────
    if len(args) == 0:
        anni = [str(a) for a in range(1956, 2027)]
    elif len(args) == 1:
        try:
            anno = int(args[0])
            if not (1956 <= anno <= 2026):
                print(f"[!] Anno non valido: {anno}."); sys.exit(1)
            anni = [str(anno)]
        except ValueError:
            print(f"[!] Argomento non valido: '{args[0]}'."); sys.exit(1)
    elif len(args) == 2:
        try:
            a_inizio, a_fine = int(args[0]), int(args[1])
            a_min, a_max = min(a_inizio, a_fine), max(a_inizio, a_fine)
            if not (1956 <= a_min and a_max <= 2026):
                print(f"[!] Range non valido."); sys.exit(1)
            anni = [str(a) for a in range(a_max, a_min - 1, -1)]
        except ValueError:
            print(f"[!] Argomenti non validi."); sys.exit(1)
    else:
        print("[!] Uso: python scraper_v2.py [anno] o [anno_inizio] [anno_fine]")
        print("         Opzioni: --da N  --a N  --solo N")
        sys.exit(1)

    if num_da is not None or num_a is not None:
        _desc = []
        if num_da is not None: _desc.append(f"da n.{num_da}")
        if num_a  is not None: _desc.append(f"a n.{num_a}")
        print(f"[i] Filtro pronuncia: {' '.join(_desc)}")

    print(f"[i] Anni: {anni[0]} → {anni[-1]} ({len(anni)} anni)")
    print(f"[i] Output: {OUTPUT_JSONL}")

    # Carica indice GU locale (prodotto da build_gu.py)
    indice_gu = carica_indice_gu(GU_INDEX_CSV)

    # Carica URL già presenti nel JSONL (ripresa)
    seen_urls = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("URL_Pronuncia"):
                        seen_urls.add(row["URL_Pronuncia"])
                except Exception:
                    pass
        print(f"[i] Ripresa: {len(seen_urls)} URL già nel JSONL")

    for anno in anni:
        driver = create_driver()
        wait   = WebDriverWait(driver, 25)
        try:
            search_anno(driver, wait, anno, seen_urls, OUTPUT_JSONL, indice_gu,
                        num_da=num_da, num_a=num_a)
        except KeyboardInterrupt:
            print("\n[!] Interrotto"); driver.quit(); break
        except Exception as e:
            import traceback
            print(f"\n[!] Errore anno {anno}: {e}"); traceback.print_exc()
        finally:
            try:
                driver.quit()
            except Exception:
                pass
        time.sleep(random.uniform(3.0, 6.0))

    print(f"\n[OK] JSONL salvato in: {OUTPUT_JSONL}")
    print(f"[i] Prossimo passo: python llm_extractor.py")


if __name__ == "__main__":
    main()