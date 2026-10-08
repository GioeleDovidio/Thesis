"""
unisci_dataset.py — assembla il dataset per il ML
=================================================
Unisce (join sull'URL della pronuncia):
  - le FEATURE estratte dal nostro estrattore (feature_estratte.csv, separatore ';')
  - i TARGET presi dal dataset delle pronunce, pronunce.csv (data GU, durata, esito)
e costruisce le due variabili da prevedere:
  - esito         : accoglimento | infondatezza | inammissibilita | altro
  - durata_giorni : giorni tra la (PRIMA) data GU e la data di deposito

Pronunce con esito misto: un dispositivo puo' contenere piu' esiti (separati da ' - '),
quando la Corte decide in modo diverso questioni diverse nella stessa pronuncia. Queste
pronunce vengono ESCLUSE dal dataset: le feature descrivono la pronuncia nel suo insieme,
e ridurre piu' esiti a uno solo richiederebbe una regola arbitraria. Restano solo le
pronunce in cui tutte le parti del dispositivo appartengono alla stessa classe
(es. "manifestamente infondata - non fondata" -> infondatezza).

Nota metodologica: dal CSV originale prendiamo SOLO i target. Le altre sue colonne
(relatore, forma, comunicato, ...) sono "a valle" della decisione: usarle come feature
sarebbe data leakage. Le feature restano quelle dell'estrattore.

Dipendenze: pip install pandas
Uso:        python unisci_dataset.py
Alla prima esecuzione, se non riconosce una colonna, stampa le colonne disponibili:
imposta i nomi giusti nella sezione CONFIG e rilancia.
"""

import re
import sys
import unicodedata
from datetime import datetime

import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
FEATURE_CSV = "feature_estratte.csv"     # dal nostro estrattore (separatore ';')
ORIGINALE_CSV = "pronunce.csv"           # dataset delle pronunce (llm_extractor.py + aggiungi_giorni_gu.py)
OUTPUT_CSV = "dataset_ml.csv"

# Nomi colonna nel CSV ORIGINALE (già impostati sul tuo file).
COL_URL_ORIG = "URL_Pronuncia"
COL_DATE_GU = "Date_GU"                 # teniamo la PRIMA data come colonna di riferimento
COL_GIORNI = "Giorni_GU_Decisione"      # TARGET durata: giorni GU->decisione, GIA' calcolato (non si ricalcola)
COL_ESITO = "Esito"

# Mappatura esito -> 4 classi, calibrata sul vocabolario reale del file.
# Ogni parte del dispositivo (separata da ' - ') viene classificata da sola; se le parti
# appartengono a classi diverse la pronuncia e' "mista" e viene esclusa (vedi docstring).
KEYWORDS_ESITO = {
    "accoglimento":    ["illegittim", "accolt", "accoglimento"],        # "illegittimità costituzionale"
    "infondatezza":    ["non fondat", "infondat"],                      # "non fondata", "manifestamente infondata"
    "inammissibilita": ["inammissibil", "irricevibil"],                 # "inammissibile"
    # tutto il resto (restituzione atti, sospesa, istruttoria, rinvio, estinto...) -> "altro" per default
}

MESI = {m: i for i, m in enumerate(
    ["gennaio","febbraio","marzo","aprile","maggio","giugno","luglio","agosto",
     "settembre","ottobre","novembre","dicembre"], start=1)}


# ── Utility ───────────────────────────────────────────────────────────────────
def norm_url(u):
    if not isinstance(u, str):
        return ""
    return u.strip().lower().rstrip("/")

def rileva_colonna(df, candidati, contiene=None):
    """Trova una colonna per nome (candidati) o per contenuto (contiene = regex sui valori)."""
    low = {c.lower().strip(): c for c in df.columns}
    for cand in candidati:
        if cand in low:
            return low[cand]
    if contiene:
        for c in df.columns:
            sample = df[c].astype(str).head(50).str.contains(contiene, case=False, regex=True, na=False)
            if sample.mean() > 0.5:
                return c
    return None

def prima_data(cella):
    """Estrae la PRIMA data che compare nella cella (qualunque separatore) e la parsa.
    Gestisce gg/mm/aaaa, gg-mm-aaaa, aaaa-mm-gg e 'gg mese aaaa'."""
    if not isinstance(cella, str) or not cella.strip():
        return pd.NaT
    s = cella.strip()
    # 1) formato numerico gg/mm/aaaa o gg-mm-aaaa o aaaa-mm-gg
    m = re.search(r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})', s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try: return pd.Timestamp(datetime(y, mo, d))
        except ValueError: pass
    m = re.search(r'(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})', s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try: return pd.Timestamp(datetime(y, mo, d))
        except ValueError: pass
    # 2) formato testuale 'gg mese aaaa'
    m = re.search(r'(\d{1,2})\s+([a-zA-Zàèéìòù]+)\s+(\d{4})', s)
    if m and m.group(2).lower() in MESI:
        d, mo, y = int(m.group(1)), MESI[m.group(2).lower()], int(m.group(3))
        try: return pd.Timestamp(datetime(y, mo, d))
        except ValueError: pass
    return pd.NaT

def classe_parte(parte):
    """Classe di una singola parte del dispositivo; 'altro' se nessuna parola chiave."""
    t = parte.lower()
    for classe, kws in KEYWORDS_ESITO.items():
        if any(kw in t for kw in kws):
            return classe
    return "altro"

def classifica_esito(cella):
    """Classe dell'esito se tutte le parti del dispositivo concordano, altrimenti 'misto'."""
    if not isinstance(cella, str) or not cella.strip():
        return "altro"
    classi = {classe_parte(p) for p in cella.split(" - ") if p.strip()}
    return classi.pop() if len(classi) == 1 else "misto"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    try:
        feat = pd.read_csv(FEATURE_CSV, sep=";", dtype=str, keep_default_na=False)
    except FileNotFoundError:
        print(f"[!] Non trovo {FEATURE_CSV}"); sys.exit(1)
    # il CSV originale: rileva il separatore automaticamente
    try:
        orig = pd.read_csv(ORIGINALE_CSV, sep=None, engine="python", dtype=str, keep_default_na=False)
    except FileNotFoundError:
        print(f"[!] Non trovo {ORIGINALE_CSV}"); sys.exit(1)

    print(f"[i] feature: {len(feat)} righe, {len(feat.columns)} colonne")
    print(f"[i] originale: {len(orig)} righe, colonne: {list(orig.columns)}\n")

    # rileva le colonne del CSV originale
    c_url = COL_URL_ORIG or rileva_colonna(orig, ["url_pronuncia","url","link"], contiene=r"scheda-pronuncia")
    c_gu  = COL_DATE_GU  or rileva_colonna(orig, ["date_gu","data_gu","gu","gazzetta"])
    c_gio = COL_GIORNI  or rileva_colonna(orig, ["giorni_gu_decisione","giorni","durata"])
    c_es  = COL_ESITO   or rileva_colonna(orig, ["esito","dispositivo","tipo_pronuncia","decisione"])
    print(f"[i] colonne rilevate -> url:{c_url} | GU:{c_gu} | giorni:{c_gio} | esito:{c_es}")
    mancanti = [n for n,v in [("URL",c_url),("date_GU",c_gu),("giorni",c_gio),("esito",c_es)] if v is None]
    if mancanti:
        print(f"[!] Non ho riconosciuto: {mancanti}. Scrivi i nomi giusti nella sezione CONFIG e rilancia.")
        sys.exit(1)

    # target dal CSV originale
    orig = orig.copy()
    orig["_url"] = orig[c_url].map(norm_url)
    orig["data_gu"] = orig[c_gu].map(prima_data)                          # prima data GU (riferimento)
    orig["giorni_gu_decisione"] = pd.to_numeric(orig[c_gio], errors="coerce")  # TARGET durata (gia' calcolato)
    orig["esito"] = orig[c_es].map(classifica_esito)                      # TARGET esito
    orig_t = orig[["_url","data_gu","giorni_gu_decisione","esito"]]

    # join sull'URL
    feat = feat.copy()
    feat["_url"] = feat["URL_Pronuncia"].map(norm_url)
    df = feat.merge(orig_t, on="_url", how="left")

    # esclusione delle pronunce con esito misto
    n_join = len(df)
    misti = df["esito"] == "misto"
    print(f"\n[i] pronunce con esito misto escluse: {misti.sum()} su {n_join}")
    df = df[~misti]

    # ── diagnostica ──
    n = len(df)
    print(f"\n=== DIAGNOSTICA ===")
    print(f"Righe totali dopo il join e l'esclusione dei misti: {n}")
    print(f"Feature senza target (URL non combaciante): {df['esito'].isna().sum()}")
    print(f"giorni_gu_decisione mancanti: {df['giorni_gu_decisione'].isna().sum()}  |  "
          f"anomali (<0 o >4000): {((df['giorni_gu_decisione']<0)|(df['giorni_gu_decisione']>4000)).sum()}")
    print("\nDistribuzione esito:")
    print(df["esito"].value_counts(dropna=False).to_string())
    d = df["giorni_gu_decisione"].dropna()
    d = d[(d>=0)&(d<=4000)]
    if len(d):
        print(f"\ngiorni_gu_decisione — n={len(d)}  min={d.min():.0f}  mediana={d.median():.0f}  "
              f"media={d.mean():.0f}  max={d.max():.0f}")
    # campione per controllo mappatura esito
    print("\nEsempi (esito originale -> classe):")
    for _, r in orig[[c_es,"esito"]].drop_duplicates(subset=c_es).head(12).iterrows():
        print(f"   [{r['esito']:16s}] <- {str(r[c_es])[:70]}")

    # salva (senza la colonna tecnica _url)
    df.drop(columns=["_url"]).to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Salvato {OUTPUT_CSV} ({n} righe).")
    print("[i] Controlla la distribuzione esiti e gli esempi qui sopra: se una classe e' sbagliata,")
    print("    aggiusta KEYWORDS_ESITO nella CONFIG e rilancia.")


if __name__ == "__main__":
    main()