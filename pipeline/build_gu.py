"""
build_gu.py — Costruisce l'indice locale delle Gazzette Ufficiali
========================================================================
Naviga il sito GU per ogni anno, recupera numero e data di ogni GU
e salva in gu_index.csv.

SITI USATI:
  Anni 1956-1985 → sito legacy PDF:
    https://www.gazzettaufficiale.it/ricerca/pdf/corte_costituzionale_legacy/2/0/0
  Anni 1986-2026 → sito corrente 1a Serie Speciale:
    https://www.gazzettaufficiale.it/ricerca/pdf/corte_costituzionale/3/0/0

UTILIZZO:
  python build_gu.py              # tutti gli anni 1956-2026
  python build_gu.py 2024         # solo 2024
  python build_gu.py 1970 1985    # solo anni legacy
  python build_gu.py 2020 2024    # dal 2020 al 2024 inclusi

OUTPUT:
  gu_index.csv  →  colonne: numero, anno, data  (es: 15,2025,09/04/2025)
"""

import csv
import os
import random
import re
import sys
import time

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc

# ── Configurazione ────────────────────────────────────────────────────────────
GU_URL     = ("https://www.gazzettaufficiale.it/ricerca/pdf/corte_costituzionale/3/0/0?reset=true")
OUTPUT_CSV = "gu_index.csv"
FIELDNAMES = ["numero", "anno", "data"]

MESI_IT = {
    "gennaio":1, "febbraio":2, "marzo":3, "aprile":4,
    "maggio":5, "giugno":6, "luglio":7, "agosto":8,
    "settembre":9, "ottobre":10, "novembre":11, "dicembre":12
}


# ── Parser risultati ──────────────────────────────────────────────────────────

def parse_righe_gu(page_source: str, anno: int) -> list:
    """
    Estrae tutte le coppie (numero_gu, data) dalla pagina risultati GU.

    FIX rispetto alla versione precedente:
    - Rimosso il limite num <= 99: negli anni storici (Serie Generale)
      i numeri GU arrivano fino a 300+
    - Aggiunto fallback in lettere per anni storici
    """
    soup  = BeautifulSoup(page_source, "lxml")
    testo = soup.get_text(" ", strip=True)
    righe = []
    visti = set()

    # Pattern principale: "n° X del GG-MM-AAAA"
    PAT = re.compile(
        r"n[°o]\s*(\d+)\s+del\s+(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})",
        re.I)
    for m in PAT.finditer(testo):
        num = int(m.group(1))
        g   = int(m.group(2))
        me  = int(m.group(3))
        a   = int(m.group(4))
        # Sanity: anno coerente (±1), mese e giorno validi, numero >= 1
        if abs(a - anno) <= 1 and num >= 1 and 1 <= me <= 12 and 1 <= g <= 31:
            chiave = (num, anno)
            if chiave not in visti:
                visti.add(chiave)
                righe.append({"numero": num, "anno": anno,
                               "data": f"{g:02d}/{me:02d}/{a}"})

    # Fallback: "n° X del GG mese AAAA" (lettere, raro)
    if not righe:
        PAT_LETT = re.compile(
            r"n[°o]\s*(\d+)\s+del\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
            re.I)
        for m in PAT_LETT.finditer(testo):
            num = int(m.group(1))
            g   = int(m.group(2))
            ms  = m.group(3).lower()
            a   = int(m.group(4))
            if ms in MESI_IT and abs(a - anno) <= 1 and num >= 1:
                chiave = (num, anno)
                if chiave not in visti:
                    visti.add(chiave)
                    righe.append({"numero": num, "anno": anno,
                                   "data": f"{g:02d}/{MESI_IT[ms]:02d}/{a}"})

    return sorted(righe, key=lambda x: x["numero"])


# ── Paginazione ───────────────────────────────────────────────────────────────

def get_next_page_url(driver) -> str | None:
    """
    Cerca il link alla pagina successiva nei risultati GU.
    Restituisce l'URL se esiste, None se siamo all'ultima pagina.
    """
    try:
        # Cerca link "Successiva", "»", "next" o numero pagina successiva
        for testo_link in ["Successiva", "successive", "»", ">", "next"]:
            els = driver.find_elements(
                By.XPATH,
                f"//a[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                f"'abcdefghijklmnopqrstuvwxyz'), '{testo_link.lower()}')]")
            for el in els:
                href = el.get_attribute("href")
                if href and "corte_costituzionale" in href:
                    return href

        # Alternativa: cerca link con "pagina" o "page" nell'href
        els = driver.find_elements(
            By.XPATH,
            "//a[contains(@href,'pagina') or contains(@href,'page') "
            "or contains(@href,'pag=') or contains(@href,'start=')]")
        for el in els:
            href = el.get_attribute("href")
            # Prendi solo quello che punta a una pagina successiva
            # (semplice euristica: non è quello corrente)
            if href and "corte_costituzionale" in href:
                return href
    except Exception:
        pass
    return None


# ── Navigazione per anno ──────────────────────────────────────────────────────

# URL dei due siti GU
GU_URL_LEGACY = ("https://www.gazzettaufficiale.it/ricerca/pdf/"
                  "corte_costituzionale_legacy/2/0/0?reset=true")


def scarica_anno_gu(driver, anno: int) -> list:
    """
    Scarica le GU di un anno.
    - Anni fino al 1985: usa il sito legacy (PDF storici)
    - Anni 1986+: usa il sito corrente (1a Serie Speciale)
    Il formato dei risultati è identico in entrambi i casi.
    """
    if anno <= 1985:
        return _scarica_anno_legacy(driver, anno)
    else:
        return _scarica_anno_corrente(driver, anno)


def _scarica_anno_legacy(driver, anno: int) -> list:
    """
    Usa il sito legacy per gli anni 1970-1985.
    Form: numero, giorno, mese, anno (select).
    Cerca solo per anno → lista di tutte le GU dell'anno.
    """
    print(f"  [GU-LEGACY] Anno {anno}...", end=" ", flush=True)

    driver.get(GU_URL_LEGACY)
    wait = WebDriverWait(driver, 15)
    time.sleep(1.5)

    # Il sito legacy ha un select per l'anno con valori 1956-1985
    # Nome del campo: da ispezionare → proviamo i nomi comuni
    for nome_campo in ["annoPubblicazione", "anno", "annoRicerca", "year"]:
        try:
            sel_el = driver.find_element(By.NAME, nome_campo)
            opzioni = [o.get_attribute("value")
                       for o in sel_el.find_elements(By.TAG_NAME, "option")]
            if str(anno) in opzioni:
                Select(sel_el).select_by_value(str(anno))
                break
            elif opzioni:
                # Campo trovato ma anno non nelle opzioni — strano
                print(f"[!] Anno {anno} non nel dropdown ({nome_campo})")
        except Exception:
            continue
    else:
        # Nessun campo trovato: prova a cliccare direttamente l'opzione
        try:
            opt = driver.find_element(
                By.XPATH, f"//option[@value='{anno}' or text()='{anno}']")
            driver.execute_script("arguments[0].selected = true;", opt)
            driver.execute_script(
                "arguments[0].dispatchEvent(new Event('change'));", opt)
        except Exception as e:
            print(f"Impossibile selezionare anno: {e}")
            return []

    time.sleep(0.3)

    # Cerca il bottone di ricerca
    for nome_btn in ["cerca", "submit", "search", "ricerca"]:
        try:
            btn = driver.find_element(By.NAME, nome_btn)
            driver.execute_script("arguments[0].click();", btn)
            break
        except Exception:
            continue
    else:
        try:
            btn = driver.find_element(
                By.CSS_SELECTOR, "input[type='submit'], button[type='submit']")
            driver.execute_script("arguments[0].click();", btn)
        except Exception as e:
            print(f"Bottone non trovato: {e}")
            return []

    time.sleep(random.uniform(3.0, 4.5))
    return _raccogli_pagine(driver, anno)


def _scarica_anno_corrente(driver, anno: int) -> list:
    """
    Usa il sito corrente per gli anni 1986+.
    Identico alla vecchia scarica_anno_gu.
    """
    print(f"  [GU] Anno {anno}...", end=" ", flush=True)

    driver.get(GU_URL)
    wait = WebDriverWait(driver, 15)
    wait.until(EC.presence_of_element_located((By.NAME, "annoPubblicazione")))
    time.sleep(1.0)

    sel_el  = driver.find_element(By.NAME, "annoPubblicazione")
    opzioni = [o.get_attribute("value")
               for o in sel_el.find_elements(By.TAG_NAME, "option")]

    if str(anno) in opzioni:
        Select(sel_el).select_by_value(str(anno))
    else:
        driver.execute_script(
            "var opt = document.createElement('option');"
            "opt.value = arguments[0]; opt.text = arguments[0];"
            "arguments[1].appendChild(opt);",
            str(anno), sel_el)
        try:
            Select(sel_el).select_by_value(str(anno))
        except Exception:
            print("non disponibile nel dropdown — skip")
            return []

    time.sleep(0.3)
    btn = driver.find_element(By.NAME, "cerca")
    driver.execute_script("arguments[0].click();", btn)
    time.sleep(random.uniform(3.0, 4.5))
    return _raccogli_pagine(driver, anno)


def _raccogli_pagine(driver, anno: int) -> list:
    """
    Raccoglie tutte le pagine di risultati da qualsiasi sito GU.
    Il formato dei risultati è identico: "n° X del GG-MM-AAAA"
    """
    tutte_righe  = []
    pagina       = 1
    pagine_viste = set()

    while True:
        src          = driver.page_source
        url_corrente = driver.current_url

        if url_corrente in pagine_viste:
            break
        pagine_viste.add(url_corrente)

        testo = BeautifulSoup(src, "lxml").get_text(" ", strip=True)

        if "nessun risultato" in testo.lower() or "nessuna gazzetta" in testo.lower():
            if pagina == 1:
                print("nessun risultato")
                return []
            break

        righe_pag = parse_righe_gu(src, anno)

        if not righe_pag:
            if pagina == 1:
                idx = testo.find("n°")
                frammento = testo[idx:idx+400] if idx >= 0 else testo[:400]
                print(f"\n    [!] Parser non ha trovato righe. Frammento:")
                print(f"    {frammento}")
            break

        numeri_gia_visti = {r["numero"] for r in tutte_righe}
        nuove = [r for r in righe_pag if r["numero"] not in numeri_gia_visti]
        tutte_righe.extend(nuove)

        if pagina == 1:
            print(f"pag.1({len(righe_pag)})", end=" ", flush=True)
        else:
            print(f"pag.{pagina}({len(righe_pag)})", end=" ", flush=True)

        next_url = get_next_page_url(driver)
        if not next_url or next_url in pagine_viste:
            break

        driver.get(next_url)
        time.sleep(random.uniform(2.5, 3.5))
        pagina += 1

        if pagina > 20:
            break

    if tutte_righe:
        tutte_righe.sort(key=lambda x: x["numero"])
        print(f"→ TOT {len(tutte_righe)} GU "
              f"(n.{tutte_righe[0]['numero']} → n.{tutte_righe[-1]['numero']})")
    return tutte_righe


# ── Driver ────────────────────────────────────────────────────────────────────

def create_driver():
    opts = uc.ChromeOptions()
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    return uc.Chrome(options=opts, version_main=153)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        anni = list(range(1956, 2027))
    elif len(args) == 1:
        try:
            anno = int(args[0])
            if not (1956 <= anno <= 2026):
                print(f"[!] Anno non valido: {anno}"); sys.exit(1)
            anni = [anno]
        except ValueError:
            print(f"[!] Argomento non valido: '{args[0]}'"); sys.exit(1)
    elif len(args) == 2:
        try:
            a_ini, a_fin = int(args[0]), int(args[1])
            if not (1956 <= a_ini <= a_fin <= 2026):
                print(f"[!] Range non valido."); sys.exit(1)
            anni = list(range(a_ini, a_fin + 1))
        except ValueError:
            print(f"[!] Argomenti non validi."); sys.exit(1)
    else:
        print("[!] Uso: python build_gu.py [anno] o [anno_inizio anno_fine]")
        sys.exit(1)

    print(f"[i] Anni: {anni[0]} → {anni[-1]} ({len(anni)} anni)")
    print(f"[i] Output: {OUTPUT_CSV}")

    # ── Carica voci già presenti (ripresa) ────────────────────────────────
    esistenti        = set()
    anni_completi    = set()
    conteggio_x_anno = {}

    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    n, a = int(row["numero"]), int(row["anno"])
                    esistenti.add((n, a))
                    conteggio_x_anno[a] = conteggio_x_anno.get(a, 0) + 1
                except Exception:
                    pass
        # Anno completo se ha ≥ 40 voci (anni recenti ~50, storici meno ma > 5)
        # Soglia bassa per non ri-scaricare anni storici con poche GU
        anni_completi = {a for a, c in conteggio_x_anno.items() if c >= 5}
        print(f"[i] Ripresa: {len(esistenti)} voci, {len(anni_completi)} anni già scaricati")

    anni_da_fare = [a for a in anni if a not in anni_completi]

    if not anni_da_fare:
        print("[OK] Tutti gli anni già presenti. Nulla da fare.")
        return

    for a in anni:
        if a in anni_completi:
            print(f"  [=] {a} già scaricato — skip")

    print(f"[i] Anni da scaricare: {len(anni_da_fare)}")

    # Crea CSV con intestazione se non esiste
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    driver     = create_driver()
    totale_new = 0

    try:
        for anno in anni_da_fare:
            try:
                righe = scarica_anno_gu(driver, anno)
                nuove = 0
                if righe:
                    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
                        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                        for r in righe:
                            chiave = (r["numero"], r["anno"])
                            if chiave not in esistenti:
                                w.writerow(r)
                                esistenti.add(chiave)
                                nuove += 1
                    totale_new += nuove
            except Exception as e:
                import traceback
                print(f"\n  [!] Errore anno {anno}: {e}")
                traceback.print_exc()

            time.sleep(random.uniform(1.5, 2.5))

    except KeyboardInterrupt:
        print("\n[!] Interrotto dall'utente")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    print(f"\n[OK] {totale_new} nuove righe aggiunte → {OUTPUT_CSV}")
    print(f"[i] Prossimo passo: python scraper_v2.py [anno]")


if __name__ == "__main__":
    main()