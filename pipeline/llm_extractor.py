"""
Script 2 — Estrattore LLM con Ollama + Qwen2.5 (locale) - Ordinato e Senza A Capo
=================================================================================
Legge il file JSONL prodotto dallo Script 1 (scraper_v2.py),
per ogni pronuncia manda a Qwen2.5:7b l'intestazione del testo integrale (campo 'testo'):
dall'inizio fino al primo tra "Ritenuto in fatto", "Considerato in diritto",
"Osservato in diritto", "per questi motivi", al massimo 3.000 caratteri.
Risolve i problemi di formattazione eliminando i ritorni a capo nei campi di testo
e riscrive il CSV mantenendo i dati ordinati (per Anno e Numero_Pronuncia).

Post-elaborazione deterministica dell'output del modello:
  - filtra_soggetti: toglie gli intervenienti (Presidente del Consiglio, Avvocatura
    dello Stato) dal campo Soggetto_Costituisce;
  - valida_giudice: il Giudice deve contenere almeno una parola chiave di un'autorità
    giudiziaria (stessa logica di pulisci_giudice.py);
  - se il Giudice non è valido, nuovo tentativo con il prompt dedicato al solo
    giudice (stessa logica di compila_giudice.py).

Modello e prompt principale sono quelli di aggiorna_storico.py
(llm_aggiorna_giudice_costituisce.py), con cui è stato prodotto il dataset.

UTILIZZO:
  python llm_extractor.py                 # elabora le pronunce non ancora nel CSV
  python llm_extractor.py --test 10       # solo le prime 10 nuove pronunce
  python llm_extractor.py --rifai-vuoti   # rielabora le righe del CSV con Giudice vuoto
                                          # (aggiorna solo Giudice e Soggetto_Costituisce)
  python llm_extractor.py --solo-filtri   # applica i filtri al CSV esistente, senza LLM

Il CSV viene salvato ogni 10 pronunce elaborate e subito in caso di Ctrl+C.
"""

import csv
import json
import os
import sys
import tempfile
import time
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ── Configurazione ──────────────────────────────────────────────────────────
INPUT_JSONL  = "pronunce_raw.jsonl"
OUTPUT_CSV   = "pronunce.csv"

MAX_CHARS_LLM   = 3000    # caratteri dell'intestazione inviati al modello
CHECKPOINT_OGNI = 10      # salva il CSV ogni N pronunce elaborate

FIELDNAMES = [
    "Numero_Pronuncia", "Tipo", "Anno", "Date_GU",
    "Camera_Consiglio", "Decisione_del", "Deposito_del",
    "Esito",
    "Giudice",
    "Soggetto_Costituisce",
    "Comunicato", "URL_Pronuncia",
]

# Intervenienti: compaiono quasi sempre ma NON sono parti costituite
INTERVENIENTI = [
    "presidente del consiglio",
    "avvocatura dello stato",
    "avvocatura generale",
    "avvocato generale dello stato",
    "avvocatura distrettuale",
]

# Parole chiave che identificano un'autorità giudiziaria valida (da pulisci_giudice.py)
KEYWORDS = [
    # Organi giudiziari ordinari (singolare e plurale)
    "tribunale", "tribunali",
    "corte", "corti",
    "pretore", "pretori",
    "pretura", "preture",
    "giudice", "giudici",
    "cassazione",
    "vicepretore",
    "conciliatore",
    # Organi amministrativi e speciali
    "consiglio", "commissione", "commissioni",
    "procura", "procuratore",
    "magistratura", "magistrato",
    "sezione",
    # Organi militari
    "militare", "capitaneria",
    # Organi arbitrali e speciali
    "collegio", "arbitro", "arbitrale",
    # Organi regionali e locali
    "commissario", "giunta", "prefettura", "prefetto",
    # Sigle e abbreviazioni ricorrenti
    "tar ", "trga", "cga",  # spazio dopo tar per non colpire "taranto"
    # Organi europei e sovranazionali
    "europea", "europeo",
    # Altri organi con funzione giurisdizionale
    "ente", "autorita", "autorità", "garante",
]

# ── Modello LLM — Ollama locale ──────────────────────────────────────────────
llm = ChatOpenAI(
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0,
    max_tokens=300,
)

# Modello per il secondo tentativo sui giudici non validi (come in compila_giudice.py)
llm_giudice = ChatOpenAI(
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0,
    max_tokens=200,
)

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM = """Sei un assistente specializzato nell'analisi di pronunce della Corte Costituzionale italiana.
Estrai le informazioni richieste e rispondi ESCLUSIVAMENTE con un oggetto JSON valido.
Non aggiungere testo prima o dopo il JSON. Non usare markdown, backtick o commenti."""

HUMAN = """Sei un estrattore di informazioni giuridiche. Analizza il testo di una pronuncia della Corte Costituzionale italiana e restituisci ESCLUSIVAMENTE un oggetto JSON con i due campi richiesti.

━━━ CAMPO 1: "giudice" ━━━
Identifica il o i giudici RIMETTENTI, cioè le autorità giudiziarie che hanno SOLLEVATO la questione di legittimità (il "giudice a quo"). NON è la Corte Costituzionale stessa.

COME TROVARLO:
• Formula singola: "promosso dal ...", "promossa dalla ...", "promosso con ordinanza del ..."
• Formula plurale con lista: "promossi con le seguenti ordinanze: 1) ... dal [GIUDICE] ... 2) ... dal [GIUDICE]"
• Formula plurale inline: "promossi con ordinanze rispettivamente emesse ... dal [GIUDICE] ... e dal [GIUDICE]"
• Se ci sono più giudici in lista numerata: estraili TUTTI, uno per voce.

COSA ESTRARRE: solo tipo organo + città. Esempi corretti:
  "Corte d'appello di Catanzaro"
  "Tribunale di Ferrara"
  "Pretore di Pieve di Cadore"
  "Commissione tributaria di primo grado di Mantova"

COSA ESCLUDERE: date, numeri di registro (n. 210, n. 8...), riferimenti GU, nomi di persone fisiche.
FORMATO: più giudici separati da " ; "

━━━ CAMPO 2: "soggetto_costituisce" ━━━
Identifica i soggetti che hanno depositato un atto formale di COSTITUZIONE di parte davanti alla Corte.

COME TROVARLO: cerca "Visti gli atti di costituzione di ...", "Visto l'atto di costituzione di ..."

REGOLA ASSOLUTA — IGNORA SEMPRE questi soggetti anche se compaiono dopo "costituzione":
  • Presidente del Consiglio dei ministri
  • Avvocatura dello Stato / Avvocatura generale dello Stato
  Questi sono INTERVENIENTI, mai parti costituite.

FORMATO: nomi separati da " ; ". Se nessun soggetto privato o regione si è costituito: scrivi esattamente "NO".

━━━ ESEMPI ━━━

Testo: "promosso con ordinanza dalla Corte d'appello di Catanzaro [...] Visti gli atti di costituzione di Barracco Maurizio e dell'Opera per la valorizzazione della Sila; visto l'atto di intervento del Presidente del Consiglio dei ministri"
→ {{"giudice": "Corte d'appello di Catanzaro", "soggetto_costituisce": "Barracco Maurizio ; Opera per la valorizzazione della Sila"}}

Testo: "promossi con le seguenti ordinanze: 1) ordinanza ... dal Tribunale di Ferrara [...] 2) ordinanza ... dal Tribunale di Varese [...] 3) ordinanza ... dal Pretore di Pieve di Cadore [...] Visto l'atto di intervento del Presidente del Consiglio dei Ministri"
→ {{"giudice": "Tribunale di Ferrara ; Tribunale di Varese ; Pretore di Pieve di Cadore", "soggetto_costituisce": "NO"}}

Testo: "promosso dal Tribunale di Milano [...] Visto l'atto di costituzione della Regione Lombardia; visto l'atto di intervento del Presidente del Consiglio"
→ {{"giudice": "Tribunale di Milano", "soggetto_costituisce": "Regione Lombardia"}}

Testo: "promosso con ordinanza del Pretore di Taranto nel procedimento" (nessun Visto)
→ {{"giudice": "Pretore di Taranto", "soggetto_costituisce": "NO"}}

Testo: "Visto l'atto di intervento del Presidente del Consiglio dei ministri" (solo intervento, nessuna costituzione)
→ {{"giudice": "...", "soggetto_costituisce": "NO"}}

━━━ TESTO DA ANALIZZARE ━━━
{testo}

Rispondi SOLO con il JSON, nient'altro:"""

# Prompt per il solo giudice (da compila_giudice.py)
HUMAN_GIUDICE = """Analizza il testo di una pronuncia della Corte Costituzionale e restituisci SOLO questo JSON:
{{"giudice": "..."}}

━━━ CAMPO "giudice" ━━━
Identifica il o i giudici RIMETTENTI: le autorità giudiziarie che hanno SOLLEVATO la questione
di legittimità costituzionale (il "giudice a quo"). NON è la Corte Costituzionale stessa.

COME TROVARLO:
• Formula singola:  "promosso dal ...", "promossa dalla ...", "promosso con ordinanza del ..."
• Lista numerata:   "promossi con le seguenti ordinanze: 1) ... dal [GIUDICE] ... 2) ... dal [GIUDICE]"
• Formula inline:   "promossi con ordinanze rispettivamente emesse ... dal [GIUDICE] ... e dal [GIUDICE]"
• Se ci sono più giudici: estraili TUTTI, uno per voce.

COSA ESTRARRE: solo tipo organo + città.
  Esempi: "Corte d'appello di Catanzaro", "Tribunale di Ferrara", "Pretore di Pieve di Cadore"

COSA ESCLUDERE: date, numeri di registro, riferimenti GU, nomi di persone fisiche.
FORMATO: più giudici separati da " ; "

━━━ ESEMPI ━━━

Testo: "promosso con ordinanza dalla Corte d'appello di Catanzaro nel procedimento..."
→ {{"giudice": "Corte d'appello di Catanzaro"}}

Testo: "promossi con le seguenti ordinanze: 1) ordinanza dal Tribunale di Ferrara [...] 2) ordinanza dal Tribunale di Varese [...] 3) ordinanza dal Pretore di Pieve di Cadore"
→ {{"giudice": "Tribunale di Ferrara ; Tribunale di Varese ; Pretore di Pieve di Cadore"}}

Testo: "promosso con ordinanza del Pretore di Taranto nel procedimento"
→ {{"giudice": "Pretore di Taranto"}}

━━━ TESTO ━━━
{testo}

Rispondi SOLO con il JSON:"""

# ── Chain LangChain ───────────────────────────────────────────────────────────

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", HUMAN),
])

chain = prompt | llm | JsonOutputParser()

prompt_giudice = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", HUMAN_GIUDICE),
])

chain_giudice = prompt_giudice | llm_giudice | JsonOutputParser()


# ── Estrazione LLM con retry ──────────────────────────────────────────────────

def estrai_con_llm(testo: str, max_retry: int = 3) -> dict:
    for tentativo in range(max_retry):
        try:
            result = chain.invoke({"testo": testo})
            return {
                "Giudice":               str(result.get("giudice", "")).strip(),
                "Soggetto_Costituisce": str(result.get("soggetto_costituisce", "")).strip(),
            }
        except KeyboardInterrupt:
            raise
        except Exception as e:
            errore = str(e)
            if "json" in errore.lower() or "parsing" in errore.lower():
                print(f"      [JSON] Tentativo {tentativo+1}/{max_retry} — riprovo")
            elif "connection" in errore.lower() or "refused" in errore.lower():
                print(f"      [ERR] Ollama non raggiungibile — assicurati che sia avviato")
                time.sleep(3)
            else:
                print(f"      [ERR] Tentativo {tentativo+1}/{max_retry}: {errore[:80]}")
                time.sleep(1)

    return {"Giudice": "", "Soggetto_Costituisce": ""}


def estrai_giudice_fallback(testo: str, max_retry: int = 3) -> str:
    """Secondo tentativo sul solo giudice con qwen2.5:7b (come compila_giudice.py)."""
    for tentativo in range(max_retry):
        try:
            result = chain_giudice.invoke({"testo": testo})
            return str(result.get("giudice", "")).strip()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            errore = str(e)
            if "json" in errore.lower() or "parsing" in errore.lower():
                print(f"      [JSON] Tentativo {tentativo+1}/{max_retry} — riprovo")
            elif "connection" in errore.lower() or "refused" in errore.lower():
                print(f"      [ERR] Ollama non raggiungibile")
                time.sleep(3)
            else:
                print(f"      [ERR] Tentativo {tentativo+1}/{max_retry}: {errore[:80]}")
                time.sleep(1)
    return ""


def pulisci_testo_da_invio(testo: str) -> str:
    """Rimuove i ritorni a capo e gli spazi multipli da qualsiasi stringa prima di salvarla"""
    if not testo:
        return ""
    # Sostituisce i ritorni a capo (\n e \r) e le tabulazioni con uno spazio semplice
    testo_pulito = re.sub(r'[\r\n\t]+', ' ', testo)
    # Collassa gli spazi multipli in un unico spazio
    testo_pulito = re.sub(r'\s+', ' ', testo_pulito)
    return testo_pulito.strip()


# ── Testo per il LLM e post-elaborazione ──────────────────────────────────────

def testo_per_llm(testo: str) -> str:
    """
    Intestazione processuale del testo integrale: dall'inizio fino al primo tra
    "Ritenuto in fatto", "Considerato in diritto", "Osservato in diritto",
    "per questi motivi", al massimo MAX_CHARS_LLM caratteri.
    """
    fine_pos = len(testo)
    for pat in [r"\bRitenuto\s+in\s+fatto\b", r"\bConsiderato\s+in\s+diritto\b",
                r"\bOsservato\s+in\s+diritto\b", r"\bper\s+questi\s+motivi\b"]:
        mf = re.search(pat, testo, re.I)
        if mf and mf.start() < fine_pos:
            fine_pos = mf.start()

    sezione = testo[:fine_pos].strip()
    sezione = re.sub(r"\n{3,}", "\n\n", sezione)
    sezione = re.sub(r" {2,}", " ", sezione)
    return sezione[:MAX_CHARS_LLM]


def filtra_soggetti(soggetto: str) -> str:
    """
    Toglie gli intervenienti dal campo Soggetto_Costituisce. Se non resta nessun
    soggetto restituisce "NO". Un campo vuoto (LLM non ha risposto) resta vuoto.
    """
    if not soggetto or soggetto.strip().upper() == "NO":
        return soggetto
    parti = [p.strip() for p in soggetto.split(";")]
    filtrate = [p for p in parti
                if p and not any(iv in p.lower() for iv in INTERVENIENTI)]
    if len(filtrate) == len([p for p in parti if p]):
        return soggetto                     # nessun interveniente: valore invariato
    return " ; ".join(filtrate) if filtrate else "NO"


def valida_giudice(giudice: str) -> str:
    """
    Restituisce il giudice se contiene almeno una parola chiave di un'autorità
    giudiziaria (confronto case-insensitive, come pulisci_giudice.py), altrimenti "".
    """
    if not giudice:
        return ""
    testo = giudice.lower()
    return giudice if any(kw in testo for kw in KEYWORDS) else ""


def estrai(testo: str) -> dict:
    """Estrazione con il modello principale, filtri e fallback sul giudice."""
    sezione = testo_per_llm(testo)
    llm_result = estrai_con_llm(sezione)

    giudice = valida_giudice(pulisci_testo_da_invio(llm_result["Giudice"]))
    if not giudice:
        print(f"      [FALLBACK] Giudice non valido — riprovo con il prompt del solo giudice")
        giudice = valida_giudice(pulisci_testo_da_invio(estrai_giudice_fallback(sezione)))

    soggetto = filtra_soggetti(pulisci_testo_da_invio(llm_result["Soggetto_Costituisce"]))
    return {"Giudice": giudice, "Soggetto_Costituisce": soggetto}


# ── Salvataggio ───────────────────────────────────────────────────────────────

def chiave_ordinamento(x):
    try:
        # Converte in interi per evitare problemi di ordinamento stringa (es: "100" < "20")
        return (int(x.get("Anno", 0)), int(x.get("Numero_Pronuncia", 0)))
    except ValueError:
        return (0, 0)


def salva_csv(csv_database: list):
    """
    Ordina per Anno e Numero_Pronuncia (decrescenti) e riscrive il CSV.
    La scrittura avviene su un file temporaneo nella stessa cartella, poi
    os.replace lo sostituisce all'originale in un'unica operazione atomica:
    un'interruzione durante la scrittura non lascia mai un CSV a metà.
    """
    csv_database.sort(key=chiave_ordinamento, reverse=True)
    cartella = os.path.dirname(os.path.abspath(OUTPUT_CSV))
    fd, tmp_path = tempfile.mkstemp(dir=cartella, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";",
                                    quoting=csv.QUOTE_MINIMAL, extrasaction="ignore")
            writer.writeheader()
            for riga_salvata in csv_database:
                writer.writerow(riga_salvata)
        os.replace(tmp_path, OUTPUT_CSV)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    max_righe = None
    if "--test" in sys.argv:
        idx = sys.argv.index("--test")
        try:
            max_righe = int(sys.argv[idx + 1])
            print(f"[i] Modalità TEST: prime {max_righe} pronunce")
        except (IndexError, ValueError):
            print("[!] --test richiede un numero. Esempio: --test 10"); sys.exit(1)
    rifai_vuoti = "--rifai-vuoti" in sys.argv
    solo_filtri = "--solo-filtri" in sys.argv

    # 1. Carichiamo tutto il CSV esistente in memoria per aggiornarlo / ordinarlo
    csv_database = []
    seen_urls = set()

    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                if row.get("URL_Pronuncia"):
                    seen_urls.add(row["URL_Pronuncia"].strip())
                    # Puliamo preventivamente le righe vecchie se contenevano errori di invio a capo
                    for campo in row:
                        row[campo] = pulisci_testo_da_invio(row[campo])
                    csv_database.append(row)
        print(f"[i] Caricate {len(csv_database)} pronunce esistenti dal CSV per ordinamento")

    # ── --solo-filtri: applica i filtri al CSV esistente, senza LLM ───────
    if solo_filtri:
        soggetti_corretti = giudici_svuotati = 0
        for row in csv_database:
            s = filtra_soggetti(row.get("Soggetto_Costituisce", ""))
            g = valida_giudice(row.get("Giudice", ""))
            soggetti_corretti += s != row.get("Soggetto_Costituisce", "")
            giudici_svuotati  += g != row.get("Giudice", "")
            row["Soggetto_Costituisce"], row["Giudice"] = s, g
        salva_csv(csv_database)
        print(f"[OK] Soggetti corretti: {soggetti_corretti} | Giudici svuotati: {giudici_svuotati}")
        print(f"[i] Per ricompilare i giudici svuotati: python llm_extractor.py --rifai-vuoti")
        return

    if not os.path.exists(INPUT_JSONL):
        print(f"[!] File '{INPUT_JSONL}' non trovato.")
        sys.exit(1)

    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=3)
    except Exception:
        print("[!] Ollama non raggiungibile su localhost:11434")
        sys.exit(1)

    # 2. Lettura del file JSONL delle nuove pronunce
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        righe_jsonl = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    righe_jsonl.append(json.loads(line))
                except Exception as e:
                    print(f"  [!] Riga JSONL non valida: {e}")

    da_processare = [r for r in righe_jsonl if r.get("URL_Pronuncia", "") not in seen_urls]

    # --rifai-vuoti: anche le righe già nel CSV con Giudice vuoto
    righe_csv_per_url = {r["URL_Pronuncia"]: r for r in csv_database}
    if rifai_vuoti:
        da_processare += [r for r in righe_jsonl
                          if r.get("URL_Pronuncia", "") in righe_csv_per_url
                          and not righe_csv_per_url[r["URL_Pronuncia"]].get("Giudice")]
    print(f"[i] Pronunce da elaborare: {len(da_processare)}")

    if max_righe:
        da_processare = da_processare[:max_righe]

    # 3. Elaborazione delle pronunce
    elaborate = 0
    try:
        for i, row in enumerate(da_processare, 1):
            url    = row.get("URL_Pronuncia", "")
            numero = row.get("Numero_Pronuncia", "?")
            anno   = row.get("Anno", "?")
            testo  = (row.get("testo") or row.get("testo_pronuncia") or "").strip()

            print(f"\n  [{i}/{len(da_processare)}] Elaborazione n.{numero}/{anno}")

            if testo:
                llm_result = estrai(testo)
            else:
                print(f"    [!] testo vuoto")
                llm_result = {"Giudice": "", "Soggetto_Costituisce": ""}

            if url in righe_csv_per_url:
                # Riga già nel CSV (--rifai-vuoti): si aggiornano SOLO i due campi LLM,
                # così le correzioni manuali agli altri campi restano intatte
                riga = righe_csv_per_url[url]
                riga["Giudice"] = llm_result["Giudice"]
                if llm_result["Soggetto_Costituisce"]:
                    riga["Soggetto_Costituisce"] = llm_result["Soggetto_Costituisce"]
            else:
                # Costruiamo la riga ed eliminiamo categoricamente i ritorni a capo da ogni campo
                riga = {}
                for campo in FIELDNAMES:
                    if campo in ("Giudice", "Soggetto_Costituisce"):
                        valore = llm_result[campo]
                    else:
                        valore = row.get(campo, "")

                    # Applica la rimozione dei ritorni a capo
                    riga[campo] = pulisci_testo_da_invio(str(valore))

                # Aggiungiamo la riga elaborata al nostro database temporaneo in memoria
                csv_database.append(riga)
                righe_csv_per_url[url] = riga
                seen_urls.add(url)

            print(f"    Giudice:   {riga['Giudice'][:70]}")
            print(f"    Esito:     {riga['Esito'][:70]}")

            elaborate += 1
            if elaborate % CHECKPOINT_OGNI == 0:
                salva_csv(csv_database)
                print(f"    [✓] Checkpoint: {elaborate}/{len(da_processare)} salvate")

    except KeyboardInterrupt:
        print(f"\n[!] Interrotto — salvo le {elaborate} pronunce elaborate...")

    finally:
        # 4. ORDINAMENTO E RISCRITTURA COMPLETA DEL FILE (anche dopo Ctrl+C)
        salva_csv(csv_database)
        print(f"\n[i] Ordinamento completato su {len(csv_database)} righe totali.")

    print(f"\n{'='*60}")
    print(f"[OK] Database riorganizzato e salvato in: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
