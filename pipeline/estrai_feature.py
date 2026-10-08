"""
Estrattore feature per le pronunce della Corte costituzionale (giudizi incidentali)
===================================================================================
Legge il JSONL grezzo dello scraper (pronunce_raw.jsonl: URL_Pronuncia, Numero_Pronuncia,
Anno, testo; e' accettato anche il formato storico url, numero, anno, testo) ed estrae le
feature strutturali della questione. I campi categorici hanno un VOCABOLARIO CHIUSO:
il prompt spiega come riconoscere ogni etichetta, lo schema impone che l'output sia
una di quelle ammesse. Cosi' l'LLM classifica (capendo testo scritto in modi diversi)
invece di inventare stringhe non uniformi.

Pipeline: pulizia spazi -> taglio a 'Considerato' (epigrafe + Ritenuto) -> estrazione
strutturata -> rete di sicurezza sui parametri (LLM vs regex). Resume via JSONL.
Provider via LangChain (Groq / Gemini / OpenAI).

Dipendenze:
    pip install langchain-core pydantic
    pip install langchain-groq            # provider = "groq"
    pip install langchain-google-genai    # provider = "gemini"
    pip install langchain-openai          # provider = "openai"
Chiave API via env: GROQ_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY

Uso:
    python estrai_feature.py --test 10
    python estrai_feature.py
"""

import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# ── Configurazione ────────────────────────────────────────────────────────────
INPUT_JSONL = "pronunce_raw.jsonl"          # prodotto da scraper_v2.py
OUTPUT_JSONL = "feature_estratte.jsonl"
OUTPUT_CSV = "feature_estratte.csv"

PROVIDER = "gemini"                          # "groq" | "gemini" | "openai"
MODEL = "gemini-2.5-flash"
MAX_WORKERS = 4
MAX_CHARS = 6000                            # le feature stanno nei primi ~3000 char; 6000 e' un margine sicuro
LUNGHEZZA_BASE = 6000                       # se NON trova il divisore "Considerato", manda i primi N caratteri
MAX_RETRY = 4
ANNO_DA = None                              # es. 2000 -> analizza dal 2000 in poi; None = tutte. Sovrascrivibile con --anno-da
REQ_PER_SEC = 2.0                          # limite proattivo di richieste/secondo (ABBASSALO a 1.0/0.5 se vedi errori 429/rate)


# ── Vocabolari chiusi (enum) ──────────────────────────────────────────────────
class TipoAtto(str, Enum):
    legge_statale = "legge_statale"
    legge_regionale = "legge_regionale"
    decreto_legislativo = "decreto_legislativo"
    decreto_legge = "decreto_legge"
    dpr = "dPR"
    regio_decreto = "regio_decreto"        # incl. regio decreto-legge; frequente nei testi ante-2000
    codice = "codice"                       # civile, penale, proc. civ., proc. pen.
    altro = "altro"

class AutoritaRimettente(str, Enum):
    tribunale_ordinario = "tribunale_ordinario"
    corte_appello = "corte_appello"
    cassazione = "cassazione"
    giudice_di_pace = "giudice_di_pace"
    gip_gup = "gip_gup"
    tar = "tar"
    consiglio_di_stato = "consiglio_di_stato"
    corte_conti = "corte_conti"
    giustizia_tributaria = "giustizia_tributaria"
    tribunale_sorveglianza = "tribunale_sorveglianza"
    magistrato_sorveglianza = "magistrato_sorveglianza"
    tribunale_minorenni = "tribunale_minorenni"
    tribunale_militare = "tribunale_militare"
    corte_assise = "corte_assise"
    giudice_tutelare = "giudice_tutelare"
    pretore = "pretore"
    altro = "altro"

class Giurisdizione(str, Enum):
    penale = "penale"
    civile = "civile"
    amministrativa = "amministrativa"
    tributaria = "tributaria"
    contabile = "contabile"
    altro = "altro"
    non_determinabile = "non_determinabile"

class ParametroInterposto(str, Enum):
    CEDU = "CEDU"
    diritto_UE = "diritto_UE"               # TFUE, direttive, regolamenti, Carta di Nizza/CDFUE
    legge_costituzionale = "legge_costituzionale"
    altro_trattato = "altro_trattato"

class Materia(str, Enum):
    penale = "penale"
    processuale = "processuale"
    civile = "civile"
    tributaria = "tributaria"
    previdenziale = "previdenziale"
    lavoro = "lavoro"
    edilizia_urbanistica = "edilizia_urbanistica"
    ambiente = "ambiente"
    immigrazione = "immigrazione"
    famiglia_persone = "famiglia_persone"
    elettorale = "elettorale"
    esecuzione_forzata = "esecuzione_forzata"
    circolazione_stradale = "circolazione_stradale"
    stupefacenti = "stupefacenti"
    sanita = "sanita"
    pubblico_impiego = "pubblico_impiego"
    ordinamento_enti_territoriali = "ordinamento_enti_territoriali"
    concorsuale = "concorsuale"
    societario_commerciale = "societario_commerciale"
    altro = "altro"


class FeaturePronuncia(BaseModel):
    tipi_atto_impugnati: List[TipoAtto] = Field(default_factory=list, description="Vedi istruzione 1.")
    autorita_rimettente_tipo: AutoritaRimettente = Field(description="Vedi istruzione 2.")
    giurisdizione_a_quo: Giurisdizione = Field(description="Vedi istruzione 3.")
    parametri_costituzionali: List[int] = Field(default_factory=list, description="Vedi istruzione 4.")
    parametri_interposti: List[ParametroInterposto] = Field(default_factory=list, description="Vedi istruzione 5.")
    materia: Materia = Field(description="Vedi istruzione 6.")


SYSTEM = (
    "Sei un giurista esperto di giustizia costituzionale italiana. Dalla parte iniziale "
    "(epigrafe + 'Ritenuto') di una pronuncia estrai le feature strutturali, ragionando sul "
    "significato giuridico (il testo puo' avere spaziature irregolari: ricostruisci dal contesto). "
    "Per i campi a scelta chiusa restituisci ESATTAMENTE una delle etichette ammesse."
)

HUMAN = """Analizza il testo ed estrai le feature. Per i campi a scelta chiusa scegli SOLO tra le etichette elencate.

1. tipi_atto_impugnati — tipo di fonte/i impugnata/e (dopo "legittimita' costituzionale dell'art. ... del/della ..."); elenca tutti gli atti.
   legge_statale | legge_regionale | decreto_legislativo | decreto_legge | dPR | regio_decreto | codice | altro
   Regole: un codice resta "codice" anche se emanato con decreto legislativo (es. Codice crisi d'impresa); "testo unico" -> classifica per l'atto che lo emana (dPR o decreto_legislativo); regio decreto(-legge) -> regio_decreto; legge di Provincia autonoma -> legge_regionale.

2. autorita_rimettente_tipo — tipo di organo rimettente (dopo "promosso ... dal ..."); ignora citta'/sezione/nomi; se piu' rimettenti, il primo.
   tribunale_ordinario | corte_appello | cassazione | giudice_di_pace | gip_gup | tar | consiglio_di_stato | corte_conti | giustizia_tributaria | tribunale_sorveglianza | magistrato_sorveglianza | tribunale_minorenni | tribunale_militare | corte_assise | giudice_tutelare | pretore | altro
   Regole: Consiglio di giustizia amministrativa siciliana -> consiglio_di_stato; TRGA Trento/Bolzano -> tar; commissione tributaria (anche centrale) -> giustizia_tributaria; sezione specializzata impresa -> tribunale_ordinario; corti militari di ogni grado (anche Corte militare d'appello) -> tribunale_militare; corte d'assise anche d'appello -> corte_assise; GIP/GUP presso il Tribunale per i minorenni -> tribunale_minorenni, presso il Tribunale militare -> tribunale_militare; organi speciali senza categoria propria -> altro.

3. giurisdizione_a_quo — natura del processo a quo: "procedimento penale"->penale; "vertente tra"/opposizione/esecuzione->civile; TAR/Consiglio di Stato->amministrativa; giudice tributario->tributaria; Corte dei conti->contabile.
   penale | civile | amministrativa | tributaria | contabile | altro | non_determinabile

4. parametri_costituzionali — SOLO i numeri degli articoli della Costituzione invocati come parametro ("in riferimento agli artt. ... della Costituzione"). NON gli articoli della norma impugnata, NON leggi cost./trattati (punto 5). Lista di interi.

5. parametri_interposti — norme non costituzionali INVOCATE come parametro (di regola via art. 117 primo comma, o "in relazione a"). Mappa ognuna:
   CEDU | diritto_UE | legge_costituzionale | altro_trattato
   CEDU (Convenzione europea) e Carta di Nizza/CDFUE sono DIVERSE: la Carta UE e direttive/regolamenti/TFUE -> diritto_UE. NON includere norme UE/internazionali solo citate nel titolo dell'atto o negli argomenti delle parti (tipicamente manca il 117 tra i parametri). Nessun interposto -> lista vuota.

6. materia — ambito sostanziale, dedotto dall'oggetto della norma e dal processo a quo. UNA etichetta:
   penale | processuale | civile | tributaria | previdenziale | lavoro | edilizia_urbanistica | ambiente | immigrazione | famiglia_persone | elettorale | esecuzione_forzata | circolazione_stradale | stupefacenti | sanita | pubblico_impiego | ordinamento_enti_territoriali | concorsuale | societario_commerciale | altro
   competenza/impugnazione/ricusazione -> processuale; dissesto/incandidabilita' enti locali -> ordinamento_enti_territoriali; esdebitazione/liquidazione -> concorsuale.

TESTO:
{testo}"""

prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])


# ── Provider LLM (LangChain) ──────────────────────────────────────────────────
def build_chain():
    from langchain_core.rate_limiters import InMemoryRateLimiter
    limiter = InMemoryRateLimiter(requests_per_second=REQ_PER_SEC,
                                  check_every_n_seconds=0.1,
                                  max_bucket_size=max(1, int(REQ_PER_SEC * 2)))
    if PROVIDER == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=MODEL, temperature=0, rate_limiter=limiter)
    elif PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # disattiva il "thinking" (i token di ragionamento si pagano come output; per una
        # classificazione non servono). Se la versione della libreria non lo supporta, fallback.
        try:
            llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0, rate_limiter=limiter, thinking_budget=0)
        except TypeError:
            llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0, rate_limiter=limiter)
    elif PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=MODEL, temperature=0, rate_limiter=limiter)
    else:
        raise ValueError(f"PROVIDER non valido: {PROVIDER}")
    return prompt | llm.with_structured_output(FeaturePronuncia)


# ── Pulizia testo + taglio (testati sui testi reali) ──────────────────────────
def normalizza_spazi(t: str) -> str:
    t = re.sub(r'[\r\n\t]+', ' ', t)
    t = re.sub(r' {2,}', ' ', t)
    return t.strip()

def _flex(parola: str) -> str:
    return r'\s*'.join(re.escape(c) for c in parola)

def _marker(frase: str) -> str:
    return r'\s+'.join(_flex(w) for w in frase.split())

MARKERS = [_marker('considerato in diritto'), _marker('considerato che'), _marker('per questi motivi')]

def parte_rimessione(t: str) -> str:
    """Normalizza gli spazi e tiene epigrafe + 'Ritenuto' tagliando al divisore 'Considerato'/
    'per questi motivi'. Se il divisore NON c'e' (pronunce non standard), manda i primi
    LUNGHEZZA_BASE caratteri (che contengono comunque epigrafe e parametri)."""
    t = normalizza_spazi(t)
    for pat in MARKERS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return t[:m.start()].strip()[:MAX_CHARS]
    return t.strip()[:LUNGHEZZA_BASE]

def parametri_regex(t: str) -> List[int]:
    found = set()
    for m in re.finditer(r'(della\s+Costituzione|Cost\.)', t, flags=re.IGNORECASE):
        window = t[max(0, m.start() - 90):m.start()]
        arts = list(re.finditer(r'\bart\w*\.?', window, flags=re.IGNORECASE))
        if not arts:
            continue
        seg = window[arts[-1].end():]
        seg = re.sub(r'comma\s+\d+', '', seg, flags=re.IGNORECASE)
        for num in re.findall(r'\d+', seg):
            found.add(int(num))
    return sorted(found)


# ── Estrazione di una pronuncia ───────────────────────────────────────────────
def estrai_una(chain, rec: dict) -> dict:
    url = rec.get("url", "")
    testo = parte_rimessione(rec.get("testo", "") or "")
    base = {"URL_Pronuncia": url, "numero": rec.get("numero", ""), "anno": rec.get("anno", "")}
    reg = parametri_regex(testo)
    feat = None
    ultimo_errore = ""
    for tentativo in range(MAX_RETRY):
        try:
            feat = chain.invoke({"testo": testo})
            break
        except Exception as e:
            ultimo_errore = str(e)
            msg = ultimo_errore.lower()
            wait = 2 ** tentativo if ("rate" in msg or "429" in msg or "quota" in msg) else 1
            time.sleep(wait)
    if feat is None:
        return {**base, "errore": ultimo_errore[:200] or "estrazione_fallita",
                "parametri_regex": ";".join(map(str, reg))}

    llm_par = sorted(set(feat.parametri_costituzionali))
    return {
        **base,
        "tipo_atto_impugnato": ";".join(dict.fromkeys(a.value for a in feat.tipi_atto_impugnati)),
        "autorita_rimettente_tipo": feat.autorita_rimettente_tipo.value,
        "giurisdizione_a_quo": feat.giurisdizione_a_quo.value,
        "parametri_costituzionali": ";".join(map(str, llm_par)),
        "parametri_interposti": ";".join(dict.fromkeys(p.value for p in feat.parametri_interposti)) or "nessuno",
        "materia": feat.materia.value,
        # solo nel JSONL, non nel CSV: errore (per il resume) e controllo parametri
        "parametri_regex": ";".join(map(str, reg)),
        "parametri_mismatch": set(llm_par) != set(reg),
        "errore": "",
    }


# ── Selezione anni e scrittura CSV (riutilizzabile anche in caso di interruzione) ──
def anno_ok(anno_str, anno_min, anno_max) -> bool:
    try:
        a = int(anno_str)
    except (TypeError, ValueError):
        return False
    if anno_min is not None and a < anno_min:
        return False
    if anno_max is not None and a > anno_max:
        return False
    return True

def _arg_ints(flag, n):
    """Legge n interi dopo un flag da riga di comando, o None se il flag non c'e'."""
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    try:
        return [int(v) for v in sys.argv[i + 1:i + 1 + n]] if len(sys.argv) > i + n else None
    except ValueError:
        print(f"[!] {flag} richiede {n} anno/i interi"); sys.exit(1)

def leggi_selezione_anni():
    """Ritorna (anno_min, anno_max):
       --anno 2000        -> solo 2000            (min=max=2000)
       --range 2000 2010  -> intervallo 2000-2010 (ordine indifferente)
       --dal 2000         -> dal 2000 in poi      (alias: --anno-da)
       --al 2010          -> fino al 2010"""
    anno_min = anno_max = None
    if _arg_ints("--anno", 1):
        y = _arg_ints("--anno", 1)[0]; anno_min = anno_max = y
    if _arg_ints("--range", 2):
        a, b = _arg_ints("--range", 2); anno_min, anno_max = min(a, b), max(a, b)
    for flag in ("--dal", "--anno-da"):
        if _arg_ints(flag, 1):
            anno_min = _arg_ints(flag, 1)[0]
    if _arg_ints("--al", 1):
        anno_max = _arg_ints("--al", 1)[0]
    return anno_min, anno_max

def normalizza_record(r):
    """Porta un record dello scraper alle chiavi usate dall'estrattore (url, numero, anno, testo).
    Accetta sia il formato di scraper_v2.py (URL_Pronuncia, Numero_Pronuncia, Anno, testo)
    sia quello storico con cui e' stato prodotto feature_estratte.jsonl (url, numero, anno, testo)."""
    return {"url": r.get("url") or r.get("URL_Pronuncia", ""),
            "numero": r.get("numero") or r.get("Numero_Pronuncia", ""),
            "anno": r.get("anno") or r.get("Anno", ""),
            "testo": r.get("testo") or r.get("testo_pronuncia", "")}

def scrivi_csv(anno_min=None, anno_max=None):
    """Rigenera il CSV dal JSONL: solo righe riuscite e nell'intervallo scelto.
       Chiamata sia a fine run sia se l'utente interrompe (Ctrl+C)."""
    if not os.path.exists(OUTPUT_JSONL):
        return
    righe = []
    with open(OUTPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                righe.append(json.loads(line))
            except Exception:
                pass
    righe = [r for r in righe if not r.get("errore") and anno_ok(r.get("anno"), anno_min, anno_max)]

    def chiave(r):
        try:
            return (int(r.get("anno", 0)), int(r.get("numero", 0)))
        except (TypeError, ValueError):
            return (0, 0)
    righe.sort(key=chiave, reverse=True)

    if not righe:
        print("[i] Nessuna riga (riuscita e nell'intervallo) da scrivere nel CSV.")
        return
    campi = ["URL_Pronuncia", "numero", "anno", "tipo_atto_impugnato",
             "autorita_rimettente_tipo", "giurisdizione_a_quo", "parametri_costituzionali",
             "parametri_interposti", "materia"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campi, delimiter=";", extrasaction="ignore")
        w.writeheader()
        for r in righe:
            w.writerow(r)
    rivedere = sum(1 for r in righe if r.get("parametri_mismatch"))
    print(f"[OK] Salvato {OUTPUT_CSV} ({len(righe)} righe). Param da rivedere: {rivedere}.")


# ── Main con resume e concorrenza ─────────────────────────────────────────────
def main():
    max_righe = None
    if "--test" in sys.argv:
        try:
            max_righe = int(sys.argv[sys.argv.index("--test") + 1])
        except (IndexError, ValueError):
            print("[!] --test richiede un numero, es: --test 10"); sys.exit(1)
        print(f"[i] Modalita' TEST: prime {max_righe} pronunce")

    anno_min, anno_max = leggi_selezione_anni()
    if anno_min is None and anno_max is None and ANNO_DA:
        anno_min = ANNO_DA
    if anno_min is not None or anno_max is not None:
        if anno_min == anno_max:
            print(f"[i] Selezione: solo anno {anno_min}")
        elif anno_min is not None and anno_max is not None:
            print(f"[i] Selezione: anni da {anno_min} a {anno_max}")
        elif anno_min is not None:
            print(f"[i] Selezione: dall'anno {anno_min} in poi")
        else:
            print(f"[i] Selezione: fino all'anno {anno_max}")

    if not os.path.exists(INPUT_JSONL):
        print(f"[!] File '{INPUT_JSONL}' non trovato."); sys.exit(1)

    fatti = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("URL_Pronuncia") and not r.get("errore"):
                        fatti.add(r["URL_Pronuncia"])
                except Exception:
                    pass
        print(f"[i] Gia' estratte: {len(fatti)} (saltate)")

    da_fare = []
    with open(INPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = normalizza_record(json.loads(line))
            except Exception:
                continue
            if not (r.get("url") and (r.get("testo") or "").strip()):
                continue
            if r["url"] in fatti:
                continue
            if not anno_ok(r.get("anno"), anno_min, anno_max):
                continue
            da_fare.append(r)
    if max_righe:
        da_fare = da_fare[:max_righe]
    print(f"[i] Da elaborare ora: {len(da_fare)}")
    if not da_fare:
        print("[i] Niente da fare."); scrivi_csv(anno_min, anno_max); return

    chain = build_chain()

    ok = err = 0
    out = open(OUTPUT_JSONL, "a", encoding="utf-8")
    ex = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    try:
        futures = {ex.submit(estrai_una, chain, rec): rec.get("url") for rec in da_fare}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            out.write(json.dumps(res, ensure_ascii=False) + "\n")
            out.flush()   # ogni pronuncia e' salvata subito: se interrompi, non perdi nulla
            if res.get("errore"):
                err += 1
                print(f"  [{i}/{len(da_fare)}] n.{res.get('numero','?')}/{res.get('anno','?')} ERRORE")
            else:
                ok += 1
                flag = "  <-- PARAM DA RIVEDERE" if res.get("parametri_mismatch") else ""
                print(f"  [{i}/{len(da_fare)}] n.{res.get('numero','?')}/{res.get('anno','?')} "
                      f"{res['autorita_rimettente_tipo']:20s} {res['materia']:18s} "
                      f"par={res['parametri_costituzionali']}{flag}")
    except KeyboardInterrupt:
        print("\n[!] Interrotto dall'utente: salvo le pronunce gia' completate...")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
        out.close()
        print(f"[i] Completate finora: {ok} ok, {err} errori.")
        scrivi_csv(anno_min, anno_max)   # rigenera il CSV da cio' che c'e' nel JSONL


if __name__ == "__main__":
    main()