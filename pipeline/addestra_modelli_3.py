"""
addestra_modelli_3.py — versione 3: ablazione dei gruppi di feature + split casuale vs temporale
===============================================================================================
Parte dalla versione 2 (stesse feature, esito a 3 classi, stesso modello) e fissa gli
iperparametri scelti dalla sua validazione a finestra mobile, cosi' le differenze misurate
dipendono solo da cio' che cambia in ciascun esperimento:

  1) ABLAZIONE: si toglie UN gruppo di feature alla volta (rimettente, giurisdizione, materia,
     atto, interposti, parametri) e si misura il peggioramento. Piu' il risultato peggiora,
     piu' il modello dipendeva da quel gruppo. Ogni configurazione e' valutata sia con la
     validazione a finestra mobile (media sulle finestre) sia sul test (anni >= ANNO_TEST_DA).

  2) SPLIT CASUALE vs TEMPORALE: lo stesso modello e' valutato su un test temporale (anni
     >= ANNO_TEST_DA) e su test casuali della stessa dimensione, estratti da tutti gli anni
     (media su N_RIPETIZIONI estrazioni). La differenza misura quanto pesa il cambiamento nel
     tempo: con lo split casuale il modello "vede" in addestramento pronunce coeve a quelle
     del test.

Dipendenze: pip install pandas numpy scikit-learn
Uso:        python addestra_modelli_3.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from addestra_modelli import ANNO_TEST_DA, DATASET, build_features
from addestra_modelli_2 import (ANNI_VALIDAZIONE, baseline_classe, baseline_mediana,
                                nuovo_clf, nuovo_reg, pesi)

# Iperparametri scelti dalla validazione a finestra mobile della versione 2
PARAMS_ESITO = {"max_depth": 3, "tau": 5, "class_w": "sqrt"}
PARAMS_DURATA = {"max_depth": 3, "tau": None}
N_RIPETIZIONI = 10                      # estrazioni casuali per lo split casuale

# Gruppi di feature: prefisso delle colonne (o nomi esatti) che li compongono
GRUPPI = {
    "rimettente":    ["autorita_rimettente_tipo_"],
    "giurisdizione": ["giurisdizione_a_quo_"],
    "materia":       ["materia_"],
    "atto":          ["atto_"],
    "interposti":    ["int_", "ha_interposti"],
    "parametri":     ["par_", "n_parametri"],
}

def colonne_gruppo(X, gruppo):
    return [c for c in X.columns if any(c.startswith(p) for p in GRUPPI[gruppo])]


# ── Valutazione di una configurazione di feature ─────────────────────────────
def valuta_esito(X, y, anno, valido, tr, te, params=PARAMS_ESITO):
    clf = nuovo_clf(params)
    clf.fit(X[tr], y[tr], sample_weight=pesi(anno[tr], y[tr], params["tau"], params["class_w"]))
    return f1_score(y[te], clf.predict(X[te]), average="macro", zero_division=0)

def valuta_durata(X, g, anno, tr, te, params=PARAMS_DURATA):
    reg = nuovo_reg(params)
    reg.fit(X[tr], np.log1p(g[tr]), sample_weight=pesi(anno[tr], tau=params["tau"]))
    return mean_absolute_error(g[te], np.expm1(reg.predict(X[te])))

def finestre_e_test(fn, anno, valido):
    """Media sulle finestre mobili (train < V, valida == V) e punteggio sul test temporale."""
    cv = [fn((anno < V) & valido, (anno == V) & valido) for V in ANNI_VALIDAZIONE]
    test = fn((anno < ANNO_TEST_DA) & valido, (anno >= ANNO_TEST_DA) & valido)
    return np.mean(cv), np.std(cv), test


# ── 1) Ablazione dei gruppi di feature ────────────────────────────────────────
def ablazione(X, y, g, anno, ok_esito, ok_durata):
    print("\n" + "="*78 + "\n  1) ABLAZIONE: si toglie un gruppo di feature alla volta\n" + "="*78)
    righe = []
    for nome in [None] + list(GRUPPI):
        cols = [] if nome is None else colonne_gruppo(X, nome)
        Xa = X.drop(columns=cols)
        e = finestre_e_test(lambda tr, te: valuta_esito(Xa, y, anno, ok_esito, tr, te), anno, ok_esito)
        d = finestre_e_test(lambda tr, te: valuta_durata(Xa, g, anno, tr, te), anno, ok_durata)
        righe.append((nome or "nessuno (tutte le feature)", len(cols), *e, *d))

    rif = righe[0]
    print(f"\n{'gruppo tolto':28s} {'col':>4s} | {'ESITO F1 macro':^30s} | {'DURATA MAE giorni':^30s}")
    print(f"{'':28s} {'':>4s} | {'finestre':>13s} {'test':>7s} {'diff':>7s} | {'finestre':>13s} {'test':>7s} {'diff':>7s}")
    for r in righe:
        print(f"{r[0]:28s} {r[1]:4d} | {r[2]:.3f}±{r[3]:.3f}  {r[4]:7.3f} {r[4]-rif[4]:+7.3f} | "
              f"{r[5]:6.1f}±{r[6]:5.1f}  {r[7]:7.1f} {r[7]-rif[7]:+7.1f}")
    print("\n  diff = differenza rispetto al modello con tutte le feature (esito: negativo = peggiora;"
          "\n      durata: positivo = peggiora). Confrontare anche le colonne 'finestre', piu' stabili.")


# ── 2) Split casuale contro split temporale ───────────────────────────────────
def split_casuale_vs_temporale(X, y, g, anno, ok_esito, ok_durata):
    print("\n" + "="*78 + "\n  2) SPLIT CASUALE vs SPLIT TEMPORALE\n" + "="*78)
    idx = X.index
    # Nessun peso di recency in questo esperimento: i pesi favoriscono gli anni vicini al test
    # temporale, ma con lo split casuale il test e' sparso su tutti gli anni e verrebbe
    # penalizzato. Senza pesi i due split sono confrontati alla pari.
    p_esito = {**PARAMS_ESITO, "tau": None}
    p_durata = {**PARAMS_DURATA, "tau": None}

    # esito
    tr_t, te_t = (anno < ANNO_TEST_DA) & ok_esito, (anno >= ANNO_TEST_DA) & ok_esito
    f1_t = valuta_esito(X, y, anno, ok_esito, tr_t, te_t, p_esito)
    c = baseline_classe(anno, y, tr_t)
    base_t = f1_score(y[te_t], np.full(te_t.sum(), c), average="macro", zero_division=0)
    f1_r, base_r = [], []
    for s in range(N_RIPETIZIONI):
        i_tr, i_te = train_test_split(idx[ok_esito], test_size=int(te_t.sum()), random_state=s,
                                      stratify=y[ok_esito])
        tr, te = idx.isin(i_tr), idx.isin(i_te)
        f1_r.append(valuta_esito(X, y, anno, ok_esito, tr, te, p_esito))
        cr = y[tr].value_counts().idxmax()          # baseline: classe piu' frequente del train
        base_r.append(f1_score(y[te], np.full(te.sum(), cr), average="macro", zero_division=0))

    # durata
    tr_t, te_t = (anno < ANNO_TEST_DA) & ok_durata, (anno >= ANNO_TEST_DA) & ok_durata
    mae_t = valuta_durata(X, g, anno, tr_t, te_t, p_durata)
    mae_base_t = mean_absolute_error(g[te_t], np.full(te_t.sum(), baseline_mediana(anno, g, tr_t)))
    mae_r, mae_base_r = [], []
    for s in range(N_RIPETIZIONI):
        i_tr, i_te = train_test_split(idx[ok_durata], test_size=int(te_t.sum()), random_state=s)
        tr, te = idx.isin(i_tr), idx.isin(i_te)
        mae_r.append(valuta_durata(X, g, anno, tr, te, p_durata))
        mae_base_r.append(mean_absolute_error(g[te], np.full(te.sum(), np.median(g[tr]))))

    print(f"\n(split casuale: {N_RIPETIZIONI} estrazioni, test della stessa dimensione del test temporale,"
          f"\n stratificato per classe per l'esito; valori = media ± deviazione standard)")
    print(f"\n{'':24s} {'split temporale':>18s} {'split casuale':>20s} {'differenza':>12s}")
    print(f"{'ESITO F1 macro':24s} {f1_t:18.3f} {np.mean(f1_r):13.3f}±{np.std(f1_r):.3f} {np.mean(f1_r)-f1_t:+12.3f}")
    print(f"{'  baseline':24s} {base_t:18.3f} {np.mean(base_r):13.3f}±{np.std(base_r):.3f}")
    print(f"{'DURATA MAE giorni':24s} {mae_t:18.1f} {np.mean(mae_r):13.1f}±{np.std(mae_r):5.1f} {np.mean(mae_r)-mae_t:+12.1f}")
    print(f"{'  baseline (mediana)':24s} {mae_base_t:18.1f} {np.mean(mae_base_r):13.1f}±{np.std(mae_base_r):5.1f}")
    print(f"{'  modello - baseline':24s} {mae_t-mae_base_t:+18.1f} {np.mean(mae_r)-np.mean(mae_base_r):+20.1f}")


def main():
    df = pd.read_csv(DATASET, dtype=str, keep_default_na=False)
    X = build_features(df)
    anno = pd.to_numeric(df["anno"], errors="coerce")
    y = df["esito"]
    g = pd.to_numeric(df["giorni_gu_decisione"], errors="coerce")
    ok_esito = y != "altro"                      # esito a 3 classi, come nella versione 2
    ok_durata = g.between(1, 2000)
    print(f"[i] dataset: {len(df)} righe  |  feature: {X.shape[1]} colonne")
    print(f"[i] iperparametri (dalla versione 2): esito {PARAMS_ESITO}, durata {PARAMS_DURATA}")

    ablazione(X, y, g, anno, ok_esito, ok_durata)
    split_casuale_vs_temporale(X, y, g, anno, ok_esito, ok_durata)


if __name__ == "__main__":
    main()
