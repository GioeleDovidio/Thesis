"""
addestra_modelli_2.py — versione 2: gestione del drift temporale + validazione a finestra mobile
================================================================================================
Stessi due task di addestra_modelli.py, con le STESSE feature (importate da li', nessuna
feature temporale), e queste modifiche:

  1) ESITO A 3 CLASSI
     - la classe 'altro' (restituzione atti, estinzione, ...) viene esclusa dal task esito:
       e' eterogenea, quasi scomparsa negli anni recenti e non e' un esito sul merito o sul rito

  2) DRIFT TEMPORALE
     - pesi di "recency": gli esempi recenti contano di piu', w = exp((anno - ultimo_anno_train) / tau)
     - class weight attenuato (radice del 'balanced') come alternativa al 'balanced' pieno
     - baseline piu' onesto: classe piu' frequente / mediana degli ULTIMI N anni di train
     L'anno NON e' una feature: serve solo per i pesi, gli split e i baseline.

  3) VALIDAZIONE A FINESTRA MOBILE (rolling origin)
     - per ogni anno di validazione V in ANNI_VALIDAZIONE: train su anni < V, valida su V
     - la media sulle finestre sceglie gli iperparametri (griglia piccola)
     - il test finale (anni >= ANNO_TEST_DA) e' usato UNA sola volta, col modello scelto

Dipendenze: pip install pandas numpy scikit-learn
Uso:        python addestra_modelli_2.py
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.utils.class_weight import compute_sample_weight

from addestra_modelli import ANNO_TEST_DA, DATASET, build_features, importanza

ESITO_3_CLASSI = True                   # True: esclude 'altro' dal task esito
ANNI_VALIDAZIONE = [2018, 2019, 2020, 2021, 2022]   # finestre mobili: train < V, valida su V
ANNI_BASELINE = 3                       # baseline calcolato sugli ultimi N anni di train

# Griglie (piccole di proposito: 5 finestre x combinazioni = numero di fit)
GRIGLIA_ESITO = {
    "max_depth":   [3, 6],
    "tau":         [None, 5, 10],       # None = nessun peso di recency
    "class_w":     ["none", "sqrt"],    # "sqrt" = radice del balanced
}
GRIGLIA_DURATA = {
    "max_depth":   [3, 6],
    "tau":         [None, 3, 5],
}


# ── Pesi ──────────────────────────────────────────────────────────────────────
def pesi(anni, y=None, tau=None, class_w="none"):
    """Peso di recency (rispetto all'ultimo anno di train) x class weight attenuato."""
    w = np.ones(len(anni))
    if tau is not None:
        w *= np.exp((anni.values - anni.max()) / tau)
    if y is not None and class_w == "sqrt":
        w *= np.sqrt(compute_sample_weight("balanced", y))
    return w


# ── Validazione a finestra mobile ─────────────────────────────────────────────
def combinazioni(griglia):
    chiavi = list(griglia)
    for valori in itertools.product(*griglia.values()):
        yield dict(zip(chiavi, valori))

def rolling_cv(anno, valido, griglia, fit_score, nome_metrica, maggiore_meglio):
    """Per ogni combinazione: media della metrica sulle finestre (train < V, valida == V)."""
    print(f"\nValidazione a finestra mobile su {ANNI_VALIDAZIONE} ({nome_metrica}):")
    risultati = []
    for params in combinazioni(griglia):
        punteggi = []
        for V in ANNI_VALIDAZIONE:
            tr = (anno < V) & valido
            va = (anno == V) & valido
            punteggi.append(fit_score(params, tr, va))
        m = float(np.mean(punteggi))
        risultati.append((m, params))
        print(f"   {nome_metrica}={m:7.3f}  (+/-{np.std(punteggi):.3f})  {params}")
    migliore = max(risultati, key=lambda r: r[0] if maggiore_meglio else -r[0])
    print(f"-> scelto: {migliore[1]}  ({nome_metrica} medio {migliore[0]:.3f})")
    return migliore[1]


# ── Task 1: esito (classificazione) ───────────────────────────────────────────
def nuovo_clf(params):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          max_depth=params["max_depth"], min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)

def baseline_classe(anno, y, tr):
    """Classe piu' frequente negli ultimi ANNI_BASELINE anni di train."""
    recenti = tr & (anno > anno[tr].max() - ANNI_BASELINE)
    return y[recenti].value_counts().idxmax()

def task_esito(df, X, anno):
    y = df["esito"]
    valido = (y != "altro") if ESITO_3_CLASSI else pd.Series(True, index=df.index)
    n_classi = y[valido].nunique()
    print("\n" + "="*66 + f"\n  TASK 1 — ESITO (classificazione, {n_classi} classi)\n" + "="*66)
    if ESITO_3_CLASSI:
        print(f"[i] esclusa la classe 'altro': {(~valido).sum()} pronunce")

    def fit_score(params, tr, va):
        clf = nuovo_clf(params)
        clf.fit(X[tr], y[tr], sample_weight=pesi(anno[tr], y[tr], params["tau"], params["class_w"]))
        return f1_score(y[va], clf.predict(X[va]), average="macro", zero_division=0)

    params = rolling_cv(anno, valido, GRIGLIA_ESITO, fit_score, "F1_macro", maggiore_meglio=True)

    # baseline sulle stesse finestre, per confronto
    f1_base = []
    for V in ANNI_VALIDAZIONE:
        tr, va = (anno < V) & valido, (anno == V) & valido
        c = baseline_classe(anno, y, tr)
        f1_base.append(f1_score(y[va], np.full(va.sum(), c), average="macro", zero_division=0))
    print(f"   (baseline ultimi {ANNI_BASELINE} anni sulle finestre: F1_macro={np.mean(f1_base):.3f})")

    # test finale
    tr = (anno < ANNO_TEST_DA) & valido
    te = (anno >= ANNO_TEST_DA) & valido
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    print(f"\nTEST FINALE  train: {len(Xtr)} (anni <{ANNO_TEST_DA})  |  test: {len(Xte)} (anni >={ANNO_TEST_DA})")

    c = baseline_classe(anno, y, tr)
    yb = np.full(len(yte), c)
    print(f"\nBASELINE (classe piu' frequente ultimi {ANNI_BASELINE} anni = '{c}'): "
          f"accuracy={(yb==yte).mean():.3f}  F1_macro={f1_score(yte, yb, average='macro', zero_division=0):.3f}")

    clf = nuovo_clf(params)
    clf.fit(Xtr, ytr, sample_weight=pesi(anno[tr], ytr, params["tau"], params["class_w"]))
    yp = clf.predict(Xte)
    print(f"MODELLO  (gradient boosting):     accuracy={(yp==yte).mean():.3f}  "
          f"F1_macro={f1_score(yte, yp, average='macro', zero_division=0):.3f}")
    print("\nReport per classe (test):")
    print(classification_report(yte, yp, zero_division=0))
    print("Matrice di confusione (righe=vero, colonne=predetto):")
    etich = sorted(y[valido].unique())
    cm = confusion_matrix(yte, yp, labels=etich)
    print("        " + "  ".join(f"{e[:7]:>8s}" for e in etich))
    for e, riga in zip(etich, cm):
        print(f"{e[:7]:>7s} " + "  ".join(f"{v:8d}" for v in riga))
    return clf, Xte, yte


# ── Task 2: durata (regressione) ──────────────────────────────────────────────
def nuovo_reg(params):
    # loss absolute_error sul log -> stima la mediana, coerente con la MAE e col baseline
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, loss="absolute_error",
                                         max_depth=params["max_depth"], min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0)

def baseline_mediana(anno, g, tr):
    """Mediana dei giorni negli ultimi ANNI_BASELINE anni di train."""
    recenti = tr & (anno > anno[tr].max() - ANNI_BASELINE)
    return float(np.median(g[recenti]))

def task_durata(df, X, anno):
    print("\n" + "="*66 + "\n  TASK 2 — DURATA giorni GU->decisione (regressione)\n" + "="*66)
    g = pd.to_numeric(df["giorni_gu_decisione"], errors="coerce")
    valido = g.between(1, 2000)                      # esclude mancanti, 0, negativi
    with np.errstate(invalid="ignore"):
        y = np.log1p(g)                              # log per l'asimmetria (i NaN restano NaN, poi mascherati)

    def fit_score(params, tr, va):
        reg = nuovo_reg(params)
        reg.fit(X[tr], y[tr], sample_weight=pesi(anno[tr], tau=params["tau"]))
        return mean_absolute_error(g[va], np.expm1(reg.predict(X[va])))

    params = rolling_cv(anno, valido, GRIGLIA_DURATA, fit_score, "MAE_giorni", maggiore_meglio=False)

    mae_base = []
    for V in ANNI_VALIDAZIONE:
        tr, va = (anno < V) & valido, (anno == V) & valido
        mae_base.append(mean_absolute_error(g[va], np.full(va.sum(), baseline_mediana(anno, g, tr))))
    print(f"   (baseline mediana ultimi {ANNI_BASELINE} anni sulle finestre: MAE={np.mean(mae_base):.1f})")

    # test finale
    tr = (anno < ANNO_TEST_DA) & valido
    te = (anno >= ANNO_TEST_DA) & valido
    Xtr, Xte, ytr, gte = X[tr], X[te], y[tr], g[te]
    print(f"\nTEST FINALE  train: {len(Xtr)}  |  test: {len(Xte)}  (righe con durata valida)")

    med = baseline_mediana(anno, g, tr)
    gb = np.full(len(gte), med)
    print(f"\nBASELINE (mediana ultimi {ANNI_BASELINE} anni = {med:.0f} gg): MAE={mean_absolute_error(gte, gb):6.1f} giorni")

    reg = nuovo_reg(params)
    reg.fit(Xtr, ytr, sample_weight=pesi(anno[tr], tau=params["tau"]))
    gp = np.expm1(reg.predict(Xte))
    mae = mean_absolute_error(gte, gp)
    rmse = mean_squared_error(gte, gp) ** 0.5
    print(f"MODELLO (grad. boosting): MAE={mae:6.1f} giorni  RMSE={rmse:6.1f}  R2={r2_score(gte, gp):.3f}")
    print(f"  (il modello sbaglia in media di ~{mae:.0f} giorni; il baseline di ~{mean_absolute_error(gte, gb):.0f})")
    return reg, Xte, gte


def main():
    df = pd.read_csv(DATASET, dtype=str, keep_default_na=False)
    print(f"[i] dataset: {len(df)} righe")
    X = build_features(df)
    anno = pd.to_numeric(df["anno"], errors="coerce")   # solo per pesi, split e baseline
    print(f"[i] feature costruite: {X.shape[1]} colonne")

    clf, Xte_c, yte_c = task_esito(df, X, anno)
    importanza("esito", clf, Xte_c, yte_c, scoring="f1_macro")

    reg, Xte_r, gte_r = task_durata(df, X, anno)
    # per la regressione l'importanza usa il target log, coerente col training
    yte_log = np.log1p(gte_r)
    importanza("durata", reg, Xte_r, yte_log, scoring="neg_mean_absolute_error")


if __name__ == "__main__":
    main()
