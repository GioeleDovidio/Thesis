"""
addestra_modelli.py — modelli predittivi su esito e durata (giorni GU->decisione)
=================================================================================
Dal dataset unito (dataset_ml.csv) costruisce le feature, fa uno SPLIT TEMPORALE
(addestra sugli anni passati, testa sugli ultimi), confronta ogni modello con un
baseline banale e riporta le metriche. Due task:
  - esito             -> classificazione (accoglimento/infondatezza/inammissibilita/altro)
  - giorni_gu_decisione -> regressione (durata in giorni)

Metodo: lo split e' temporale, non casuale, perche' il caso d'uso e' prevedere questioni
future e la relazione feature-esito cambia nel tempo. La regressione lavora sul log dei
giorni (distribuzione asimmetrica) e l'errore e' riportato in giorni reali.

Dipendenze: pip install pandas numpy scikit-learn
Uso:        python addestra_modelli.py
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.utils.class_weight import compute_sample_weight

DATASET = "dataset_ml.csv"
ANNO_TEST_DA = 2023          # train: anni < 2023 ; test: anni >= 2023
MIN_FREQ_PARAM = 10          # un articolo Cost. diventa colonna se compare almeno N volte


# ── Costruzione delle feature ─────────────────────────────────────────────────
def multi_hot(serie, prefix, vocab):
    """Una colonna binaria per ogni valore di vocab; la cella e' lista separata da ';'."""
    out = {}
    valori = serie.fillna("").str.lower()
    for v in vocab:
        out[f"{prefix}_{v}"] = valori.apply(lambda s: int(v in [x.strip() for x in s.split(";")]))
    return pd.DataFrame(out, index=serie.index)

def build_features(df):
    F = pd.DataFrame(index=df.index)
    train = pd.to_numeric(df["anno"], errors="coerce") < ANNO_TEST_DA

    # categorici a valore singolo -> one-hot
    for col in ["autorita_rimettente_tipo", "giurisdizione_a_quo", "materia"]:
        d = pd.get_dummies(df[col].fillna("na"), prefix=col).astype(int)
        F = pd.concat([F, d], axis=1)

    # tipo_atto (multi-valore) -> multi-hot
    vocab_atto = ["legge_statale","legge_regionale","decreto_legislativo","decreto_legge",
                  "dpr","regio_decreto","codice","altro"]
    F = pd.concat([F, multi_hot(df["tipo_atto_impugnato"], "atto", vocab_atto)], axis=1)

    # interposti (multi-valore) -> multi-hot + flag
    vocab_int = ["cedu","diritto_ue","legge_costituzionale","altro_trattato"]
    inter = multi_hot(df["parametri_interposti"], "int", vocab_int)
    F = pd.concat([F, inter], axis=1)
    F["ha_interposti"] = (inter.sum(axis=1) > 0).astype(int)

    # parametri costituzionali -> multi-hot (solo articoli frequenti) + conteggio
    # la soglia di frequenza si calcola SOLO sugli anni di train: scegliere le colonne
    # guardando anche il test userebbe informazione del test nella costruzione del modello
    liste = df["parametri_costituzionali"].fillna("").apply(
        lambda s: [x.strip() for x in s.split(";") if x.strip().isdigit()])
    freq = pd.Series([a for L in liste[train] for a in L]).value_counts()
    art_freq = [a for a, c in freq.items() if c >= MIN_FREQ_PARAM]
    for a in art_freq:
        F[f"par_{a}"] = liste.apply(lambda L: int(a in L))
    F["n_parametri"] = liste.apply(len)

    return F


def split_temporale(df, X, y, mask_valid=None):
    anno = pd.to_numeric(df["anno"], errors="coerce")
    tr = anno < ANNO_TEST_DA
    te = anno >= ANNO_TEST_DA
    if mask_valid is not None:
        tr &= mask_valid; te &= mask_valid
    return X[tr], X[te], y[tr], y[te]


# ── Task 1: esito (classificazione) ───────────────────────────────────────────
def task_esito(df, X):
    print("\n" + "="*66 + "\n  TASK 1 — ESITO (classificazione, 4 classi)\n" + "="*66)
    y = df["esito"].fillna("altro")
    Xtr, Xte, ytr, yte = split_temporale(df, X, y)
    print(f"train: {len(Xtr)} (anni <{ANNO_TEST_DA})  |  test: {len(Xte)} (anni >={ANNO_TEST_DA})")

    base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    yb = base.predict(Xte)
    print(f"\nBASELINE (classe piu' frequente): accuracy={ (yb==yte).mean():.3f}  "
          f"F1_macro={f1_score(yte, yb, average='macro', zero_division=0):.3f}")

    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         max_depth=6, random_state=0)
    clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    yp = clf.predict(Xte)
    print(f"MODELLO  (gradient boosting):     accuracy={ (yp==yte).mean():.3f}  "
          f"F1_macro={f1_score(yte, yp, average='macro', zero_division=0):.3f}")
    print("\nReport per classe (test):")
    print(classification_report(yte, yp, zero_division=0))
    print("Matrice di confusione (righe=vero, colonne=predetto):")
    etich = sorted(y.unique())
    cm = confusion_matrix(yte, yp, labels=etich)
    print("        " + "  ".join(f"{e[:7]:>8s}" for e in etich))
    for e, riga in zip(etich, cm):
        print(f"{e[:7]:>7s} " + "  ".join(f"{v:8d}" for v in riga))
    return clf, Xte, yte


# ── Task 2: durata (regressione) ──────────────────────────────────────────────
def task_durata(df, X):
    print("\n" + "="*66 + "\n  TASK 2 — DURATA giorni GU->decisione (regressione)\n" + "="*66)
    g = pd.to_numeric(df["giorni_gu_decisione"], errors="coerce")
    valido = g.between(1, 2000)                      # esclude mancanti, 0, negativi, assurdi
    with np.errstate(invalid="ignore"):
        y = np.log1p(g)                              # log per l'asimmetria (i NaN restano NaN, poi mascherati)
    Xtr, Xte, ytr, yte = split_temporale(df, X, y, mask_valid=valido)
    gte = np.expm1(yte)                              # giorni reali nel test
    print(f"train: {len(Xtr)}  |  test: {len(Xte)}  (righe con durata valida)")

    base = DummyRegressor(strategy="median").fit(Xtr, ytr)
    gb = np.expm1(base.predict(Xte))
    print(f"\nBASELINE (mediana):     MAE={mean_absolute_error(gte, gb):6.1f} giorni")

    reg = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                        max_depth=6, random_state=0)
    reg.fit(Xtr, ytr)
    gp = np.expm1(reg.predict(Xte))
    mae = mean_absolute_error(gte, gp)
    rmse = mean_squared_error(gte, gp) ** 0.5
    print(f"MODELLO (grad. boosting): MAE={mae:6.1f} giorni  RMSE={rmse:6.1f}  R2={r2_score(gte, gp):.3f}")
    print(f"  (il modello sbaglia in media di ~{mae:.0f} giorni; il baseline di ~{mean_absolute_error(gte, gb):.0f})")
    return reg, Xte, gte


# ── Importanza delle feature ──────────────────────────────────────────────────
def importanza(nome, model, Xte, yte, scoring):
    print(f"\n--- Feature piu' importanti ({nome}, permutation importance) ---")
    r = permutation_importance(model, Xte, yte, n_repeats=5, random_state=0, scoring=scoring)
    top = pd.Series(r.importances_mean, index=Xte.columns).sort_values(ascending=False).head(15)
    for k, v in top.items():
        print(f"   {v:+.4f}  {k}")


def main():
    df = pd.read_csv(DATASET, dtype=str, keep_default_na=False)
    print(f"[i] dataset: {len(df)} righe")
    X = build_features(df)
    print(f"[i] feature costruite: {X.shape[1]} colonne")

    clf, Xte_c, yte_c = task_esito(df, X)
    importanza("esito", clf, Xte_c, yte_c, scoring="f1_macro")

    reg, Xte_r, gte_r = task_durata(df, X)
    # per la regressione l'importanza usa il target log, coerente col training
    yte_log = np.log1p(gte_r)
    importanza("durata", reg, Xte_r, yte_log, scoring="neg_mean_absolute_error")


if __name__ == "__main__":
    main()