# -*- coding: utf-8 -*-
"""
Problema 2 - Regresión en Boston Housing (506 viviendas, variable respuesta MEDV).

Datos: data/Boston.csv  (copia del dataset 'Boston' del paquete ISLP, el mismo del curso;
trae 12 atributos + medv, no incluye la variable B del dataset original).

Uso:  python p2_boston.py
Genera figuras en ./figuras y un resumen en ./figuras/p2_resultados.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

HERE = Path(__file__).parent
FIG = HERE / "figuras"
FIG.mkdir(exist_ok=True)
SEED = 42
out = []

# --------------------------------------------------------------------------
# 1. Carga y análisis exploratorio
# --------------------------------------------------------------------------
df = pd.read_csv(HERE / "data" / "Boston.csv")
X = df.drop(columns="medv")
y = df["medv"]
out.append(f"Dimensiones: {df.shape};  valores faltantes: {int(df.isna().sum().sum())}")
out.append("\nResumen de medv:\n" + y.describe().round(2).to_string())
corr = df.corr()
out.append("\nCorrelación con medv:\n" + corr["medv"].drop("medv").sort_values().round(3).to_string())
out.append(f"\nViviendas con medv = 50 (valor tope): {(y == 50).sum()}")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.1), gridspec_kw={"width_ratios": [1, 1, 1.15]})
sns.histplot(y, bins=25, kde=True, ax=ax[0], color="#6baed6")
ax[0].set_title("Histograma de MEDV (miles de USD)")
ax[1].scatter(df["lstat"], y, s=8, alpha=0.6, color="#3182bd")
ax[1].set_xlabel("lstat (% población de bajo nivel)"); ax[1].set_ylabel("medv")
ax[1].set_title(f"MEDV vs lstat (r = {corr.loc['lstat','medv']:.2f})")
ax[2].scatter(df["rm"], y, s=8, alpha=0.6, color="#e6550d")
ax[2].set_xlabel("rm (habitaciones por vivienda)")
ax[2].set_title(f"MEDV vs rm (r = {corr.loc['rm','medv']:.2f})")
plt.tight_layout()
plt.savefig(FIG / "p2_eda.png", dpi=200)
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".1f", cmap="coolwarm", annot_kws={"size": 6}, cbar=False)
plt.title("Matriz de correlación")
plt.tight_layout()
plt.savefig(FIG / "p2_correlacion.png", dpi=200)
plt.close()

# --------------------------------------------------------------------------
# 2. Entrenamiento / validación (80 % / 20 %)
# --------------------------------------------------------------------------
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED)
out.append(f"\nEntrenamiento: {len(Xtr)} filas;  validación: {len(Xva)} filas")


def mse(modelo, A, b):
    return mean_squared_error(b, modelo.predict(A))


def pipe_ols(grado):
    return make_pipeline(PolynomialFeatures(grado, include_bias=False), StandardScaler(),
                         LinearRegression())


def pipe_ridge(grado, lam):
    return make_pipeline(PolynomialFeatures(grado, include_bias=False), StandardScaler(),
                         Ridge(alpha=lam))


# --------------------------------------------------------------------------
# 3 y 4. Modelo lineal (grado 1) y polinomiales (grados 2 y 3), sin regularizar
# --------------------------------------------------------------------------
grados = [1, 2, 3]
ols, mse_tr, mse_va, n_feat = {}, {}, {}, {}
for g in grados:
    m = pipe_ols(g).fit(Xtr, ytr)
    ols[g] = m
    mse_tr[g], mse_va[g] = mse(m, Xtr, ytr), mse(m, Xva, yva)
    n_feat[g] = m[0].n_output_features_
out.append("\nMínimos cuadrados (sin regularizar):")
out.append(f"{'grado':>6} {'#atributos':>11} {'MSE train':>11} {'MSE valid':>11}")
for g in grados:
    out.append(f"{g:>6} {n_feat[g]:>11} {mse_tr[g]:>11.2f} {mse_va[g]:>11.2f}")

# --------------------------------------------------------------------------
# 5. Ridge con las características polinomiales: lambda por validación cruzada (5 folds
#    sobre el conjunto de entrenamiento; la validación NO se usa para elegir lambda)
# --------------------------------------------------------------------------
lambdas = np.logspace(-3, 4, 36)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
ridge, mejor_lam, cv_curvas, cv_min = {}, {}, {}, {}
va_curvas = {}
for g in grados:
    gs = GridSearchCV(
        make_pipeline(PolynomialFeatures(g, include_bias=False), StandardScaler(), Ridge()),
        {"ridge__alpha": lambdas}, scoring="neg_mean_squared_error", cv=kf)
    gs.fit(Xtr, ytr)
    cv_curvas[g] = -gs.cv_results_["mean_test_score"]
    mejor_lam[g] = gs.best_params_["ridge__alpha"]
    cv_min[g] = cv_curvas[g].min()
    ridge[g] = pipe_ridge(g, mejor_lam[g]).fit(Xtr, ytr)
    va_curvas[g] = [mse(pipe_ridge(g, l).fit(Xtr, ytr), Xva, yva) for l in lambdas]

out.append("\nRidge (lambda elegido por CV 5-fold en entrenamiento):")
out.append(f"{'grado':>6} {'lambda*':>10} {'MSE CV':>9} {'MSE train':>11} {'MSE valid':>11}")
for g in grados:
    out.append(f"{g:>6} {mejor_lam[g]:>10.3f} {cv_min[g]:>9.2f} "
               f"{mse(ridge[g], Xtr, ytr):>11.2f} {mse(ridge[g], Xva, yva):>11.2f}")

# --------------------------------------------------------------------------
# Gráficas
# --------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
# (a) MSE vs grado del polinomio
ax[0].plot(grados, [mse_tr[g] for g in grados], "o-", label="train (MCO)")
ax[0].plot(grados, [mse_va[g] for g in grados], "s-", label="validación (MCO)")
ax[0].plot(grados, [mse(ridge[g], Xva, yva) for g in grados], "^--", color="#31a354",
           label="validación (Ridge)")
ax[0].set_yscale("log")
ax[0].set_xticks(grados)
ax[0].set_xlabel("grado del polinomio"); ax[0].set_ylabel("MSE (escala log)")
ax[0].set_title("MSE según el grado"); ax[0].legend(fontsize=7)
# (b) MSE vs lambda
for g, c in zip(grados, ["#3182bd", "#e6550d", "#31a354"]):
    ax[1].plot(lambdas, cv_curvas[g], color=c, label=f"CV, grado {g}")
    ax[1].plot(mejor_lam[g], cv_min[g], "o", color=c)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel(r"$\lambda$"); ax[1].set_ylabel("MSE de validación cruzada")
ax[1].set_title(r"Ridge: MSE vs $\lambda$"); ax[1].legend(fontsize=7)
# (c) predicho vs real, validación
g_ols = min(grados, key=lambda g: mse_va[g])
g_rid = min(grados, key=lambda g: cv_min[g])
lim = [0, 52]
ax[2].plot(lim, lim, "k--", lw=1)
ax[2].scatter(yva, ols[g_ols].predict(Xva), s=14, alpha=0.7,
              label=f"MCO grado {g_ols} (MSE {mse_va[g_ols]:.1f})")
ax[2].scatter(yva, ridge[g_rid].predict(Xva), s=14, alpha=0.7, color="#31a354",
              label=f"Ridge grado {g_rid} (MSE {mse(ridge[g_rid], Xva, yva):.1f})")
ax[2].set_xlabel("MEDV real"); ax[2].set_ylabel("MEDV predicho")
ax[2].set_title("Validación: predicho vs real"); ax[2].legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG / "p2_resultados.png", dpi=200)
plt.close()
out.append(f"\nMejor MCO (por validación): grado {g_ols}.  Mejor Ridge (por CV): grado {g_rid}, "
           f"lambda = {mejor_lam[g_rid]:.3f}")

# --------------------------------------------------------------------------
# Chequeo de robustez: repetir con 30 particiones train/validación distintas
# --------------------------------------------------------------------------
R = 30
filas = []
for r in range(R):
    A, B, a, b = train_test_split(X, y, test_size=0.2, random_state=1000 + r)
    fila = {}
    for g in grados:
        fila[f"MCO g{g}"] = mse(pipe_ols(g).fit(A, a), B, b)
        fila[f"Ridge g{g}"] = mse(pipe_ridge(g, mejor_lam[g]).fit(A, a), B, b)
    filas.append(fila)
rob = pd.DataFrame(filas)
out.append(f"\nRobustez ({R} particiones aleatorias, lambda fijo = el elegido por CV): MSE de validación")
out.append(pd.DataFrame({"mediana": rob.median(), "media": rob.mean(), "desvío": rob.std()}).round(2).to_string())

txt = "\n".join(out)
print(txt)
(FIG / "p2_resultados.txt").write_text(txt, encoding="utf-8")
