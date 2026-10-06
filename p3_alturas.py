# -*- coding: utf-8 -*-
"""
Problema 3 - Alturas: modelo lineal para la altura de los hijos (estudiantes del curso)
a partir de la altura del padre, la altura de la madre y el sexo.

Uso:  python p3_alturas.py
Genera figuras en ./figuras y un resumen en ./figuras/p3_resultados.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

HERE = Path(__file__).parent
FIG = HERE / "figuras"
FIG.mkdir(exist_ok=True)
out = []

# --------------------------------------------------------------------------
# Carga y limpieza
# --------------------------------------------------------------------------
raw = pd.read_csv(HERE / "data" / "Alturas.csv", encoding="utf-8-sig")
raw.columns = ["id", "hijo", "sexo", "padre", "madre"]
out.append(f"Respuestas: {len(raw)}")
out.append("Valores faltantes por columna:\n" + raw.isna().sum().to_string())
out.append("Filas con faltantes (id): " + str(raw[raw.isna().any(axis=1)]["id"].tolist()))

df = raw.dropna().copy()                         # 2 filas con datos faltantes (id 4 y 36)
df["mujer"] = (df["sexo"] == "Femenino").astype(int)
out.append(f"\nFilas usadas: {len(df)}  (mujeres: {df['mujer'].sum()}, varones: {(1 - df['mujer']).sum()})")
out.append("\nResumen por sexo:\n" + df.groupby("sexo")[["hijo", "padre", "madre"]].agg(["mean", "std"]).round(1).to_string())
out.append("\nCorrelaciones:\n" + df[["hijo", "padre", "madre", "mujer"]].corr().round(2).to_string())

# --------------------------------------------------------------------------
# Comparación de modelos candidatos
# --------------------------------------------------------------------------
formulas = {
    "M1: padre":                    "hijo ~ padre",
    "M2: padre + madre":            "hijo ~ padre + madre",
    "M3: sexo":                     "hijo ~ mujer",
    "M4: padre + sexo":             "hijo ~ padre + mujer",
    "M5: padre + madre + sexo":     "hijo ~ padre + madre + mujer",
    "M6: M5 + padre:sexo + madre:sexo": "hijo ~ padre + madre + mujer + padre:mujer + madre:mujer",
}


def loocv_mse(res):
    """MSE de leave-one-out sin reajustar: e_i / (1 - h_ii)."""
    h = res.get_influence().hat_matrix_diag
    return np.mean((res.resid / (1 - h)) ** 2)


filas = []
fits = {}
for nombre, f in formulas.items():
    r = smf.ols(f, df).fit()
    fits[nombre] = r
    filas.append({"modelo": nombre, "R2 ajust.": r.rsquared_adj, "AIC": r.aic,
                  "sigma (cm)": np.sqrt(r.scale), "LOOCV MSE": loocv_mse(r)})
tabla = pd.DataFrame(filas).set_index("modelo")
out.append("\nComparación de modelos (45 - 2 = 43 observaciones):\n" + tabla.round(2).to_string())

# --------------------------------------------------------------------------
# Modelo elegido: M5
# --------------------------------------------------------------------------
m5 = fits["M5: padre + madre + sexo"]
out.append("\n" + str(m5.summary()))
ic = m5.conf_int()
out.append("\nIntervalos de confianza 95 %:\n" + ic.round(2).to_string())

# Intercepto interpretable: centrando padre y madre en sus promedios
mp, mm = df["padre"].mean(), df["madre"].mean()
dfc = df.assign(padre_c=df["padre"] - mp, madre_c=df["madre"] - mm)
m5c = smf.ols("hijo ~ padre_c + madre_c + mujer", dfc).fit()
out.append(f"\nPromedios: padre = {mp:.1f} cm, madre = {mm:.1f} cm")
out.append("Modelo con padre/madre centrados (mismos coeficientes salvo el intercepto):\n"
           + m5c.params.round(3).to_string())

# --------------------------------------------------------------------------
# Diagnóstico: observaciones atípicas / influyentes
# --------------------------------------------------------------------------
infl = m5.get_influence()
cook = infl.cooks_distance[0]
stud = infl.resid_studentized_external
d = df.assign(cook=cook, estud=stud, hat=infl.hat_matrix_diag)
out.append("\nObservaciones más influyentes (distancia de Cook):\n"
           + d.sort_values("cook", ascending=False).head(5)[["id", "hijo", "sexo", "padre", "madre", "cook", "estud"]]
           .round(2).to_string())
umbral = 4 / len(df)
out.append(f"Umbral orientativo de Cook 4/n = {umbral:.3f}")

# Sensibilidad: sacar la observación más influyente / las que |residuo estudentizado| > 2
sospechosas = d[(d["cook"] > umbral) | (d["estud"].abs() > 2)]["id"].tolist()
out.append(f"\nObservaciones sobre el umbral de Cook o con |residuo estud.| > 2: {sospechosas}")
mas_infl = int(d.sort_values("cook", ascending=False).iloc[0]["id"])
m5_sin = smf.ols("hijo ~ padre + madre + mujer", df[df["id"] != mas_infl]).fit()
out.append(f"\nSensibilidad: modelo M5 sin la observación id={mas_infl}:\n"
           + pd.DataFrame({"con todos": m5.params, "sin esa obs.": m5_sin.params,
                           "ee con": m5.bse, "ee sin": m5_sin.bse}).round(3).to_string()
           + f"\n sigma con = {np.sqrt(m5.scale):.2f}  sin = {np.sqrt(m5_sin.scale):.2f}"
           + f"   R2aj con = {m5.rsquared_adj:.3f}  sin = {m5_sin.rsquared_adj:.3f}")

# Sensibilidad: imputar al estudiante sin sexo (id 4) como varón
d4 = raw[raw["id"] == 4].copy()
d4["sexo"] = "Masculino"; d4["mujer"] = 0
m5_imp = smf.ols("hijo ~ padre + madre + mujer", pd.concat([df, d4])).fit()
out.append("\nSensibilidad: agregando id=4 como varón:\n" + m5_imp.params.round(3).to_string())

# --------------------------------------------------------------------------
# Gráficas
# --------------------------------------------------------------------------
col = {"Masculino": "#3182bd", "Femenino": "#e6550d"}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for s, g in df.groupby("sexo"):
    ax[0].scatter(g["padre"], g["hijo"], c=col[s], s=22, label=s)
    ax[1].scatter(g["madre"], g["hijo"], c=col[s], s=22, label=s)
ax[0].set_xlabel("altura del padre (cm)"); ax[0].set_ylabel("altura del hijo/a (cm)")
ax[0].set_title("Hijo/a vs padre"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("altura de la madre (cm)"); ax[1].set_title("Hijo/a vs madre")
for s, g in d.groupby("sexo"):
    ax[2].scatter(m5.fittedvalues[g.index], g["hijo"], c=col[s], s=22)
for _, r in d[d["cook"] > umbral].iterrows():
    ax[2].annotate(f"id {int(r['id'])}", (m5.fittedvalues[_], r["hijo"]), fontsize=7,
                   xytext=(4, 3), textcoords="offset points")
lim = [df["hijo"].min() - 2, df["hijo"].max() + 2]
ax[2].plot(lim, lim, "k--", lw=1)
ax[2].set_xlabel("altura ajustada (cm)"); ax[2].set_ylabel("altura real (cm)")
ax[2].set_title("Modelo elegido: ajustado vs real")
plt.tight_layout()
plt.savefig(FIG / "p3_alturas.png", dpi=200)
plt.close()

txt = "\n".join(out)
print(txt)
(FIG / "p3_resultados.txt").write_text(txt, encoding="utf-8")
