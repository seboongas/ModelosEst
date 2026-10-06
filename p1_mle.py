# -*- coding: utf-8 -*-
"""
Problema 1 - Estimador de máxima verosimilitud
Densidad: f(x) = 1/2 (1 + theta x),  x in [-1, 1],  theta in [-1, 1].
Valor verdadero: theta = -1/3.

Uso:  python p1_mle.py
Genera figuras en ./figuras y un resumen en ./figuras/p1_resultados.txt
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

FIG = Path(__file__).parent / "figuras"
FIG.mkdir(exist_ok=True)

THETA = -1 / 3
rng = np.random.default_rng(2026)


# --------------------------------------------------------------------------
# 1. Simulación por transformada inversa
# --------------------------------------------------------------------------
# F(x) = (x + 1)/2 + theta (x^2 - 1)/4,  x en [-1, 1].
# Resolviendo F(x) = u:  theta x^2 + 2 x + (2 - theta - 4u) = 0
#   x = (-1 + sqrt(1 - theta (2 - theta - 4u))) / theta     (raíz que cae en [-1, 1])
def simular(n, theta=THETA, rng=rng):
    """Simula una muestra de tamaño n de f(x) = 1/2 (1 + theta x) en [-1, 1]."""
    u = rng.uniform(0, 1, n)
    if abs(theta) < 1e-12:                     # theta = 0: uniforme en [-1, 1]
        return 2 * u - 1
    return (-1 + np.sqrt(1 - theta * (2 - theta - 4 * u))) / theta


# --------------------------------------------------------------------------
# 2. Verosimilitud y Golden search
# --------------------------------------------------------------------------
def loglik(theta, x):
    """log L(theta) = sum log f(x_i | theta) = sum log(1 + theta x_i) - n log 2."""
    return np.sum(np.log(1 + theta * x)) - len(x) * np.log(2)


def neg_loglik(theta, x):
    return -loglik(theta, x)


def golden_search(f, a, b, tol=1e-8):
    """Minimiza una función unimodal f en [a, b] por búsqueda de la sección áurea."""
    phi = (np.sqrt(5) - 1) / 2                 # ~ 0.618
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    while abs(b - a) > tol:
        if fc < fd:                            # el mínimo está en [a, d]
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:                                  # el mínimo está en [c, b]
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return (a + b) / 2


def mle(x):
    """Estimador de máxima verosimilitud de theta (golden search sobre [-1, 1])."""
    return golden_search(lambda t: neg_loglik(t, x), -1.0, 1.0)


def montecarlo(n, M, theta=THETA):
    """M muestras de tamaño n -> M estimaciones de theta."""
    return np.array([mle(simular(n, theta)) for _ in range(M)])


def resumen(est, theta=THETA):
    sesgo = est.mean() - theta
    var = est.var()                            # ddof=0  =>  MSE = sesgo^2 + var exacto
    mse = np.mean((est - theta) ** 2)
    return sesgo, var, mse


if __name__ == "__main__":
    out = []

    # ---- Parte 1: chequeo del simulador --------------------------------
    x_chk = simular(200_000)
    xs = np.linspace(-1, 1, 200)
    assert x_chk.min() >= -1 and x_chk.max() <= 1
    out.append(f"E[X] simulado = {x_chk.mean():.4f}  (teórico theta/3 = {THETA/3:.4f})")

    # ---- Parte 2: muestras de tamaño 10 ---------------------------------
    # Se grafican 3 muestras distintas para ver que el máximo cambia con la muestra
    grid = np.linspace(-0.999, 0.999, 800)
    rng_demo = np.random.default_rng(1)          # semilla aparte para las 3 muestras ilustrativas
    colores = ["#3182bd", "#e6550d", "#31a354"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    ax[0].hist(x_chk, bins=60, density=True, color="#9ecae1", edgecolor="white",
               label="muestra simulada")
    ax[0].plot(xs, 0.5 * (1 + THETA * xs), "r", lw=2, label=r"$f(x)$ teórica")
    ax[0].set_title(r"Simulador ($\theta=-1/3$)")
    ax[0].set_xlabel("x"); ax[0].legend(fontsize=8)
    for k, c in enumerate(colores):
        xk = simular(10, rng=rng_demo)
        ll = np.array([loglik(t, xk) for t in grid])
        th_hat = mle(xk)
        th_sp = minimize_scalar(lambda t: neg_loglik(t, xk), bounds=(-1, 1),
                                method="bounded", options={"xatol": 1e-10}).x
        out.append(f"Muestra {k+1} (n=10): theta_hat (golden) = {th_hat:.5f} | scipy = {th_sp:.5f}")
        assert abs(th_hat - th_sp) < 1e-4
        ax[1].plot(grid, np.exp(ll - ll.max()), color=c)
        ax[1].plot(th_hat, 1, "o", color=c)
        ax[2].plot(grid, -ll, color=c)
        ax[2].plot(th_hat, -loglik(th_hat, xk), "o", color=c)
    ax[1].axvline(THETA, color="r", ls=":", label=r"$\theta$ real")
    ax[1].set_title(r"$L(\theta)/\max L$  ($n=10$, 3 muestras)")
    ax[1].set_xlabel(r"$\theta$"); ax[1].legend(fontsize=8)
    ax[2].axvline(THETA, color="r", ls=":")
    ax[2].set_title(r"$-\log L(\theta)$  ($n=10$, puntos = $\hat\theta$)")
    ax[2].set_xlabel(r"$\theta$")
    plt.tight_layout()
    plt.savefig(FIG / "p1_verosimilitud.png", dpi=200)
    plt.close()

    # ---- Partes 3 y 4: Monte Carlo ------------------------------------
    M = 1000
    resultados = {}
    out.append(f"\nMonte Carlo con M = {M} muestras por tamaño")
    out.append(f"{'n':>6} {'media(theta_hat)':>17} {'sesgo':>9} {'varianza':>10} {'MSE':>10} {'%en borde':>10}")
    for n in (10, 100, 1000):
        est = montecarlo(n, M)
        resultados[n] = est
        s, v, m = resumen(est)
        borde = 100 * np.mean(np.abs(est) > 1 - 1e-6)
        out.append(f"{n:>6} {est.mean():>17.4f} {s:>9.4f} {v:>10.5f} {m:>10.5f} {borde:>9.1f}%")

    # Con M = 50 (como sugiere la letra) para n = 10, para comparar el ruido de Monte Carlo
    est50 = montecarlo(10, 50)
    s, v, m = resumen(est50)
    out.append(f"\n(n=10, M=50)   sesgo={s:.4f}  var={v:.5f}  MSE={m:.5f}")

    # ---- Comparación con la teoría asintótica -------------------------
    I = quad(lambda x: x ** 2 / (2 * (1 + THETA * x)), -1, 1)[0]   # información de Fisher
    out.append(f"\nInformación de Fisher I(theta) = {I:.4f};  1/I = {1/I:.4f}")
    for n in (10, 100, 1000):
        out.append(f"  n={n:>4}:  n*Var simulada = {n * resultados[n].var():.3f}   (teoría asintótica 1/I = {1/I:.3f})")

    # ---- Figura de las distribuciones del estimador -------------------
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.9), sharex=True)
    for a, n in zip(ax, (10, 100, 1000)):
        e = resultados[n]
        a.hist(e, bins=30, color="#9ecae1", edgecolor="white")
        a.axvline(THETA, color="r", ls="--", label=r"$\theta=-1/3$")
        a.axvline(e.mean(), color="k", ls="-", label="media de $\\hat\\theta$")
        a.set_title(f"n = {n}   (var = {e.var():.4f})")
        a.set_xlabel(r"$\hat\theta$")
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "p1_distribucion_estimador.png", dpi=200)
    plt.close()

    txt = "\n".join(out)
    print(txt)
    (FIG / "p1_resultados.txt").write_text(txt, encoding="utf-8")
