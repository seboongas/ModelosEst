# ModelosEst

Grupo conformado por Sebastian Guigou, Francisco Scaletti y Dionisio Bell.

Entrega 1 de *Modelos Estadísticos para la Regresión y la Clasificación* (2026).

## Contenido

| Archivo | Qué hace |
|---|---|
| `p1_mle.py` | Problema 1: simulación por transformada inversa, log-verosimilitud, golden search y Monte Carlo (sesgo, varianza y MSE para n = 10, 100 y 1000). |
| `p2_boston.py` | Problema 2: Boston Housing, regresión lineal, polinomios de grado 2 y 3 y Ridge con λ elegido por validación cruzada. |
| `p3_alturas.py` | Problema 3: modelo lineal para la altura de los estudiantes a partir de la altura del padre, de la madre y el sexo. |
| `data/Alturas.csv` | Datos de alturas recabados en clase. |
| `data/Boston.csv` | Dataset `Boston` del paquete ISLP (12 atributos + `medv`). |
| `figuras/` | Gráficas y resúmenes numéricos generados por los scripts. |
| `informe/informe.tex`, `informe/informe.pdf` | Informe de la entrega (3 carillas). |

## Cómo correrlo

```bash
pip install -r requirements.txt
python p1_mle.py
python p2_boston.py
python p3_alturas.py
```

Cada script guarda sus gráficas y un `.txt` con los resultados en `figuras/`. Todos usan semillas fijas, así que los números del informe se reproducen.
Para recompilar el informe: `cd informe && pdflatex informe.tex`.
