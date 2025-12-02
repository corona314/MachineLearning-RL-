# Quantum vs Classical Pong Agent — Benchmarking Environment

Este repositorio contiene un entorno simplificado de Pong junto con dos agentes de aprendizaje por refuerzo:
- **Agente clásico tabular (Q-Learning)**
- **Agente cuántico basado en QNN (Qiskit + PyTorch)**

El propósito es comparar su rendimiento bajo un presupuesto fijo de episodios o tiempo, generando métricas reproducibles y gráficas automáticas.

---

## Requisitos

Las dependencias exactas están en `requirements.txt`, pero como referencia:

```text
# Librerías científicas
numpy==1.26.0
matplotlib==3.8.0
pandas==2.2.2

# IPython display
ipython==8.17.0

# PyTorch y torchvision (CPU)
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1

# Qiskit y Machine Learning
qiskit==0.45.0
qiskit-machine-learning==0.7.1
qiskit-aer==0.13.0
```

## Instalación y entorno virtual

Crear el entorno virtual:
```
python -m venv quantum_env
```

Activarlo:

Windows (PowerShell)
```
.\quantum_env\Scripts\Activate.ps1
```

Linux/Mac
```
source quantum_env/bin/activate
```


Instalar dependencias:
```
pip install -r requirements.txt
```

## Ejecución del experimento

El script principal es experiment_new.py, que acepta parámetros vía CLI:
```
python experiment.py --mode --budget --repeats --out --agent --reward
```

##### Parámetros disponibles
Parámetro ->	Tipo -> Descripción
--mode ->	episodes o time ->	Controla si se limita por episodios o por tiempo real
--budget -> float -> Nº de episodios o segundos según el modo
--repeats -> int -> Número de repeticiones del experimento (default: 1)
--out -> str -> Carpeta donde guardar métricas y gráficos
--agent -> quantum o classical -> Selección del agente (default: both)
--reward -> float -> Recompensa por acierto (default: 10)


## Resultados

Tras cada ejecución se generan:
CSV con métricas por episodio
Gráficas de comparación
Logs resumidos
Carpeta de salida configurable
Todo aparece en la ruta que definas con --out.
