# 🎧 Deep ANC: Control Activo de Ruido con Aprendizaje Profundo

Implementación de un sistema de **Active Noise Control (ANC)** basado en redes neuronales profundas, siguiendo el paper:  
> **Deep ANC: A deep learning approach to active noise control**  
> *Neural Networks, Vol. 141 (2021), pp. 1–10*

---

## Descripción del sistema

El sistema simula una sala acústica con **Pyroomacoustics**, genera las **Respuestas al Impulso (RIR)** para los caminos **primario** (fuente de ruido → micrófono) y **secundario** (altavoz → micrófono), y utiliza modelos de **Machine Learning** para aprender filtros óptimos de cancelación activa de ruido.

---

## Características principales

- Simulación acústica 3D realista con `pyroomacoustics`
- Modelado del entorno usando RIRs (Respuestas al Impulso)
- Implementación clásica de filtros LMS y solución por mínimos cuadrados
- Modelos de **redes neuronales convolucionales (Conv1D)** con TensorFlow/Keras
- Entrenamiento supervisado y evaluación offline / tiempo real
- Comparación entre enfoques tradicionales y basados en aprendizaje profundo

---

## Estructura del proyecto

- **Simulación acústica:** creación del entorno y obtención de RIRs  
- **Solución analítica:** filtro óptimo por mínimos cuadrados  
- **Solución adaptativa:** LMS clásico  
- **Solución ML:** modelo CNN para aprendizaje del filtro de cancelación  
- **Simulaciones:** evaluación offline y en tiempo real  

---

## Dependencias principales

```bash
pip install numpy matplotlib scipy tensorflow scikit-learn pyroomacoustics
```

---

## Ejecución básica

### 1️⃣ Clona el repositorio
```
git clone https://github.com/tuusuario/Deep-ANC-ML.git
cd Deep-ANC-ML
```

 ### 2️⃣ Ejecuta el script principal
```
python deep_anc.py
```

### 3️⃣ Visualiza los resultados

- RIRs generadas (camino primario y secundario)

- Filtros aprendidos por ML

- Comparativa de señales antes y después de la cancelación


---

## Resultados esperados

- Atenuación significativa del ruido residual en el micrófono de error

- Filtros aprendidos similares a los obtenidos con métodos óptimos

- Reducción progresiva del error de cancelación en las simulaciones

---

## Ejemplo de resultado

![Resultado de cancelación](resultadocancelacion.png)

---

## Referencias

- Deep ANC: A deep learning approach to active noise control, Neural Networks 141 (2021) 1–10

- Pyroomacoustics Documentation

- TensorFlow/Keras API Reference

  ---

 ## Autor

David Ramos

Grado en Tecnologías Interactivas y de Diseño de Medios (GTDM)

Universitat Politècnica de València (UPV)
