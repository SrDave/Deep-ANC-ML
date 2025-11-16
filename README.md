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

## No linealidades en el sistema ANC

Se pueden introducir no linealidades en diferentes partes del sistema:

- **Camino primario:** respecto de la señal de referencia
- **Señal de referencia:** ruido que llega al punto de control
- **Camino secundario:** en la generación de la señal por el altavoz
- Combinaciones de los anteriores

### Modelos de no linealidad implementados

1. **Modelo polinómico:**
   $$x^{nl} (n) = 2a x(n) + ax^2(n) + x^3(n), \quad a = \log(\epsilon/10)+0.1, \epsilon = [2,3,4,5]$$
2. **Exponencial:** 
   $$x^{nl}(n) = 1 - e^{-0.3x(n)}$$
3. **Saturación hard:** valores limitados entre \(-x_{\max}\) y \(x_{\max}\)  
4. **Saturación soft:** 
   $$x^{nl}(n)=\frac{x(n)x_{max}}{\sqrt[q]{|x_{max}|^q+|x(n)|^q}}$$
5. **Sigmoide:**
   $$x^{nl}(k)=\lambda\left(\frac{1}{1+e^{-vz(n)}}-\frac{1}{2} \right), \quad z(n)=1.5x(n)-0.3x^2(n)$$
6. **Kernel Volterra segundo orden:**
   $$x^{nl}(n) = h_0 + \sum_k h_1(k)x(n-k) + \sum_{k_1,k_2} h_2(k_1,k_2)x(n-k_1)x(n-k_2)$$

> Los modelos 1–5 son sin memoria; el Volterra (6) incluye memoria.

---

### Implementación en Python / TensorFlow

```python
from tensorflow.keras.layers import Layer, Conv1D, Activation, Concatenate, Dense, Input
import tensorflow as tf
import numpy as np

# Función de no linealidad
def no_linealidad_tf(x, tipo=0):
    x = tf.cast(x, tf.float32)
    if tipo==1: a = tf.math.log(2.0 / 10)/tf.math.log(10) + 0.1; return 2*a*x + a*x**2 + x**3
    elif tipo==2: return 1 - tf.exp(-0.3*x)
    elif tipo==3: return tf.clip_by_value(x, -0.7, 0.7)
    elif tipo==4:
        max_x = tf.reduce_max(x)
        return x*max_x/(tf.pow(tf.abs(max_x),2)+tf.pow(tf.abs(x),2))**0.5
    elif tipo==5:
        z = 1.5*x-0.3*x**2
        v = tf.where(z>0, 4.0, 0.5)
        return 4*(1/(1+tf.exp(-v*z))-0.5)
    else: return x
```

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
