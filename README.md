# 🧠 ElectronicComponentCNN

Este proyecto implementa una **Red Neuronal Convolucional (CNN)** para la clasificación automática de componentes electrónicos a partir de imágenes.  
El modelo ha sido diseñado para reconocer distintas clases de componentes (ej. resistores, capacitores, transistores, etc.) en imágenes en escala de grises.

---

## ⚙️ Características principales
- **Arquitectura CNN**: capas convolucionales + fully connected para clasificación.
- **Entrenamiento**: realizado con imágenes organizadas en carpetas (`ImageFolder`), donde cada carpeta representa una clase.
- **Número de clases**: se determina automáticamente según las subcarpetas en `./train_data`.
- **Preprocesamiento de imágenes**:
  - Conversión a **grayscale** (1 canal).
  - Redimensionado a **256×256 píxeles**.
  - Normalización en el rango [-1, 1].
- **Inferencia**: dado un archivo de imagen, el modelo predice la clase correspondiente.

---

## 📂 Estructura del proyecto
```bash
ElectronicComponentCNN/ 
│── model.py # Definición de la arquitectura CNN 
│── train.py # Script de entrenamiento 
│── checkpoints/ # Pesos entrenados (.pth) 
│── test_data/ # Imágenes de prueba organizadas por clase 
│── show_results.ipynb # Notebook para visualizar resultados
│── train_data/ # Dataset de entrenamiento
```


---

## 🚀 Uso básico
1. **Entrenamiento**:
   ```bash
   python train.py --data_dir ./train_data --epochs 50 --batch_size 4
2. **Inferencia en notebook**:

Cargar el modelo y los pesos .pth.

Preprocesar la imagen (grayscale, resize, normalize).

Ejecutar inferencia y obtener la clase predicha.

---
## 📊 Ejemplo de salida

Archivo	Clase (índice)	Etiqueta
img1.png	0	           resistor
img2.png	1	          capacitor
img3.png	2         	transistor
