# Proyecto: Transformer Nahuatl–Español (From Scratch)

Este proyecto implementa un modelo de traducción automática basado en la arquitectura **Transformer**, diseñado para la traducción bidireccional entre **español y náhuatl**. 
El trabajo forma parte de una investigación académica orientada a la preservación y revitalización digital del náhuatl mediante técnicas de **Procesamiento de Lenguaje Natural (PLN)**.

## Estructura del proyecto

```
articulo/
│── checkpoints/              # Pesos del modelo entrenado
│── logs/                     # Registros de entrenamiento
│── salida/                   # Resultados y evaluaciones
│── spm/                      # Archivos de tokenización (SentencePiece)
│── corpus_ncx_es.xlsx        # Corpus paralelo (náhuatl ↔ español)
│── Transformer_Nahuatl_Espanol_FromScratch.py   # Script principal de entrenamiento
│── jw_ncx_es_corpus.txt      # Corpus JW paralelo en formato texto
```

## Requisitos

- Python 3.10 o superior
- PyTorch
- SentencePiece
- NumPy y Pandas
- Matplotlib (opcional, para visualización de resultados)
- CUDA (opcional, para entrenamiento en GPU)

Instalación de dependencias:

```bash
pip install torch sentencepiece numpy pandas matplotlib
```

## Instrucciones de uso

### 1. Preprocesamiento de datos
Ejecutar la tokenización con SentencePiece sobre el corpus:

```bash
python spm/train_spm.py
```

### 2. Entrenamiento del modelo
Entrenar el modelo Transformer desde cero:

```bash
python Transformer_Nahuatl_Espanol_FromScratch.py --train
```

### 3. Evaluación del modelo
Validar el modelo y obtener métricas de desempeño (BLEU, ROUGE):

```bash
python evaluate.py
```

### 4. Traducción de prueba
Ejecutar una traducción de ejemplo:

```bash
python translate.py --input "¿Cómo estás?" --lang es-ncx
```

Salida esperada:

```
Náhuatl: Quen tinemi?
```

## Resultados esperados

- Cálculo de métricas BLEU y ROUGE sobre el corpus JW.
- Traducciones coherentes entre **náhuatl ↔ español**.
- Registros de entrenamiento disponibles en `logs/` y modelos entrenados en `checkpoints/`.

## Autores

- Samuel Pérez Zistecatl – Maestría en Sistemas Computacionales, TecNM Campus Apizaco.  

Este proyecto se desarrolla en el marco de un trabajo académico de investigación sobre traducción automática español–náhuatl.
