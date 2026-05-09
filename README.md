# clasificacion-documentos-ia



# INSTALACION

## Crear entorno virtual

```bash
python -m venv venv
```

## Ejecutar bypass solo si no esta activa la ejecuion de scripts


```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```


## Activar entorno virtual

```bash
venv\Scripts\activate
```



## Instalar dependencias

```bash
pip install -r install.txt
```

## Ejecutar instalaciones manuales

```bash
python -m spacy download es_core_news_md
```

```bash
python -m pywin32_postinstall install
```



# Arquitectura General del Sistema

```text
Interfaz local
│
├─ Selección de carpeta
├─ Escaneo del repositorio
├─ Visualización de archivos
└─ Lanzar clasificación / OCR / reorganización

Backend documental
│
├─ Ingesta de archivos
│
├─ Clasificación técnica inicial
│   ├─ Texto nativo
│   ├─ Candidato OCR
│   └─ No soportado / error
│
├─ Extracción de texto
│   ├─ PDF
│   ├─ Word
│   ├─ Excel
│   ├─ TXT
│   ├─ XML
│   └─ OCR para imágenes o PDFs escaneados
│
├─ Normalización de texto
│
├─ Representación semántica
│   ├─ embeddings
│   ├─ etiquetas
│   └─ resumen breve
│
├─ Clasificación documental
│   ├─ tipo documental
│   ├─ temática
│   ├─ periodo temporal
│   └─ normativa / área si aplica
│
├─ Detección de duplicados
│
└─ Agente de reorganización
    ├─ propone carpeta destino
    ├─ detecta archivos mal ubicados
    └─ genera plan de reorganización

Modo seguro
│
├─ Shadow mode
│   └─ solo propone cambios
│
└─ Modo ejecución
    └─ mueve / copia archivos tras validación

Evaluación
│
├─ precisión de clasificación
├─ recall
├─ F1-score
├─ cobertura de categorías
└─ reducción de tiempo frente a clasificación manual
```


# Pipeline Inteligente de Clasificación Documental



```text

1. Documento entra
   ↓
2. Extracción de texto por tipo de archivo
   ├─ Word
   ├─ PDF
   ├─ Excel
   ├─ TXT
   ├─ XML
   └─ Imagen / OCR
   ↓
3. Normalización documental
   ├─ limpieza
   ├─ bloques
   ├─ OCR si aplica
   ├─ PLN / entidades
   └─ representación estructurada
   ↓
4. LLM local analiza el documento
   ↓
   Genera:
   ├─ clasificación sugerida
   ├─ tipo_documento
   ├─ área_funcional
   ├─ tema
   ├─ palabras clave
   └─ resumen
   ↓
5. Embeddings comparan la salida semántica
   ↓
   Contra:
   ├─ categorías existentes
   ├─ documentos previos
   └─ memoria semántica
   ↓
6. Decisión del agente
   ├─ similitud alta
   │   └─ reutiliza categoría existente
   │
   ├─ similitud media
   │   └─ valida con LLM / revisión
   │
   └─ similitud baja
       └─ crea categoría nueva
   ↓
7. Guardar memoria semántica
   ├─ categoría final
   ├─ embedding
   ├─ ejemplos
   ├─ confianza
   └─ historial
   ↓
8. Reorganización documental
   ├─ propone carpeta destino
   ├─ detecta duplicados
   └─ modo simulación / shadow mode
   ↓
9. Evaluación
   ├─ accuracy
   ├─ precision
   ├─ recall
   ├─ F1-score
   └─ tiempo de procesamiento

```

## Enfoque de Arquitectura

- El LLM local actúa como componente principal de interpretación y clasificación.

- Los embeddings funcionan como soporte semántico para:
  - reutilizar categorías existentes
  - evitar categorías duplicadas
  - detectar similitud documental
  - mejorar consistencia de clasificación


# EJEMPLO DE SALIDA DE TEXTO NORMALIZADO


```text
FEATURES GENERALES
====================

UBICACIONES:
- Ciudad Central
- Zona Empresarial Norte
- Parque Industrial Delta

PERSONAS:
- Jane Doe
- Jon Doe
- Jim Doe
- Jack Doe
- Joe Doe

CONCEPTOS:
- Oficina principal
- Proyecto Alpha

====================

TIPO: paragraph

FECHAS:
- 15 de marzo de 2026

TEXTO:
Ciudad Central, 15 de marzo de 2026.

====================

TIPO: paragraph

TEXTO:
Reporte interno No.204-2026...Ref....JDD./adm.

====================

TIPO: paragraph

TEXTO:
DEPARTAMENTO DE OPERACIONES ZONA EMPRESARIAL NORTE.

====================

TIPO: paragraph

HORAS:
- 09:30

TEXTO:
El día de hoy a las 09:30 horas se realizó una reunión entre los integrantes del equipo para discutir avances relacionados con el Proyecto Alpha.

====================

TIPO: paragraph

CANTIDADES:
- 5 años

TEXTO:
Jane Doe, con experiencia de 5 años en administración de proyectos, presentó el resumen ejecutivo correspondiente al cierre trimestral.

====================

TIPO: paragraph

FECHAS:
- 16-03-2026

HORAS:
- 14:00
- 16:30

CANTIDADES:
- 3 equipos

TEXTO:
Jon Doe indicó que el día 16-03-2026 entre las 14:00 y 16:30 horas se completó la instalación de 3 equipos en la Oficina Principal del Parque Industrial Delta.

====================

TIPO: paragraph

TEXTO:
Jim Doe solicitó que el documento final fuera enviado al área administrativa para revisión y validación correspondiente.

====================

TIPO: paragraph

TEXTO:
Jack Doe confirmó la recepción del informe y Joe Doe quedó encargado del seguimiento técnico del Proyecto Alpha.
```
