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
- San Antonio Aguas Calientes
- Sacatepéquez Su Despacho
- Antigua Guatemala
- departamento de Sacatepéquez

PERSONAS:
- Jane Doe
- Jon Doe
- Jim Doe
- Jack Doe
- Joe Doe

CONCEPTOS:
- Calle final norte
- De Subestacion 74-4-3

====================

TIPO: paragraph

FECHAS:
- 21 de mayo de 2023

TEXTO:
San Antonio Aguas Calientes 21 de mayo de 2023.

====================

TIPO: paragraph

TEXTO:
Diligencia Policial No.123-456...Ref....ARLG./aap.

====================

TIPO: paragraph

TEXTO:
JUEZ DE PAZ LOCAL SAN ANTONIO AGUAS CALIENTES, SACATEPÉQUEZ SU DESPACHO.

====================

TIPO: paragraph

HORAS:
- 10:05

TEXTO:
Atentamente me dirijo a usted, con la finalidad de informarle que el día de hoy siendo las 10:05 horas, a esta subestación policial se presentó la señorita, quien dijo ser de los datos de identificación siguientes.

====================

TIPO: paragraph

CANTIDADES:
- 19 años

TEXTO:
Jane Doe: de 19 años de edad, soltera, instruida, secretaria bilingüe, originaria de Antigua Guatemala y residente en 4a. Calle final norte, del municipio de San Antonio Aguas Calientes, departamento de Sacatepéquez, hija de Jon Doe y de Jim Doe, quien no presentó documento personal de identificación, proporcionando el número de teléfono 5954-3592. Para cualquier notificación.

====================

TIPO: paragraph

FECHAS:
- 21-05-2023

HORAS:
- 23:00
- 23:30

CANTIDADES:
- 20 años

TEXTO:
MANIFESTANDO: que tiene dos meses de convivir con el señor: Jack Doe de 20 años de edad, quien puede ser localizado en la misma dirección de la denunciante o al número de celular 4310-2383, indicando la denunciante que desde hace tres semanas ha tenido inconvenientes con su conviviente por celos, es el caso que el día de ayer 21-05-2023 a eso de las 23:00 horas retornaron de una boda y siendo las 23:30 horas llegaron unos amigos a convivir con ellos y bebieron unas cervezas, en ese transcurso su conviviente la empujó y cayó en las gradas por lo que se ocasionó dos erosiones uno en el brazo izquierdo y pierna izquierda indicando no ameritar asistencia médica.

====================

TIPO: paragraph

TEXTO:
Por lo antes descrito solicita se hiciera del conocimiento de ese juzgado, persona quien fue prevenida de comparecer, en horas y días hábiles, para ratificar lo antes expuesto.

====================

TIPO: paragraph

TEXTO:
Del señor (a) Juez deferentemente.

====================

TIPO: paragraph

TEXTO:
Joe Doe ENCARGADO DE SUBESTACION 74-4-3 SAN ANTONIO AGUAS CALIENTES
```