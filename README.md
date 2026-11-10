# MEMERAG: base experimental de jueces LLM (español)

Experimento piloto: comparar un juez LLM con anotaciones humanas de fidelidad **por oración**.

## Preparación (Windows PowerShell)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Colocar la clave de API en `.env`. 

## Dataset

Fuente oficial: https://github.com/amazon-science/MEMERAG

Descargar el JSONL de español del dataset original o de la versión de votación mayoritaria y colocarlo en `data/raw/es.jsonl`. Verificar en el repositorio oficial el nombre y ubicación exactos del archivo, y que el esquema contenga `query`, `context`, `answer` con `sentence`, `sentence_id` y `factuality`. El script **no** utiliza predicciones precomputadas.

## Ejecutar

```powershell
python src/prepare_data.py --file data/raw/es.jsonl --n 30 --seed 42
python src/run_experiment.py                                   # proveedor por defecto (PROVEEDOR)
python src/run_experiment.py --proveedor openai                # juez OpenAI
python src/run_experiment.py --proveedor anthropic --nombre medium   # -> results/<modelo>_medium-<fecha>-<hora>/
```

Las API keys van en `.env` (no versionado). En `src/config.py`, `API_KEY_ENV` indica qué variable de `.env` usa cada proveedor; para usar otra clave basta con cambiar ese nombre (p. ej. `ANTHROPIC_API_KEY_CURSO`).

Cada ejecución crea una carpeta nueva `results/<modelo>[_<nombre>]-<AAAAMMDD>-<HHMMSS>/`: siempre empieza por el modelo realmente usado y `--nombre` solo añade una etiqueta opcional, por lo que **ninguna corrida sobrescribe a otra**. El modelo se define **solo en `.env`** (`OPENAI_MODEL`, `ANTHROPIC_MODEL`); para comparar modelos o parámetros, editar `.env` (`ANTHROPIC_MODEL`, `ANTHROPIC_EFFORT`, …) o usar `--proveedor`, y volver a ejecutar.

Contenido de cada carpeta de corrida:

- `config.json`: proveedor, modelo, parámetros, prompt del sistema, archivo de entrada, hora de inicio/fin y número de errores de API.
- `predictions.csv`: salida del juez.
- `metrics.json`, `disagreements.csv`, `confusion_matrix.png`: generados por `evaluate.py`.
- `error_analysis/`: `report.html`, `disagreements_detailed.csv`, `summary.json`, `categories.png`.

Los informes se generan automáticamente (usar `--skip-reports` para omitirlos).
Para regenerarlos sobre una corrida existente:

```powershell
python src/evaluate.py --input results/<corrida>/predictions.csv
python src/analyze_errors.py --predictions results/<corrida>/predictions.csv
```

Ambos escriben por defecto dentro de la carpeta del CSV indicado.

## Metodología

- Etiquetas humanas: `Supported` (1) y `Not Supported` (0).
- Se excluyen las etiquetas humanas `Challenging to determine` del piloto binario.
- La unidad de evaluación es una oración de respuesta, no una respuesta completa.
- El juez no recibe la etiqueta humana.
- Se guarda la respuesta textual del juez para auditoría.
- Se registra el modelo y se fija la semilla de muestreo.
- El tamaño pequeño sirve para validar la infraestructura, no para sacar conclusiones generales.

## Limitaciones

La selección es aleatoria simple, no estratificada. Puede presentar desequilibrio de clases. La temperatura 0 no garantiza determinismo completo. Documentar versión del modelo, fecha, costos y procedencia exacta del dataset. Para una comparación formal con el artículo MEMERAG, usar sus prompts y protocolo oficial.
