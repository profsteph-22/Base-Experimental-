"""Parámetros generales del experimento.

Punto único para cambiar rutas, muestreo, prompt y modelos. Los valores que dependen
del entorno (modelos, esfuerzo) se leen de variables de entorno / .env, con el valor
por defecto indicado aquí. Las rutas son relativas a la raíz del repositorio, desde
donde deben ejecutarse los scripts.

Se importa como `import config as cfg` para no chocar con las variables locales `config`.
"""
import os
from dotenv import load_dotenv

# Carga .env antes de leer las variables de abajo. override=True: .env manda sobre variables
# que hayan quedado definidas en la sesión de la terminal (p. ej. un $env:ANTHROPIC_MODEL viejo).
# .env es la única fuente del modelo (OPENAI_MODEL / ANTHROPIC_MODEL); no hay argumento --modelo.
load_dotenv(override=True)

# --- Rutas ---------------------------------------------------------------
ARCHIVO_MUESTRA = 'data/processed/sample.jsonl'   # salida de prepare_data.py / entrada de los jueces
CARPETA_RESULTADOS = 'results'                    # carpeta base de las corridas
CARPETA_ANALISIS_ERRORES = 'error_analysis'       # subcarpeta de analyze_errors.py dentro de cada corrida

# --- Dataset y muestreo --------------------------------------------------
MEMERAG_URL = 'https://raw.githubusercontent.com/amazon-science/MEMERAG/main/data/memerag_ext_w_majority_voting/es.jsonl'
TAMANO_MUESTRA = 30
SEMILLA = 42
ETIQUETAS = ('Supported', 'Not Supported')       # etiquetas binarias válidas (humano y juez)

# --- Prompt del juez (compartido por OpenAI y Anthropic) -----------------
SYSTEM_PROMPT = '''Eres un evaluador de fidelidad de RAG. Evalúa SOLO si la afirmación está respaldada por el contexto proporcionado, no por conocimiento externo. Responde exclusivamente un objeto JSON con dos campos: label ("Supported" o "Not Supported") y reason (explicación breve en español). Si no hay evidencia suficiente, responde "Not Supported".'''

# --- Juez ----------------------------------------------------------------
# Proveedor por defecto de run_experiment.py (se puede cambiar con --proveedor).
PROVEEDOR = os.getenv('PROVEEDOR', 'anthropic')   # openai | anthropic

# Nombre de la variable de .env de la que se toma la API key de cada proveedor.
# Las claves NUNCA se escriben aquí: van en .env (no versionado). Para usar otra
# clave basta con cambiar el nombre, p. ej. 'ANTHROPIC_API_KEY_CURSO'.
API_KEY_ENV = {
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
}

# --- Juez OpenAI ---------------------------------------------------------
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
OPENAI_TEMPERATURE = 0

# --- Juez Anthropic ------------------------------------------------------
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-opus-5-5')
ANTHROPIC_EFFORT = os.getenv('ANTHROPIC_EFFORT', 'medium')   # low | medium | high | xhigh | max
ANTHROPIC_MAX_TOKENS = 16000


def obtener_api_key(proveedor):
    """Devuelve la API key del proveedor leyendo la variable configurada en API_KEY_ENV."""
    variable = API_KEY_ENV[proveedor]
    clave = os.getenv(variable, '').strip()
    if not clave:
        raise SystemExit(f'ERROR: falta la API key de {proveedor}. Define {variable} en .env '
                         f'(o cambia API_KEY_ENV["{proveedor}"] en src/config.py).')
    return clave

# --- Informes ------------------------------------------------------------
DPI_GRAFICOS = 160
