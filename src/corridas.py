"""Utilidades compartidas para guardar cada corrida en su propia carpeta.

Cada ejecución de un juez crea results/<modelo>-<AAAAMMDD>-<HHMMSS>/ con:
  - config.json        modelo, parámetros, prompt y archivos de entrada
  - predictions.csv    salida del juez
  - metrics.json, disagreements.csv, confusion_matrix.png   (evaluate.py)
  - error_analysis/    (analyze_errors.py)
Así, probar otro modelo o cambiar parámetros nunca sobrescribe una corrida anterior.
"""
import json, re, subprocess, sys
from datetime import datetime
from pathlib import Path
import config as cfg


def agregar_argumentos(parser):
    """Añade los argumentos comunes de corrida a un ArgumentParser."""
    parser.add_argument('--input', default=cfg.ARCHIVO_MUESTRA)
    parser.add_argument('--skip-reports', action='store_true', help='No generar informes automáticos')
    parser.add_argument('--nombre', default=None,
                        help='Etiqueta opcional; la carpeta será <modelo>_<nombre>-<fecha>-<hora> (sin ella: <modelo>-<fecha>-<hora>)')
    parser.add_argument('--resultados', default=cfg.CARPETA_RESULTADOS,
                        help=f'Carpeta general donde se crean las corridas (por defecto: {cfg.CARPETA_RESULTADOS})')


def crear_carpeta(base, nombre):
    """Crea results/<nombre>-<AAAAMMDD>-<HHMMSS>; nunca reutiliza una carpeta existente."""
    nombre = re.sub(r'[^\w.-]+', '_', nombre).strip('_') or 'experimento'
    marca = datetime.now().strftime('%Y%m%d-%H%M%S')
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    carpeta = base / f'{nombre}-{marca}'
    sufijo = 2
    while True:
        try:
            carpeta.mkdir()
            return carpeta
        except FileExistsError:
            carpeta = base / f'{nombre}-{marca}-{sufijo}'
            sufijo += 1


def guardar_config(carpeta, config):
    (Path(carpeta) / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')


def resumen_filas(rows):
    return {
        'filas': len(rows),
        'errores_api': sum(1 for r in rows if r['error']),
        'fin': datetime.now().isoformat(timespec='seconds'),
    }


def hay_predicciones_validas(rows):
    """Si todas las filas fallaron, muestra los errores distintos y devuelve False."""
    if any(r['judge_label'] for r in rows):
        return True
    errores = sorted({r['error'] for r in rows if r['error']})
    print('\nERROR: ninguna fila obtuvo una etiqueta del juez; no se generan informes.')
    for e in errores[:3]:
        print(f'  - {e}')
    return False


def ejecutar_informes(predicciones, muestra, carpeta):
    """Ejecuta evaluate.py y analyze_errors.py escribiendo dentro de la carpeta de la corrida."""
    scripts = Path(__file__).resolve().parent
    raiz = scripts.parent
    carpeta = Path(carpeta).resolve()
    comandos = [
        [sys.executable, str(scripts / 'evaluate.py'), '--input', str(Path(predicciones).resolve()),
         '--output-dir', str(carpeta)],
        [sys.executable, str(scripts / 'analyze_errors.py'), '--predictions', str(Path(predicciones).resolve()),
         '--sample', str(Path(muestra).resolve()), '--output-dir', str(carpeta / cfg.CARPETA_ANALISIS_ERRORES)],
    ]
    for comando in comandos:
        print(f'\nEjecutando informe: {Path(comando[1]).name}', flush=True)
        subprocess.run(comando, cwd=raiz, check=True)
    print(f'\nExperimento e informes completados. Abre {carpeta / cfg.CARPETA_ANALISIS_ERRORES / "report.html"}')
