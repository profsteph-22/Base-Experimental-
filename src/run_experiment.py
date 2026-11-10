"""Ejecuta el juez LLM (OpenAI o Anthropic) sobre la muestra y guarda la corrida en results/.

Uso:
    python src/run_experiment.py                          # proveedor por defecto (config.PROVEEDOR)
    python src/run_experiment.py --proveedor openai
    python src/run_experiment.py --proveedor anthropic --nombre medium

El modelo se define SOLO en .env (OPENAI_MODEL / ANTHROPIC_MODEL); para cambiarlo, editar .env.

Cada proveedor solo define cómo llamar a su API; el bucle, el CSV, la carpeta de la
corrida y los informes son comunes, así ambos jueces usan el mismo prompt y formato.
"""
import sys, json, csv, argparse
from datetime import datetime
from pathlib import Path
import config as cfg
import corridas

SYSTEM = cfg.SYSTEM_PROMPT

# Esquema de salida estructurada (Anthropic): la API garantiza un JSON que cumple este esquema.
OUTPUT_FORMAT = {
    'type': 'json_schema',
    'schema': {
        'type': 'object',
        'properties': {
            'label': {'type': 'string', 'enum': list(cfg.ETIQUETAS)},
            'reason': {'type': 'string'},
        },
        'required': ['label', 'reason'],
        'additionalProperties': False,
    },
}


def juez_openai(modelo):
    """Devuelve (parametros, llamar) para OpenAI; llamar(user) -> texto JSON del modelo."""
    from openai import OpenAI
    client = OpenAI(api_key=cfg.obtener_api_key('openai'))
    temperature = cfg.OPENAI_TEMPERATURE
    parametros = {'temperature': temperature, 'response_format': 'json_object'}

    def llamar(user):
        response = client.chat.completions.create(
            model=modelo, temperature=temperature, response_format={'type': 'json_object'},
            messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}])
        return response.choices[0].message.content

    return parametros, llamar


def juez_anthropic(modelo):
    """Devuelve (parametros, llamar) para Anthropic; llamar(user) -> texto JSON del modelo."""
    import anthropic
    import certifi
    # Usa el bundle de CA de certifi: el almacén de Windows de este equipo no tiene la raíz GTS Root R4
    client = anthropic.Anthropic(api_key=cfg.obtener_api_key('anthropic'),
                                 http_client=anthropic.DefaultHttpxClient(verify=certifi.where()))
    effort, max_tokens = cfg.ANTHROPIC_EFFORT, cfg.ANTHROPIC_MAX_TOKENS
    parametros = {'effort': effort, 'max_tokens': max_tokens, 'output_format': OUTPUT_FORMAT}

    def llamar(user):
        response = client.messages.create(
            model=modelo, max_tokens=max_tokens, system=SYSTEM,
            messages=[{'role': 'user', 'content': user}],
            output_config={'format': OUTPUT_FORMAT, 'effort': effort})
        if response.stop_reason == 'refusal':
            category = response.stop_details.category if response.stop_details else None
            raise ValueError(f'El modelo rechazó la solicitud (categoría: {category})')
        if response.stop_reason == 'max_tokens':
            raise ValueError('Respuesta truncada por max_tokens')
        return next(b.text for b in response.content if b.type == 'text')

    return parametros, llamar


# proveedor -> (fábrica del juez, modelo definido en .env / config.py)
JUECES = {
    'openai': (juez_openai, cfg.OPENAI_MODEL),
    'anthropic': (juez_anthropic, cfg.ANTHROPIC_MODEL),
}


def mensaje_usuario(r):
    context = r['context']
    context_text = ('\n'.join(x.get('text', str(x)) if isinstance(x, dict) else str(x) for x in context)
                    if isinstance(context, list) else str(context))
    return f"Pregunta: {r['query']}\nContexto:\n{context_text}\nAfirmación a evaluar: {r['sentence']}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--proveedor', choices=sorted(JUECES), default=cfg.PROVEEDOR,
                   help=f'Proveedor del juez (por defecto: {cfg.PROVEEDOR}, de config.PROVEEDOR)')
    corridas.agregar_argumentos(p); a = p.parse_args()
    if a.proveedor not in JUECES:
        p.error(f'Proveedor no válido: {a.proveedor!r}. Opciones: {", ".join(sorted(JUECES))}')

    fabrica, model = JUECES[a.proveedor]   # el modelo viene de .env vía config.py
    parametros, llamar = fabrica(model)

    # La carpeta siempre empieza por el modelo real; --nombre solo añade una etiqueta.
    nombre = f'{model}_{a.nombre}' if a.nombre else model
    carpeta = corridas.crear_carpeta(a.resultados, nombre)
    config = {'nombre': nombre, 'proveedor': a.proveedor, 'modelo': model,
              'api_key_env': cfg.API_KEY_ENV[a.proveedor], 'parametros': parametros,
              'entrada': a.input, 'inicio': datetime.now().isoformat(timespec='seconds'), 'system_prompt': SYSTEM}
    corridas.guardar_config(carpeta, config)
    print(f'Proveedor: {a.proveedor} - modelo: {model} (definido en .env)')
    print(f'Carpeta de la corrida: {carpeta}')

    rows = []
    for line in Path(a.input).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        raw = ''
        try:
            raw = llamar(mensaje_usuario(r))
            parsed = json.loads(raw)
            label = parsed.get('label', '')
            if label not in cfg.ETIQUETAS: raise ValueError('Etiqueta no válida')
            reason = parsed.get('reason', ''); error = ''
        except Exception as e:
            label = ''; reason = ''; error = str(e)
        rows.append({'id': r['id'], 'human_label': r['human_label'], 'judge_label': label, 'reason': reason,
                     'raw_output': raw, 'error': error, 'model': model,
                     'fine_grained_factuality': r.get('fine_grained_factuality', '')})
        print(f"{r['id']}: humano={r['human_label']} juez={label or 'ERROR'}")

    output = carpeta / 'predictions.csv'
    with output.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    config.update(corridas.resumen_filas(rows)); corridas.guardar_config(carpeta, config)
    print(f'Resultados guardados en {output}')
    if not corridas.hay_predicciones_validas(rows): sys.exit(1)
    if not a.skip_reports:
        corridas.ejecutar_informes(output, a.input, carpeta)


if __name__ == '__main__':
    main()
