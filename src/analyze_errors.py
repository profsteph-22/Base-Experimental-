"""Analiza desacuerdos entre un juez LLM y las anotaciones de MEMERAG.

No llama a ningún modelo ni modifica los resultados originales.
"""
import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
import config as cfg


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def load_sample(path):
    if not path.exists():
        return {}, False
    result = {}
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        key = str(item.get('id', ''))
        if not key:
            raise ValueError(f'Muestra sin id en línea {line_number}')
        if key in result:
            raise ValueError(f'ID duplicado en muestra: {key}')
        result[key] = item
    return result, True


def readable_context(context):
    if isinstance(context, str):
        return context
    if isinstance(context, list):
        return '\n\n'.join(
            x.get('text', json.dumps(x, ensure_ascii=False)) if isinstance(x, dict) else str(x)
            for x in context
        )
    return json.dumps(context, ensure_ascii=False, indent=2) if context is not None else ''


def error_type(human, judge):
    if human == 'Not Supported' and judge == 'Supported':
        return 'Aceptación incorrecta (falso Supported)'
    if human == 'Supported' and judge == 'Not Supported':
        return 'Rechazo incorrecto (falso Not Supported)'
    return 'Otro desacuerdo'


def write_csv(path, rows):
    columns = ['id', 'human_label', 'judge_label', 'error_type',
               'fine_grained_factuality', 'query', 'sentence', 'context',
               'reason', 'model', 'source_found']
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_html(path, rows, counts, missing):
    esc = lambda value: html.escape(str(value or ''))
    cards = []
    for i, row in enumerate(rows, 1):
        sections = [
            ('Pregunta', row['query']),
            ('Afirmación evaluada', row['sentence']),
            ('Contexto recuperado', row['context']),
            ('Justificación del juez', row['reason']),
        ]
        content = ''.join(f'<h3>{esc(title)}</h3><pre>{esc(value) or "No disponible"}</pre>' for title, value in sections)
        cards.append(
            f'<article><h2>Caso {i}: {esc(row["id"])}</h2>'
            f'<p><strong>Humano:</strong> {esc(row["human_label"])} · '
            f'<strong>Juez:</strong> {esc(row["judge_label"])} · '
            f'<strong>Tipo:</strong> {esc(row["error_type"])} · '
            f'<strong>Categoría MEMERAG:</strong> {esc(row["fine_grained_factuality"])}</p>'
            + content + '</article>'
        )
    count_html = ''.join(f'<li>{esc(name)}: {number}</li>' for name, number in counts.most_common())
    warning = (f'<p class="warning">Faltan los datos originales de {missing} casos. '
               f'Coloca {cfg.ARCHIVO_MUESTRA} para incluir afirmaciones y contextos.</p>') if missing else ''
    page = f'''<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>MEMERAG — Análisis de desacuerdos</title>
<style>body{{font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:40px auto;padding:0 18px;line-height:1.5;background:#f6f8fb;color:#182332}}article{{background:white;border:1px solid #dbe2eb;border-radius:12px;padding:22px;margin:24px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f8;padding:14px;border-radius:8px;font-family:inherit}}h1,h2{{color:#183b68}}h3{{margin-bottom:6px}}.warning{{background:#fff1dc;padding:14px;border-radius:8px}}</style>
</head><body><h1>MEMERAG: análisis de desacuerdos</h1>
<p>Casos discrepantes: <strong>{len(rows)}</strong>. Categorías de MEMERAG (no causas verificadas):</p><ul>{count_html}</ul>
{warning}{''.join(cards)}<p>Las anotaciones humanas se usan como referencia; un desacuerdo no demuestra por sí solo un error del juez.</p></body></html>'''
    path.write_text(page, encoding='utf-8')


def write_chart(path, counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels, values = zip(*counts.most_common())
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(labels) * 0.8)))
    bars = ax.barh(labels, values)
    ax.invert_yaxis()
    ax.bar_label(bars, padding=3)
    ax.set_xlim(0, max(values) + 1)
    ax.set_xlabel('Número de desacuerdos')
    ax.set_title('MEMERAG: desacuerdos por categoría de anotación')
    fig.tight_layout()
    fig.savefig(path, dpi=cfg.DPI_GRAFICOS)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True,
                        help='CSV de predicciones (p. ej. results/<corrida>/predictions.csv)')
    parser.add_argument('--sample', type=Path, default=Path(cfg.ARCHIVO_MUESTRA))
    parser.add_argument('--output-dir', type=Path, default=None,
                        help=f'Por defecto: <carpeta del CSV>/{cfg.CARPETA_ANALISIS_ERRORES}')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.predictions.resolve().parent / cfg.CARPETA_ANALISIS_ERRORES
    if not args.predictions.exists():
        parser.error(f'No existe {args.predictions}. Ejecuta primero run_experiment.py o indica --predictions.')
    predictions = load_csv(args.predictions)
    sample, sample_exists = load_sample(args.sample)
    rows = []
    seen = set()
    for pred in predictions:
        key = str(pred.get('id', ''))
        if not key or key in seen:
            raise ValueError(f'ID vacío o duplicado en predicciones: {key!r}')
        seen.add(key)
        human, judge = pred.get('human_label', ''), pred.get('judge_label', '')
        if human not in cfg.ETIQUETAS or judge not in cfg.ETIQUETAS:
            continue  # Excluir respuestas inválidas y errores de API.
        if human == judge:
            continue
        source = sample.get(key, {})
        rows.append({
            'id': key, 'human_label': human, 'judge_label': judge,
            'error_type': error_type(human, judge),
            'fine_grained_factuality': pred.get('fine_grained_factuality', ''),
            'query': source.get('query', ''), 'sentence': source.get('sentence', ''),
            'context': readable_context(source.get('context')),
            'reason': pred.get('reason', ''), 'model': pred.get('model', ''),
            'source_found': bool(source),
        })
    counts = Counter(row['fine_grained_factuality'] or 'Sin categoría' for row in rows)
    missing = sum(not row['source_found'] for row in rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'disagreements_detailed.csv', rows)
    write_html(args.output_dir / 'report.html', rows, counts, missing)
    summary = {
        'total_predictions': len(predictions), 'disagreements': len(rows),
        'incorrect_acceptances': sum(r['human_label'] == 'Not Supported' for r in rows),
        'incorrect_rejections': sum(r['human_label'] == 'Supported' for r in rows),
        'categories': dict(counts), 'sample_file_found': sample_exists,
        'cases_without_original_context': missing,
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if counts:
        try:
            write_chart(args.output_dir / 'categories.png', counts)
        except ImportError:
            print('Aviso: instala matplotlib para generar categories.png')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Informe: {args.output_dir / "report.html"}')
    if missing:
        print(f'AVISO: faltan {missing} contextos/afirmaciones. Usa --sample con el sample.jsonl original.')


if __name__ == '__main__':
    main()
