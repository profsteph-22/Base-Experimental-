import argparse, json, random
from pathlib import Path
import config as cfg

URL = cfg.MEMERAG_URL  # referencia; el archivo se descarga manualmente

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--file', type=Path, required=True, help='JSONL español descargado del repositorio oficial MEMERAG')
    p.add_argument('--n', type=int, default=cfg.TAMANO_MUESTRA)
    p.add_argument('--seed', type=int, default=cfg.SEMILLA)
    a=p.parse_args()
    rows=[]
    for line in a.file.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        obj=json.loads(line)
        for ans in obj.get('answer', []):
            label=ans.get('factuality')
            if label not in cfg.ETIQUETAS: continue
            rows.append({'id':f"{obj.get('query_id')}_{ans.get('sentence_id')}", 'query':obj.get('query',''), 'context':obj.get('context',[]), 'sentence':ans.get('sentence',''), 'human_label':label, 'fine_grained_factuality':ans.get('fine_grained_factuality','')})
    if not rows: raise ValueError('No se encontraron registros compatibles: verificar esquema y dataset')
    random.Random(a.seed).shuffle(rows)
    sample=rows[:a.n]
    dest=Path(cfg.ARCHIVO_MUESTRA); dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sample),encoding='utf-8')
    print(f'Muestra: {len(sample)} / {len(rows)}; archivo: {dest}')
if __name__=='__main__': main()
