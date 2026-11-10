import json, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import config as cfg

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,help='CSV de predicciones (p. ej. results/<corrida>/predictions.csv)')
    p.add_argument('--output-dir',default=None,help='Carpeta de salida (por defecto: la carpeta del CSV de entrada)');a=p.parse_args()
    df=pd.read_csv(a.input).fillna('')
    valid=df[df.judge_label.isin(cfg.ETIQUETAS) & df.human_label.isin(cfg.ETIQUETAS)]
    if valid.empty: raise ValueError('No hay predicciones válidas')
    y=(valid.human_label=='Supported').astype(int); pred=(valid.judge_label=='Supported').astype(int)
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    metrics={'total':len(df),'evaluated':len(valid),'errors':len(df)-len(valid),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'precision_supported':precision,'recall_supported':recall,'f1_supported':f1,'cohen_kappa':cohen_kappa_score(y,pred),'disagreements':int((y!=pred).sum())}
    out=Path(a.output_dir) if a.output_dir else Path(a.input).resolve().parent;out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    valid.loc[y!=pred].to_csv(out/'disagreements.csv',index=False)
    ConfusionMatrixDisplay(confusion_matrix(y,pred,labels=[0,1]),display_labels=['Not Supported','Supported']).plot(cmap='Blues')
    plt.tight_layout();plt.savefig(out/'confusion_matrix.png',dpi=cfg.DPI_GRAFICOS)
    print(json.dumps(metrics,indent=2));print(f'Métricas guardadas en {out}')
if __name__=='__main__': main()
