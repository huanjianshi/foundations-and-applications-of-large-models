"""Create report comparison panels from real outputs, without altering originals."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'

def panel(items,cols,dest,size=512):
    rows=(len(items)+cols-1)//cols
    sheet=Image.new('RGB',(cols*size,rows*(size+44)), 'white')
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('times.ttf',27)
    for i,(path,title) in enumerate(items):
        x=(i%cols)*size;y=(i//cols)*(size+44)
        im=Image.open(path).convert('RGB')
        sheet.paste(im.resize((size,size),Image.Resampling.LANCZOS),(x,y+44))
        draw.text((x+12,y+6),title,fill='black',font=font)
    sheet.save(dest)

def main():
    rows=list(csv.DictReader((OUT/'results.csv').open(encoding='utf-8-sig')))
    assert [r['run'] for r in rows]==list('ABCDE')
    panel([(OUT/f'demo{res}_{kind}{suffix}.png',f'{kind} {res}')
           for res in [512,768] for kind,suffix in [('clay',''),('depth','_0001'),('normal','_0001')]],
          3,OUT/'control_comparison.png')
    panel([(OUT/'demo512_clay.png','Clay reference')]+[(ROOT/r['image'],
           f"{r['run']}  {r['resolution']} / {r['steps']} / cfg{r['cfg']} / cn{r['control_strength']}")
           for r in rows],3,OUT/'generation_comparison.png')
    panel([(OUT/'A_photoreal.png','Control 0.9'),(OUT/'probe_zero_photoreal.png','Control 0.0'),
           (OUT/'probe_inverse_photoreal.png','Inverted depth')],3,OUT/'control_probe.png')
    panel([(OUT/'demo512_clay.png','Clay'),(OUT/'A_photoreal.png','Q4_0 (A)'),
           (OUT/'probe_f16.png','F16 diagnostic')],3,OUT/'precision_comparison.png')
    ref=np.asarray(Image.open(OUT/'A_photoreal.png').convert('RGB'),dtype=float)
    metrics={}
    for name in ['D','E','probe_zero','probe_inverse']:
        other=np.asarray(Image.open(OUT/f'{name}_photoreal.png').convert('RGB'),dtype=float)
        metrics[name]={'mean_absolute_pixel_difference':float(np.abs(ref-other).mean()),
                       'identical':bool(np.array_equal(ref,other))}
    (OUT/'pixel_differences.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
