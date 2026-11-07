"""Transparent post-evaluation annotation errata. Re-score ALL methods without changing rankings.
All 90 questions reviewed against the 30-card corpus; no question text or model setting changes.
This is a CODEX audit, NOT an independent human judgment.
"""
import sys,json,csv,hashlib,statistics,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from retrieval import ROOT,read_json,metrics

CORRECTIONS={
 'N023':(['C01','C02'],'显式区分导数计算与参数更新；C02说明两个步骤，C01直接定义更新过程，二者均支持问题。'),
 'N031':(['C01','C03'],'C01与C03都直接说明学习率控制更新步长，不能将C01的正确首位命中判错。'),
 'N153':(['C13','C15'],'显式比较神经向量与字符检索；需要同时计入句向量与词法表示。'),
 'N221':(['C21','C22'],'比较LoRA与QLoRA，与原开发问题Q22a采用相同的双相关项口径。'),
 'N241':(['C24','C26'],'DPO卡片和偏好对质量卡片都明确描述同一提示下的偏好/非偏好回答对。'),
 'N243':(['C26','C29'],'原C24没有直接给出“loss下降不等于事实正确”；C26明确给出该限制，C29定义事实支持的检查。'),
 'N302':(['C21','C30'],'两卡片均直接说明冻结/少量更新不免除基础模型前向计算。'),
}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=ROOT/'results_v2');args=parser.parse_args()
 out=args.results;qs=read_json(ROOT/'data/v2/new_test.json');records=read_json(out/'new_test_per_question.json')
 original_text_hash=hashlib.sha256(json.dumps([(q['id'],q['text']) for q in qs],ensure_ascii=False).encode()).hexdigest()
 events=[]
 for q in qs:
  if q['id'] in CORRECTIONS:
   relevant,reason=CORRECTIONS[q['id']];events.append(dict(query_id=q['id'],old_relevant=q['relevant'],new_relevant=relevant,reason=reason));q['relevant']=relevant
 lookup={q['id']:q for q in qs};rows=[]
 for r in records:
  updated=dict(r);updated['relevant']=lookup[r['query_id']]['relevant'];updated.update(metrics(r['ranked_ids'],updated['relevant']));rows.append(updated)
 original_summary=read_json(out/'new_test_summary.json');summary=[]
 for item in original_summary:
  subset=[r for r in rows if r['method']==item['method']]
  summary.append(dict(item,**{k:statistics.mean(r[k] for r in subset) for k in ['recall@1','recall@3','recall@5','mrr@5']}))
 audit=dict(stage='post-evaluation annotation audit; no subsequent model/configuration tuning',reviewer='CODEX, not independent human',questions_reviewed=90,corrections=len(events),criteria='Relevant iff the card directly defines the requested mechanism/distinction; explicit comparisons may require more than one card. Mere topic association is insufficient.',question_text_sha256=original_text_hash,rankings_unchanged=all(a['ranked_ids']==b['ranked_ids'] for a,b in zip(records,rows)),changes=events,original_results_retained=True)
 for p,obj in [(ROOT/'data/v2/qrels_audit.json',audit),(ROOT/'data/v2/new_test_audited.json',qs),(out/'audited_per_question.json',rows),(out/'audited_summary.json',summary)]:
  p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 with (out/'audited_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 by_type=[]
 for item in summary:
  for kind in ['term','description','contrast']:
   rr=[r for r in rows if r['method']==item['method'] and r['type']==kind]
   by_type.append(dict(method=item['method'],type=kind,n=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in ['recall@1','recall@3','recall@5','mrr@5']}))
 with (out/'audited_by_type.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(by_type[0]));w.writeheader();w.writerows(by_type)
 print(json.dumps({'audit':audit,'summary':summary},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
