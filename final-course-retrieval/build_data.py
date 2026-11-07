"""Create original educational summaries and an authored, non-public-benchmark query set.
Run before evaluation only. Existing data cannot be silently overwritten.
"""
import json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
 'DL': 'https://www.deeplearningbook.org/contents/optimization.html',
 'REG': 'https://www.deeplearningbook.org/contents/regularization.html',
 'ATT': 'https://arxiv.org/abs/1706.03762',
 'GEN': 'https://huggingface.co/docs/transformers/main_classes/text_generation',
 'EMB': 'https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
 'IR': 'https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html',
 'TFIDF': 'https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting',
 'RRF': 'https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf',
 'LORA': 'https://arxiv.org/abs/2106.09685',
 'QLORA': 'https://arxiv.org/abs/2305.14314',
 'DPO': 'https://arxiv.org/abs/2305.18290',
 'RAG': 'https://arxiv.org/abs/2005.11401',
 'IOU': 'https://arxiv.org/abs/1902.09630',
 'GRAPH': 'https://arxiv.org/abs/2404.16130',
 'SELF': 'https://arxiv.org/abs/2203.14465',
}
# title, original summary, source key, direct query, paraphrase/confusion query
ROWS = [
('梯度下降','梯度下降沿损失对参数的负梯度方向更新权重。学习率控制更新步长；目标是降低训练目标，而非保证每个样本的损失都下降。','DL','梯度下降怎样更新参数？','知道误差后，网络应该朝哪个方向调整权重？'),
('反向传播','反向传播通过链式法则计算损失对各层参数的导数，再交给优化器更新。求梯度与参数更新是两个步骤，计算图连接二者。','DL','反向传播的作用是什么？','误差信息如何从输出端传回前面的神经网络层？'),
('学习率','学习率决定参数每次移动的步长。过大可能越过较优区域或造成数值不稳定，过小可能收敛缓慢。损失波动本身不等于发散。','DL','学习率太大和太小分别有什么影响？','训练走得很慢，或者每次调整都跳过合适位置，应该检查哪个超参数？'),
('过拟合','过拟合表现为训练表现较好而未见数据上的表现变差。模型可能记住样本细节或噪声；降低训练损失不能单独证明泛化改善。','REG','什么是过拟合？','为什么练过的题都能答对，一换新题就频繁出错？'),
('Dropout正则化','Dropout在训练期间随机屏蔽部分单元，减轻对特定特征组合的依赖。推理阶段关闭随机屏蔽；应切换模型的训练和评估模式。','REG','Dropout在训练和推理时有何区别？','每次测试同一输入结果都在随机变化，哪种正则层的模式可能没切对？'),
('独立测试集','开发数据用于调试和选择设置，独立测试数据用于最终评估。若反复依据测试成绩修改模型，测试集就参与了选择，评估会过于乐观。','REG','开发集与独立测试集分别用来做什么？','能不能看完最终考试答案，再改系统并把新的分数当成独立测评？'),
('位置编码','自注意力需要额外信息区分输入顺序。位置编码将位置信息加入表示，帮助模型识别相同词语在不同排列中的作用。','ATT','Transformer为什么需要位置编码？','句子包含相同的词，调换先后次序却意思不同，模型如何获得顺序线索？'),
('自注意力','自注意力由输入计算查询Q、键K和值V，按Q与K的匹配权重聚合V。不同位置之间可以交换信息，形成依赖上下文的表示。','ATT','自注意力中的Q、K、V有什么作用？','一个词怎样从句子里其他位置挑出需要的信息并整合进自身表示？'),
('因果注意力掩码','自回归语言模型的因果掩码阻止当前位置读取未来token。这样训练时的可见信息与逐步生成一致，避免利用尚未生成的答案。','ATT','因果注意力掩码屏蔽什么？','训练续写模型时，怎样阻止它偷看后面还没写出的词？'),
('Token与分词','分词器把文本映射为token编号，token可能是字、词或子词，并不固定等于一个汉字。截断上限按token数计算，会包含特殊符号的开销。','GEN','Token数是否等于汉字数？','输入限制是一百多个单位，为什么不能直接按中文字符数判断是否超长？'),
('温度采样','温度在采样前缩放logits。较低温度通常使概率更集中，较高温度通常增加随机性；它改变生成分布，不更新模型权重。','GEN','温度参数怎样影响文本采样？','想让同一个模型写得更随机一些，需要重新训练还是调整生成分布？'),
('Top-k采样','Top-k采样先保留概率最高的k个候选token，再在其中重新归一化并抽样。这里的k控制生成候选集合，与检索返回条数不是同一操作。','GEN','Top-k采样怎样筛选候选token？','只从最可能的几个后续词中抽一个，这属于哪种生成策略？'),
('句向量嵌入','句向量把文本映射到定长稠密空间，用于相似度比较。当前MiniLM模型输出384维；注意力掩码均值池化避免把padding当作有效词平均。','EMB','句向量是什么，均值池化为何需要掩码？','如何把长短不同的句子变成等长数字列表，以便比较它们的意思？'),
('余弦相似度','余弦相似度比较两个非零向量的方向，等于内积除以两者模长的乘积。L2归一化后可直接用内积计算，但相似度不是答案正确概率。','EMB','归一化向量怎样计算余弦相似度？','两个文本向量的数值大小不同，怎样主要比较它们的方向接近程度？'),
('字符TF-IDF检索','字符二元组TF-IDF按相邻字符片段建立稀疏表示，词频乘以逆文档频率后归一化。它依赖词面重合，对不同说法的同义问题可能漏检。','TFIDF','字符TF-IDF检索为什么可能漏掉同义表达？','不用神经模型，只凭相邻文字片段的出现次数和稀有程度，能怎样搜索资料？'),
('RRF排名融合','RRF将不同检索器的排名转为1除以常数加名次，再相加排序。它融合名次而非直接混加不同量纲分数；本文预先固定常数60。','RRF','RRF为什么可以融合不同量纲的检索结果？','两套搜索系统打分尺度不同，怎样只看名次把列表合成一个？'),
('Recall召回率','Recall@K等于前K项中命中的相关资料数除以全部标注相关资料数。若一道题有两个相关文档只找回一个，召回率是二分之一而不是一。','IR','有两个相关文档只找到一个，Recall是多少？','只要找到一条答案就算全对，这和找全所有相关资料的指标一样吗？'),
('MRR倒数排名','MRR@K对每题计算前K项中第一个相关结果名次的倒数，再求平均。未命中记零；首个正确结果越靠前，得分越高。','IR','MRR@K如何计算？','两次搜索都找到答案，但一个排第一一个排第三，什么指标能区分这种差别？'),
('检索增强生成','RAG先从外部资料检索证据，再由生成模型结合证据组织回答。检索与回答生成是不同环节；本项目只实现检索，不把原文展示叫作模型生成。','RAG','检索增强生成包括哪些环节？','把查到的资料交给语言模型再写回答，与只返回原文片段有什么区别？'),
('图结构检索','图结构检索利用实体、关系和社区等结构组织信息。构图质量会影响检索，预置关系或摘要必须能追溯到来源，不能当作自动抽取结果。','GRAPH','图结构检索依赖什么信息？','除了比较两段文字的意思，能否沿人物和事物之间的联系查找材料？'),
('LoRA低秩适配','LoRA冻结基础权重，用两个低秩矩阵的乘积表示增量更新。训练参数量减少，但基础模型仍参与前向计算，不代表整模型参数消失。','LORA','LoRA为什么减少可训练参数？','想少训练一些权重，能否冻结原模型，只学习一个低秩的增量？'),
('量化与QLoRA','量化用较低精度表示基础模型以节省内存。QLoRA结合量化基座与低秩适配；只用了LoRA而没量化时，不能称为实际四比特QLoRA。','QLORA','LoRA和四比特QLoRA有什么区别？','只训练了适配器但基座仍是全精度，是否就已经做了四比特训练？'),
('监督微调SFT','监督微调用提示与目标回答训练模型。只监督回答时，将提示位置标签设为-100，使其不参与损失；答案结束符的监督也需明确。','DPO','SFT为什么把提示位置标签设为-100？','训练问答模型时，怎样只要求它学会回答，而不让它重复学习输入问题？'),
('DPO偏好优化','DPO利用同一提示下偏好与非偏好回答的概率对比训练模型，并用参考策略约束变化。偏好数据质量和训练预算会影响实际生成表现。','DPO','DPO用什么形式的数据进行训练？','没有逐词标准答案，却知道两个回答中更喜欢哪一个，可以如何改进模型？'),
('DPO参考策略','DPO参考策略提供固定概率基准，不进行梯度更新。LoRA场景可用关闭适配器的基座表示参考策略，但必须确认它确实保持冻结。','DPO','DPO参考模型为什么不更新？','偏好训练中用来衡量变化幅度的原始策略，应该跟着当前模型一起学吗？'),
('偏好对质量','偏好对包含同一提示下的chosen与rejected。顺序颠倒或把错误回答标为优选会给出错误学习信号；降低偏好损失不等于事实正确。','DPO','chosen与rejected标反会怎样？','如果训练资料把胡说的回答标成更好，优化过程会不会反而奖励错误？'),
('交并比IoU','IoU等于预测区域和真实区域交集面积除以并集面积，常用于检测定位。类别答对不保证位置正确；预测不合法时需按固定规则处理。','IOU','IoU如何衡量定位结果？','模型认出了是什么物体，但框的位置偏离了，该怎样量化重叠程度？'),
('自举反馈与验证','自举式改进用模型产生并筛选的新样本继续训练。若验证器会误接收错误，错误可循环放大；样本池变大不直接证明能力提高。','SELF','自举训练为什么需要可靠的验证器？','让模型拿自己写的答案继续学习，如果没人正确把关，会产生什么问题？'),
('事实忠实度','事实忠实度关注回答是否被资料支持。关键词出现不等于正确，回声、否定和矛盾回答都需检查；应区分字面匹配与实质判断。','RAG','关键词匹配为何不等于回答事实正确？','回答复述了问题中的词，却没有给出资料支持的结论，能算答对吗？'),
('参数量与推理耗时','总参数量、可训练参数量和推理耗时是不同概念。冻结权重不免除计算；计时应区分加载、建索引和单次查询，明确设备与测量范围。','LORA','可训练参数少是否保证推理更快？','只改了很少的模型权重，为什么运行时仍然可能要花不少时间？'),
]

def main():
    data = ROOT/'data'
    if (data/'freeze.json').exists():
        raise SystemExit('Dataset is frozen; do not overwrite. Create an explicitly versioned dataset instead.')
    cards=[]; questions=[]
    for i,(title,text,src,direct,other) in enumerate(ROWS,1):
        cid=f'C{i:02d}'
        cards.append(dict(id=cid,title=title,text=text,source=SOURCES[src],provenance='CODEX原创教学概括，来源为相关技术依据，非原文摘录',human_reviewed=False))
        for kind,q in [('direct',direct),('paraphrase' if i%3 else 'confusion',other)]:
            split='dev' if kind=='direct' and i<=20 else 'test'
            # Questions explicitly asking for a comparison require both supporting concepts.
            relevant=[cid]
            if i==22 and kind=='direct': relevant=['C21','C22']
            if i==6 and kind=='direct': relevant=['C06']
            questions.append(dict(id=f'Q{i:02d}{"a" if kind=="direct" else "b"}',text=q,type=kind,split=split,relevant=relevant,rationale=f'问题针对「{title}」；相关卡片直接定义其机制或区别。',human_reviewed=False))
    data.mkdir(exist_ok=True)
    for name,obj in [('cards.json',cards),('questions.json',questions),('sources.json',SOURCES)]:
        (data/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest={n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in ['cards.json','questions.json','sources.json']}
    (data/'freeze.json').write_text(json.dumps(dict(version='1.1',frozen_before_evaluation=True,cards=30,dev=20,test=40,hashes=manifest,author='CODEX',human_review_status='pending; all labels are provisional until user review',selection='Fixed design, no model scores used to author or select queries.'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Frozen 30 cards, 20 dev queries, 40 test queries.')

if __name__=='__main__': main()
