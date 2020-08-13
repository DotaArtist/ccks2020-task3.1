### 记录尝试过的一些方法与疑惑
- baseline: CRFSUITE
- 词库匹配：利用整理的词库匹配与修正。
- bilstm + word2vec/glove: 预训练词向量模型。
- elmo pretrain + bilstm: 预训练语言模型。
- bert fine-tuning：微调模型。
- albet(todo)：也是微调模型，感觉效果应该和bert类似。
- xlnet：想利用超长文本模型，端到端解决抽取问题。
- LexiconAugmentedNER（todo）：想利用词典信息训练，增强模型学习能力。
