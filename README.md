Prototype of sentiment classifiers

#### conditions
- d_emb: 256
- d_hiddem: 128
- epoch: 100
- max_seq_len: 50
- vocab_size: 100000+2

#### max validation score
- MLP ... accuracy: --- f_score: ---
- BiLSTM(1 layer) ... accuracy: 0.937 f_score: 0.936
- BiLSTM(1 layer)+self attention ... accuracy: 0.939 f_score: 0.938
- CNN ... accuracy: 0.929 f_score: 0.928
- Transformer(6 layers) ... accuracy: 0.935 f_score: 0.934
