# Why In-Context Learning Models are Good Few-Shot Learners?
paper: https://openreview.net/pdf?id=iLUcsecZJp

##  Investigating ICL model does learn data-dependent optimaml learning algorithms on single/mixed type of tasks:
* install tools for transformer models from public code (https://github.com/lucidrains/linear-attention-transformer); 
* copy './proving/.' to 'linear-attention-transformer/';
* run `python usim.py`/`python uproto.py`/`python uave.py` for training ICL model on single type of tasks from pair-wise metric-based, class-prototype metric-based, amortization-based respectively; 
* run `python usim_meta.py`/`python uproto_meta.py`/`python uave_meta.py` for training typical meta-learners (MatchNet, ProtoNet and CNPs) on single type of tasks from pair-wise metric-based, class-prototype metric-based, amortization-based respectively; 
* run `python umix.py` for training ICL model and typical meta-learners (MatchNet, ProtoNet and CNPs) on miexed type of tasks;
* run `python test.py` for meta-testing the trained learners and showing the results.

##  Improving ICL through transferring deep-Learning techniques to meta-level:
This part is implemented based on the public code provided with 
**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066; 
* get the public code (https://github.com/dtsip/in-context-learning); 
* copy './improving/conf/.' to 'in-context-learning/src/conf/'; copy './improving/cevel.py' and './improving/meta_train.py' to 'in-context-learning/src/';
* run `python train.py --config conf/lr_curri_0.yaml` for training ICL model; run `python train.py --config conf/lr_curri_dim.yaml` for training ICL model with meta-level curriculum about dimension;
* run `python meta_train.py --config conf/meta.yaml` for training ICL model with meta-level meta-learning;
* run `python ceval.py` for meta-testing the trained learners and showing the results.

@inproceedings{wucontext,
  title={Why In-Context Learning Models are Good Few-Shot Learners?},
  author={Wu, Shiguang and Wang, Yaqing and Yao, Quanming},
  booktitle={The Thirteenth International Conference on Learning Representations}
}