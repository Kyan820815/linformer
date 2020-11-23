# CSCI-1470-Final-Project
Linformer: Self-Attention with Linear Complexity

##### Install
```
	git clone git@github.com:Kyan820815/CSCI1470-Final-Project.git
	cd CSCI1470-Final-Project
```


##### Variable Explanation
* num_tokens: size of vocab
* input_size: window_size for encoder & decoder
* channels: size of a word vector (before embedding)
* dim_d: inner size of head computation, like W_q: d_m x d_k, where d_k is dim_d and d_m is embedding size
* dim_k: main idea of paper
* dim_ff: size of ff layer
* nhead: number of head
* depth: how many encoder or decoder of one pass for transformer
* emb_dim: embedding size
* parameter_sharing: what level of parameter sharing to use.
 


