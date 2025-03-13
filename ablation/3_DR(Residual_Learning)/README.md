## Ablation 3: Effectiveness of Deep Residual Learning (PatchLatentMLP).
* Our Proposed Framework Review: Dynamic Factor Decompsition (DF) + Deep Residual Learning (DL)
  * DF: DFM ( t ) $\rightarrow$ Granger Causality + VAR ( t+ $\alpha$ ) $\rightarrow$ Kalman
  * DL: PatchLatentMLP ( t+ $\alpha$ )
* Goal: To demonstrate superior performnace of **DL** phase
  * Compare <span>$\color{#DD6565}DF + PatchLatentMLP$</span> model with <span>$\color{#6580DD}DF + other$</span> models

### <span>$\color{#DD6565}DF + PatchLatentMLP$</span> VS <span>$\color{#6580DD}DF + xPatch,CARD,MLP,LSTM,RNN,VAR,ARIMA$</span>
