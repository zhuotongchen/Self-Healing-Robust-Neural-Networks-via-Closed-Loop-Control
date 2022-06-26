# Self-Healing-Robust-Neural-Networks-via-Closed-Loop-Control
This repo contains necessary code for the paper **Self-Healing Robust Neural Networks via Closed-Loop Control**
by [Zhuotong Chen](https://scholar.google.com/citations?user=OVs7TPUAAAAJ&hl=en), [Qianxiao Li](https://discovery.nus.edu.sg/9699-qianxiao-li) 
and [Zheng Zhang](https://web.ece.ucsb.edu/~zhengzhang/).

## Description
This proposed closed-loop control method addresses the robustness issue of deep neural network from the perspective of dynamic system and optimal control.
It can be applied to pre-trained deep neural networks (standard or robustly trained) to further improve its performance against various types of perturbations.
The closed-loop control method relies on a set of embedding manifolds that encode the ideal states where the underlying model performs well.
Solving the optimal control problem corresponds to projecting the perturbed state onto this set of embedding manifolds.
