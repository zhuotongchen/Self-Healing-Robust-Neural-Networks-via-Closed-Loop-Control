# Self-Healing-Robust-Neural-Networks-via-Closed-Loop-Control
This repo contains necessary code for the paper **Self-Healing Robust Neural Networks via Closed-Loop Control**
by [Zhuotong Chen](https://scholar.google.com/citations?user=OVs7TPUAAAAJ&hl=en), [Qianxiao Li](https://discovery.nus.edu.sg/9699-qianxiao-li) 
and [Zheng Zhang](https://web.ece.ucsb.edu/~zhengzhang/).

## Description
This proposed closed-loop control method addresses the robustness issue of deep neural network from the perspective of dynamic system and optimal control.
It can be applied to pre-trained deep neural networks (standard or robustly trained) to further improve its performance against various types of perturbations.
The closed-loop control method relies on a set of embedding manifolds that encode the ideal states where the underlying model performs well.
Solving the optimal control problem corresponds to projecting the perturbed state onto this set of embedding manifolds.

### Algorithm demonstration
The following figure demonstrates the working principle of the closed-loop control algorithm.
As shown in Fig (a), while the binary classifier has 100 % accuracy on clean data, adversarial attack leads to 0 % accuracy.
Fig (b) shows the reconstruction loss field where clean samples are located in the low loss recion and adversarial examples fall out of the embedding manifold.
The control process adjusts adversarial examples towards the embedding manifold, and the classifier predicts those with 100 % accuracy (Fig (c)).
Essentially, this process forms a new decision boundary and enlarges the classification margin as shown in Fig (d).

![alt text](https://github.com/zhuotongchen/Self-Healing-Robust-Neural-Networks-via-Closed-Loop-Control/blob/master/assets/demonstration.png)
