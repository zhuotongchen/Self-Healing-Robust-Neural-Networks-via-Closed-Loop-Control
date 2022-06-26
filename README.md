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

## Description of each file
### train_models.py
This contains implementation of training baseline models (standard SGD and [adversarial training](https://github.com/yaodongyu/TRADES)).
Selecting --train_method to be either standard or robust to choose the training method.

### control_functions.py
This contains functions to generate adversarial examples via [Autoattack](https://github.com/fra31/auto-attack),
test the performence of classifiers against those adversarial examples.
Train embedding function at input layer and hidden layers.

### evaluate_models.py
This is the main script for testing the closed-loop control algorithm.

By selecting --generate_adversarial_dataset,
it generates adversarial examples of pre-trained baseline models with autoattack.

By selecting --train_embedding_function_input,
it trains an embedding function at input layer (default autoencoder is [FCN](https://arxiv.org/pdf/1411.4038.pdf), other option includes [SegNet](https://arxiv.org/pdf/1511.00561.pdf)).

By selecting --train_embedding_function_hidden,
it trained an embedding function at hidden layer (default autoencoder is a 2-layer convolutional autoencoder).

By selecting --test_models,
it tests closed-loop controlled baseline module.

### control_module.py
This is the implementaion of closed-loop control algorithm.
It converts a given baseline model into a controlled module.
Default hyper-parameters are: maximum outer iterations is 3, maximum inner iterations is 10, control regularization is 0.001.

## Numerical result (cifar-10)
![alt text](https://github.com/zhuotongchen/Self-Healing-Robust-Neural-Networks-via-Closed-Loop-Control/blob/master/assets/result.png)
