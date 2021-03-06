# RL: Vanilla REINFORCE algorithms
<img src="https://github.com/coldhenry/RL-REINFORCE-Pytorch/blob/main/pic/openai-pytorch.jpg" weight="300" height="100">

## REINFORCE algorithms

REINFORCE algorithm is the most basic policy grdient method that applies likelihood ratio policy gradient to learn a suitable policy. 

### pseudocode[1]

<img src="https://github.com/coldhenry/RL-REINFORCE-Pytorch/blob/main/pic/pseudo.png" weight="638" height="231">

However, in my implementation, the policy gradient has combined with a baseline to increase stability. It is modified as followed:

<img src="https://github.com/coldhenry/RL-REINFORCE-Pytorch/blob/main/pic/baseline.jpg" weight="626" height="100">

## Environment and Results
* Discrete Action space : CartPole-v0
* Continuous Action space: 2-link arm

<img src="https://github.com/coldhenry/RL-REINFORCE-Pytorch/blob/main/pic/demo.png" weight="561" height="560">


## Reference 

[1] [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
