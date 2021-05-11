

[open AI](https://gym.openai.com/envs/CartPole-v0/)

[Torch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## What is CNN

- CNN convolutional -neural network

![Convolutional Neural Networks (CNN) in a Brief - DEV Community](https://res.cloudinary.com/practicaldev/image/fetch/s--w1RZuJPn--/c_imagga_scale,f_auto,fl_progressive,h_420,q_auto,w_1000/https://dev-to-uploads.s3.amazonaws.com/i/1inc9c00m35q12lidqde.png)

  - neural network

  

  ![asdf](https://d2r55xnwy6nx47.cloudfront.net/uploads/2019/01/NeuralNetwok_560_rev.jpg)


  [MNIST](http://yann.lecun.com/exdb/mnist/)

 ## plugin to Q Learning

- Q learning
  $$
  Q^*: State \times Action \rightarrow \mathbb{R}
  $$
  


Reward take from prevoius experience, recent experience make bigger effect.

discount :$\gamma$
$$
R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t
$$


The $\pi$ fuction get best rewards from Learnig exprience

in this state this action get best result
$$
\pi^*(s) = \arg\!\max_a \ Q^*(s, a)
$$
new ```Q Value= Bellman```

$Q^\pi = reward+discount\cdot $ Best act for next state
$$
Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))
$$


temporal difference:

the value need to make from current to best state
$$
\delta = Q(s, a) - (r + \gamma \max_a Q(s_{t+1}, a))
$$




![678cb558a9d59c33ef4810c9618baf34a9577686](README/678cb558a9d59c33ef4810c9618baf34a9577686.svg)
$$
Q^{new}=Q(s_t,a_t)+\alpha\cdot (r_t+\gamma\cdot Q^{\pi} - Q(s_t,a_t))
$$
![Deep Q-Learning | An Introduction To Deep Reinforcement Learning](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-04-16-at-5.46.01-PM.png)

After Q learn from percitular State(image) and relation ship between action and result

i can update itself.