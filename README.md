# 130 项目

## Resources

[openAI Mario Environment](https://pypi.org/project/gym-super-mario-bros/)
[DeepLearning FlappyBird DQN](https://github.com/yenchenlin/DeepLearningFlappyBird)
[RL tutorial](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)
[Mario DQN](https://github.com/aleju/mario-ai)

# Process
- [ ] 如何部署在 Amazon/Google 上


- [ ] 什么是 Q learning
- [ ] 如何让 Mario 看到图
- [ ] 回馈
- [ ] 哈?


# 算法
[Q learning Wikipedia](https://en.wikipedia.org/wiki/Q-learning)

## epsilon-Greedy

贪婪算法的一种

单纯的贪婪算法只能出一两个最佳选择, (局部最优) 不适用于有动态敌人的 N 步游戏



```e Greedy```

按一定概率去选择局部最优还是尝试另外的方法



```保留, 可遇到所有敌人都避过去```



### Q table 

---

更新逻辑

![678cb558a9d59c33ef4810c9618baf34a9577686](README/678cb558a9d59c33ef4810c9618baf34a9577686.svg)

