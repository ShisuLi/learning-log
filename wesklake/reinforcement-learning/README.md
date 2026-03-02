# Reinforcement Learning Study Notes

Personal study notes on reinforcement learning, covering both mathematical foundations and deep RL algorithms.

## Course Sources

### 西湖大学·赵世钰（Shiyu Zhao）— Mathematical Foundation of RL

- **Course**: [Mathematical Foundation of Reinforcement Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/tree/main)
- **Instructor**: 赵世钰（Shiyu Zhao），西湖大学
- **Materials**: Slides and textbook PDF in [`slides_shiyuzhao/`](slides_shiyuzhao/) and [`tutorial_shiyuzhao/`](tutorial_shiyuzhao/)

### 清华大学·许华哲（Huazhe Xu）— Deep Reinforcement Learning

- **Course**: Deep Reinforcement Learning
- **Instructor**: 许华哲（Huazhe Xu），清华大学，[http://hxu.rocks/](http://hxu.rocks/)

## Repository Structure

```
reinforcement-learning/
├── notes/                    # Personal study notes
│   ├── RL-note1.md           # Note 1: MDP and Bellman equations
│   └── RL-algorithm.md       # Key formula reference sheet
├── slides_shiyuzhao/         # Lecture slides (Shiyu Zhao)
│   ├── L1-Basic concepts.pdf
│   ├── L2-Bellman equation.pdf
│   ├── L3-Bellman optimality equation.pdf
│   ├── L4-Value iteration and policy iteration.pdf
│   ├── L5-Monte Carlo methods.pdf
│   ├── L6-Stochastic approximation.pdf
│   ├── L7-Temporal-Difference Learning.pdf
│   ├── L8-Value function methods.pdf
│   ├── L9-Policy gradient methods.pdf
│   └── L10-Actor Critic.pdf
└── tutorial_shiyuzhao/       # Textbook PDF (Shiyu Zhao)
    └── mathematical_foundations_of_rl.pdf
```

## Update Log

- **2026-03-02**: Note 1 — MDP and Bellman equations, covering the Lec 1-3 materials from Professor Zhao's course and Lec 1 from Professor Xu's course:
  - MDP five-tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$: state/action spaces, transition probability, reward function, discount factor
  - Markov property, policy $\pi$, value functions $V^\pi$ and $Q^\pi$
  - Bellman Expectation Equation: scalar form, matrix/vector form, analytic solution, iterative solution and convergence proof
  - Bellman Optimality Equation: optimal value functions $V^*$, $Q^*$, Bellman operators $T^\pi$ and $T^*$, contraction mapping theorem
  - Value Iteration and Policy Iteration: algorithms, comparison, and theoretical guarantees
