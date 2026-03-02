
# 重要公式速查表

## MDP 基础

**折扣累积回报（Return）：**
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}, \quad |G_t| \leq \frac{R_{\max}}{1 - \gamma}$$

---
## 价值函数

**1. 状态价值 (State Value):**
$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$$

**2. 动作价值 (Action Value):**
$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$$

**3. $V^\pi$ 与 $Q^\pi$ 的互相表达：**
$$V^\pi(s) = \sum_a \pi(a \mid s) \cdot Q^\pi(s, a)$$
$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

---
## Bellman 期望方程

**4. 标量形式：**
$$V^\pi(s) = \sum_a \pi(a|s) \underbrace{\sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]}_{Q^\pi(s,a)}$$

**5. 矩阵向量形式：**
$$V^\pi = R^\pi + \gamma P^\pi V^\pi$$

其中：$R^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) R(s,a,s')$，$P^\pi_{ij} = \sum_a \pi(a|s_i) P(s_j|s_i,a)$

**6. 解析解（小规模可用）：**
$$V^\pi = (I - \gamma P^\pi)^{-1} R^\pi$$

**7. 迭代求解（实用）：**
$$V_{k+1} = R^\pi + \gamma P^\pi V_k \quad \xrightarrow{k\to\infty} \quad V^\pi$$

---
## Bellman 最优方程

**8. 最优价值函数定义：**
$$V^*(s) = \max_\pi V^\pi(s), \qquad Q^*(s,a) = \max_\pi Q^\pi(s,a)$$

$$V^*(s) = \max_a Q^*(s, a)$$

**9. 向量符号约定（$|\mathcal{S}|=n$，固定动作 $a$）：**
$$[R_a]_i = \sum_{s'} P(s' \mid s_i, a)\,R(s_i, a, s'), \qquad [P_a]_{ij} = P(s_j \mid s_i, a)$$

**10. 最优 $V$ 的 Bellman 方程（标量 / 向量）：**
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

$$\boxed{V^* = \max_a \left( R_a + \gamma P_a V^* \right)}$$

**11. 最优 $Q$ 的 Bellman 方程：**
$$Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s', a') \right]$$

**12. 最优策略提取：**
$$\pi^*(s) = \arg\max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

---
## Bellman 算子

**13. 期望算子 $T^\pi$（不动点为 $V^\pi$）：**
$$T^\pi V = R^\pi + \gamma P^\pi V, \qquad V^\pi = T^\pi V^\pi$$

**14. 最优算子 $T^*$（不动点为 $V^*$）：**
$$T^* V = \max_a \left( R_a + \gamma P_a V \right), \qquad V^* = T^* V^*$$

**压缩系数：** $\|T^* V_1 - T^* V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$

---
## 价值迭代（Value Iteration）

**15. 迭代更新：**
$$V_{k+1} = \max_a \left( R_a + \gamma P_a V_k \right) \quad \xrightarrow{\|V_{k+1}-V_k\|_\infty < \varepsilon} \quad V^*$$

**16. 策略提取（收敛后执行一次）：**
$$\pi^*(s) = \arg\max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

---
## 策略迭代（Policy Iteration）

**17. 策略评估（解线性方程组）：**
$$V^{\pi_k} = \left(I - \gamma P_{a_k}\right)^{-1} R_{a_k}, \quad a_k(s) = \pi_k(s)$$

**18. 策略改进（贪婪更新）：**
$$\pi_{k+1}(s) = \arg\max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^{\pi_k}(s') \right]$$

**策略改进定理：** $V^{\pi_{k+1}}(s) \geq V^{\pi_k}(s),\ \forall s \in \mathcal{S}$

**收敛条件：** $\pi_{k+1} = \pi_k \Rightarrow \pi_k = \pi^*$
