# 强化学习（Reinforcement Learning, RL）- Note 1：MDP与Bellman方程
强化学习的核心思想是：**智能体（Agent）** 在 **环境（Environment）** 中通过 **尝试与错误（Trial and Error）** 来学习，目标是最大化累积 **奖励（Reward）**。
- RL基本交互框架：Agent 感知环境状态 $s$，执行动作 $a$，环境返回奖励 $r$ 并转移到下一个状态 $s'$，循环往复。
### MDP：数学形式化框架
- 强化学习的过程可以抽象为 **马尔可夫决策过程 (Markov Decision Process, MDP)**，用五元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ 精确描述：

|      符号       |  名称  | 角色             |
| :-----------: | :--: | :------------- |
| $\mathcal{S}$ | 状态空间 | 环境所有可能状态的集合    |
| $\mathcal{A}$ | 动作空间 | 智能体所有可能动作的集合   |
|      $P$      | 转移概率 | 环境动态，即"世界的规则"  |
|      $R$      | 奖励函数 | 环境对动作的即时反馈信号   |
|   $\gamma$    | 折扣因子 | 平衡即时奖励与长期奖励的权重 |

- **核心假设——马尔可夫性（Markov Property）：** 转移概率只依赖于**当前状态**和**当前动作**，与历史轨迹无关：
$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$
- 在五元组基础上，引入两个**派生概念**（用于分析和求解MDP，不是MDP本身的组成部分）：
	- **策略 (Policy, $\pi$):** 智能体在每个状态如何选择动作的规则，详见[策略节](#策略-π-policy)。
	- **价值函数 (Value Function, $V/Q$):** 对未来长期累积回报的预测，是RL算法的核心优化目标，详见[价值函数节](#价值函数-value-function)。

#### 状态空间 $\mathcal{S}$（State Space）
- 状态 $s \in \mathcal{S}$ 是Agent在某一时刻感知到的**完整环境描述**，是MDP做决策的信息基础。
- 状态必须满足马尔可夫性：$s_t$ 包含了预测未来所需的**全部**信息，不需要知道历史 $s_0, s_1, \ldots, s_{t-1}$。
- 状态空间可以是**离散的**（如棋盘的有限局面数）或**连续的**（如机器人关节角度的实数值）。
#### 动作空间 $\mathcal{A}$（Action Space）
- 动作 $a \in \mathcal{A}$ 是Agent在某状态下可以执行的选择。**本笔记中 $a \in \mathcal{A}$ 专指动作**，不与其他含义混用。
- 动作空间分为两大类，这个区别直接决定了用什么算法：
	- **离散动作空间**：有限个选项，如"上下左右"4个方向、游戏按键 → 适合 **DQN**。
	- **连续动作空间**：实数值，如机器人关节力矩 $\tau \in [-1, 1]^n$、连续控制信号 → 适合 **SAC / PPO**。
#### 转移概率 $P$（Transition Probability）
$$P(s' \mid s, a)$$
描述在状态 $s$ 执行动作 $a$ 后，环境转移到状态 $s'$ 的概率。$P$ 完全刻画了**环境的动态规则**，Agent无法控制，只能适应。
- $P$ 满足归一化条件：$\sum_{s' \in \mathcal{S}} P(s' \mid s, a) = 1, \quad \forall s \in \mathcal{S},\ a \in \mathcal{A}$
- 若 $P$ 已知，称为**有模型（Model-Based）** 设定；若 $P$ 未知，需从交互中学习，称为**无模型（Model-Free）** 设定（Q-Learning、PPO等均属此类）。
#### 奖励函数 $R$（Reward Function）
$$R(s, a, s')$$
Agent执行动作后从环境获得的即时反馈标量，可正（鼓励）可负（惩罚）。
- 奖励函数是**设计者向Agent传递目标**的唯一渠道。奖励设计（Reward Shaping）至关重要——设计不当会导致Agent找到人类意图之外的"捷径"（即奖励欺骗，Reward Hacking）。
- 实践中有时简化为 $R(s, a)$（不依赖 $s'$），两者在期望意义下等价：$R(s,a) = \sum_{s'} P(s'\mid s,a)\,R(s,a,s')$。
#### 折扣因子 $\gamma$（Discount Factor）
$$\gamma \in [0, 1)$$
- $\gamma$ 接近0，Agent极度短视，只在乎眼前奖励；$\gamma$ 接近1，Agent高度有远见，远期奖励的衰减极慢，几乎与近期奖励同等权重。
- 加入折扣后，Agent的目标是最大化**折扣累积回报（Return）**：

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

$$\gamma \to 0 \quad \Rightarrow \quad \text{极度短视，只在乎下一步奖励}$$

$$\gamma \to 1 \quad \Rightarrow \quad \text{极度有远见，远期奖励衰减极慢，权重趋近于近期奖励}$$

- $\gamma = 0.99$ 是最常见的选择。在这个设定下，100步后的奖励只值现在的 $0.99^{100} \approx 0.37$ 倍——有价值但权重递减。
##### 理解折扣因子
- 如果没有折扣因子，Return就是
$$G_t = R_{t+1} + R_{t+2} + R_{t+3} + \cdots$$
- 如果每步奖励都是正数，这个求和会**发散到无穷大**——无法比较两个策略哪个更好，因为两者的Return都是 $+\infty$。
- 加入 $\gamma \in [0,1)$ 后即为 **Infinite Horizon with Discounting**（无限步折扣求和）
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$
- 由于奖励有界（假设 $|R_t| \leq R_{\max}$），这是一个等比级数，上界为：
$$|G_t| \leq \frac{R_{\max}}{1 - \gamma}$$
- $\gamma < 1$ **充分保证**了无限求和收敛（有界），且最优策略是 **平稳的**（stationary）——同样的状态永远对应同样的最优动作，跟时间步无关。这让理论分析和算法设计都优雅得多。
- 除此之外还有不同可选方案：
	- **Finite Horizon**（有限步求和，不折扣）：
$$U = R_1 + R_2 + \cdots + R_H$$
	- 只看固定的 $H$ 步，每步奖励权重相等。适合有明确终止时间的任务（比如下棋），但问题是最优策略会依赖于当前是第几步——同样的状态，第5步和第95步的最优行动可能不同，这让分析变得复杂。
	- **Average Reward**（平均奖励）：
$$U = \lim_{T\to\infty} \frac{1}{T} \sum_{t=1}^{T} R_t$$
	- 直接优化长期平均奖励，适合持续运行、没有自然终止点的任务（比如服务器调度）。不需要折扣因子，代价是算法更复杂，需要额外估计平均奖励。
#### 策略 $\pi$ (Policy)
- 在MDP框架下，Agent的行为由**策略 $\pi$（Policy）** 决定：
$$\pi(a \mid s) = P(\text{选择动作} \ a \mid \text{当前状态} \ s)$$
- 策略是一个从状态到动作的映射，有两种形式：
	- **确定性策略（Deterministic Policy）**：每个状态只对应一个确定动作，$\pi(s) = a$，可以看作概率版本的极端情况（某个动作概率为1，其余为0）。
	- **随机性策略（Stochastic Policy）**：每个状态对应动作的概率分布，$\pi(a \mid s) \in [0,1]$。PPO、SAC等现代算法都学习随机策略。
- 强化学习的目标即找到最优策略 $\pi^*$，使得期望累积回报最大化。
- 定义策略 $\pi$ 比 $\pi'$ 更好，即满足 $\pi$ 在**所有状态下**的期望回报都不低于 $\pi'$：
$$\pi \geq \pi' \quad \Leftrightarrow \quad V^\pi(s) \geq V^{\pi'}(s) \quad \forall s \in \mathcal{S}$$
- **最优策略** $\pi^*$ 是同时在所有状态下都最好的策略：
$$\pi^* \geq \pi \quad \forall \pi$$
- 即：
$$V^{\pi^*}(s) \geq V^\pi(s) \quad \forall s \in \mathcal{S},\ \forall \pi$$
> **定理（最优策略存在性）：** 对于任何有限MDP（$|\mathcal{S}| < \infty,\ |\mathcal{A}| < \infty,\ \gamma < 1$），最优策略一定存在，且一定存在一个**确定性**的最优策略。
#### 价值函数 Value Function
- 为了比较策略好坏，我们定义价值函数。
	- **状态价值函数 $V^\pi(s)$**：从状态 $s$ 出发、遵循策略 $\pi$，期望能拿到多少总回报：
$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right]$$
	- **动作价值函数 $Q^\pi(s, a)$**：在状态 $s$ 执行动作 $a$、之后遵循策略 $\pi$，期望能拿到多少总回报：
$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]$$
- $Q$ 函数比 $V$ 函数多涵盖了"在当前状态下，具体哪个动作更好"的信息。DQN直接学习 $Q$ 函数，然后选 $Q$ 值最大的动作。
$$V^\pi(s) = \sum_a \pi(a|s) \cdot Q^\pi(s, a)$$
> 状态价值 = 所有动作的Q值按策略概率加权求和。
#### 理解MDP
- MDP中Agent与环境的交互过程：
```
初始状态 s₀
    ↓
Agent用策略π采样动作 a₀ ~ π(·|s₀)
    ↓
环境根据P(s'|s₀,a₀)转移到新状态 s₁，并返回奖励 R₁
    ↓
Agent用策略π采样动作 a₁ ~ π(·|s₁)
    ↓
... 循环直到终止状态
    ↓
累积所有奖励计算Return: G₀ = R₁ + γR₂ + γ²R₃ + ...
```
- 这个循环产生一条**轨迹（Trajectory）**：$\tau = (s_0, a_0, R_1, s_1, a_1, R_2, \ldots)$，RL算法的本质就是从这些轨迹中学习，不断改进策略。
##### MDP假设
- **状态完全可观测假设**：MDP假设Agent能直接观测到完整的系统状态。现实中往往不行（比如在扑克牌里看不到对手的牌）——这时用**部分可观测MDP（POMDP）**，但求解更难。
- **环境平稳性**：MDP假设转移概率 $P$ 和奖励函数 $R$ 不随时间变化。现实中环境可能非平稳。
- **单Agent**：标准MDP只有一个决策者。多智能体场景需要**MARL（Multi-Agent RL）**。
- MDP = 用 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ 五元组描述"Agent在世界中做序列决策"这件事，核心假设是马尔可夫性（未来只取决于现在），核心目标是找到最优策略最大化折扣累积回报。理解了这个框架，后续的Q-Learning、Policy Gradient、PPO都只是在用不同方式求解这个框架下的优化问题。
### Bellman方程
#### Bellman期望方程 (Bellman Expectation Equation)
- MDP中状态价值函数满足递归关系：
$$\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi[G_t \mid s_t = s]\\
&= \mathbb{E}_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid s_t = s]\\
&= \mathbb{E}_\pi[R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + \cdots) \mid s_t = s]\\
&= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid s_t = s]\\
&= \mathbb{E}_\pi[R_{t+1} \mid s_t = s] + \gamma \mathbb{E}_\pi[G_{t+1} \mid s_t = s]
\end{aligned}$$
> 当前状态的价值 = 即时奖励 + 折扣后的下一状态价值
- Agent在状态 $s$ 时，有两个随机源：
	- 策略 $\pi$ 选择动作 $a$，概率为 $\pi(a \mid s)$
	- 环境根据 $P(s' \mid s, a)$ 转移到下一状态 $s'$
$$
\begin{aligned}
\mathbb{E}_\pi[R_{t+1} \mid s_t = s] &= \sum_{a} \pi(a|s)\mathbb{E}_\pi[R_{t+1} \mid s_t = s, a_t = a] \\
&= \sum_{a} \pi(a|s) \sum_{s'} P(s' \mid s, a) R(s, a, s')
\end{aligned}
$$
$$
\begin{aligned}
\mathbb{E}_\pi[G_{t+1} \mid s_t = s]
&= \sum_{s'} \mathbb{E}_\pi[G_{t+1} \mid s_{t+1} = s'] \underbrace{\sum_a P(s' \mid s, a)\pi(a|s)}_{P(s' \mid s)\ \text{（对动作边际化）}} \\
&= \sum_{s'} V^\pi(s') \sum_a P(s' \mid s, a)\pi(a|s)
\end{aligned}
$$
$$
\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi[R_{t+1} \mid s_t = s] + \gamma \mathbb{E}_\pi[G_{t+1} \mid s_t = s] \\
&= \sum_a \pi(a|s) \sum_{s'} P(s' \mid s, a) R(s, a, s') + \gamma \sum_a \pi(a|s) \sum_{s'} P(s' \mid s, a) V^\pi(s') \\
&= \sum_a \pi(a|s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right], \quad \forall s \in \mathcal{S}
\end{aligned}
$$

- 通过以上方式逐步展开期望即得到**Bellman期望方程**：
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$
> 当前状态的价值 = 对所有可能的动作（按策略概率加权）× 对所有可能的下一状态（按转移概率加权）× (执行动作后的即时奖励 + 折扣后的下一状态价值的期望)

##### 矩阵形式
- 将按**单个状态 $s$** 展开的标量方程，转化为涵盖**所有状态 $\mathcal{S}$** 的向量/矩阵方程。假设状态空间中有 $n$ 个状态，即 $|\mathcal{S}| = n$。
- **状态价值向量 $V^\pi$**：一个 $n \times 1$ 的列向量，第 $i$ 个元素是 $V^\pi(s_i)$。
$$V^\pi = \begin{bmatrix} V^\pi(s_1) \\ V^\pi(s_2) \\ \vdots \\ V^\pi(s_n) \end{bmatrix}$$
- **策略下的期望奖励向量 $R^\pi$**：一个 $n \times 1$ 的列向量。它的第 $i$ 个元素代表在状态 $s_i$ 下遵循策略 $\pi$ 获得的**单步期望奖励**。
$$R^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) R(s,a,s')$$
$$R^\pi = \begin{bmatrix} R^\pi(s_1) \\ R^\pi(s_2) \\ \vdots \\ R^\pi(s_n) \end{bmatrix}$$
- **策略下的状态转移概率矩阵 $P^\pi$**：一个 $n \times n$ 的方阵。它的第 $i$ 行第 $j$ 列的元素 $P^\pi_{ij}$ 代表：在状态 $s_i$ 下遵循策略 $\pi$，转移到状态 $s_j$ 的概率。
$$P^\pi_{ij} = \sum_a \pi(a|s_i) P(s_j|s_i,a)$$
$$P^\pi = \begin{bmatrix}
    P^\pi_{11} & P^\pi_{12} & \cdots & P^\pi_{1n} \\
    P^\pi_{21} & P^\pi_{22} & \cdots & P^\pi_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    P^\pi_{n1} & P^\pi_{n2} & \cdots & P^\pi_{nn}
    \end{bmatrix}$$
- 矩阵形式的Bellman方程即为：
$$V^\pi = R^\pi + \gamma P^\pi V^\pi$$
- 可得解析解：
$$
\begin{aligned}
V^\pi - \gamma P^\pi V^\pi &= R^\pi \\
(I - \gamma P^\pi) V^\pi &= R^\pi \\
V^\pi &= (I - \gamma P^\pi)^{-1} R^\pi
\end{aligned}
$$
- 但解析解涉及到对高维矩阵 $P^{\pi}$ 求逆，在工程中不可行，需要通过迭代更新逼近 $V^{\pi}$。
- 将原等式 $V^\pi = R^\pi + \gamma P^\pi V^\pi$ 中的左右两边分别视为"下一步的值"和"当前步的值"，构造一个迭代式：
$$V_{k+1} = R^\pi + \gamma P^\pi V_k$$
- 设定一个初始的价值向量 $V_0$，利用上述公式，代入 $V_0$ 计算出 $V_1$，再代入 $V_1$ 计算出 $V_2$……，产生一个价值向量的序列 $\{V_0, V_1, V_2, \dots, V_k\}$。
- 由于贝尔曼算子是一个**压缩映射（Contraction Mapping）**，数学上可以严格证明，无论初始向量 $V_0$ 是什么，随着迭代次数 $k$ 的增加，这个序列最终一定会收敛到真实的策略价值函数 $V^\pi$。用公式表示即为：
$$\lim_{k \to \infty} V_k = V^\pi = (I - \gamma P^\pi)^{-1} R^\pi$$
###### 收敛性证明
- 定义第 $k$ 步迭代的误差为 $\delta_k$：
$$ \delta_k = V_k - V^\pi $$
- 可以得到 $V_{k+1} = \delta_{k+1} + V^\pi$ 以及 $V_k = \delta_k + V^\pi$。
- 代入迭代更新公式 $V_{k+1} = R^\pi + \gamma P^\pi V_k$ 中，得到：
$$ \delta_{k+1} + V^\pi = R^\pi + \gamma P^\pi(\delta_k + V^\pi) $$
$$ \delta_{k+1} = -V^\pi + R^\pi + \gamma P^\pi \delta_k + \gamma P^\pi V^\pi $$
- 由于 $V^\pi = R^\pi + \gamma P^\pi V^\pi$，上式简化为：
$$ \delta_{k+1} = \gamma P^\pi \delta_k $$
- 即每进行一次迭代，误差都会被状态转移矩阵 $P^\pi$ 乘以折扣因子 $\gamma$ 进行缩放。
$$
\begin{aligned}
\delta_{k+1} &= \gamma P^\pi \delta_k \\
&= \gamma^2 (P^\pi)^2 \delta_{k-1} \\
&= \cdots \\
&= \gamma^{k+1} (P^\pi)^{k+1} \delta_0
\end{aligned}
$$
- 状态转移概率矩阵 $P^\pi$ 每一行元素之和都为 1，每个元素都满足 $0 \le [(P^\pi)^k]_{ij} \le 1$；误差因子满足 $0 \le \gamma < 1$，$\lim_{k \to \infty} \gamma^k = 0$。
- 因为 $(P^\pi)^{k+1}$ 是有界（且最大为1）的，而 $\gamma^{k+1}$ 趋于 0，所以它们的乘积必然趋于 0：
$$ \lim_{k \to \infty} \delta_{k+1} = \lim_{k \to \infty} \gamma^{k+1} (P^\pi)^{k+1} \delta_0 = \mathbf{0} $$
- **因此无论选择任意初始价值 $V_0$，不断使用贝尔曼方程进行迭代，最终一定能收敛到真实的策略价值 $V^\pi$。**
$$\lim_{k \to \infty} V_k = V^\pi = (I - \gamma P^\pi)^{-1} R^\pi$$

- 而动作价值函数 $Q^\pi(s,a)$ 的含义是：在状态 $s$ **先执行动作 $a$**，之后遵循策略 $\pi$：
$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]$$
$$V^\pi(s) = \sum_a \pi(a|s) \cdot Q^\pi(s, a)$$
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$
- 比较上述公式可得
$$\begin{aligned}
Q^\pi(s, a) &= \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]\\
&= \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a') \right]
\end{aligned}$$
- 即 $V$ 和 $Q$ 的函数可以互相表达：
$$V^\pi(s) = \sum_a \pi(a \mid s) \cdot Q^\pi(s, a)$$
$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$
#### Bellman最优方程 (Bellman Optimality Equation)
- 最优价值函数定义为：
$$V^*(s) = \max_\pi V^\pi(s)$$
- 最优Q函数：
$$Q^*(s, a) = \max_\pi Q^\pi(s, a)$$
- 两者关系为：
$$V^*(s) = \max_a Q^*(s, a)$$
- $Q^*$ 已知，最优策略为：
$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

##### 向量符号约定：$R_a$ 与 $P_a$

> 在最优方程的推导中，我们固定一个具体动作 $a \in \mathcal{A}$，定义以下两个对象（假设 $|\mathcal{S}| = n$）：
>
> - **动作 $a$ 的奖励向量 $R_a$**：一个 $n \times 1$ 的列向量，第 $i$ 个分量为在状态 $s_i$ 执行动作 $a$ 的**期望即时奖励**：
> $$[R_a]_i = \sum_{s'} P(s' \mid s_i, a)\, R(s_i, a, s')$$
>
> - **动作 $a$ 的转移矩阵 $P_a$**：一个 $n \times n$ 的方阵，第 $(i,j)$ 个分量为在状态 $s_i$ 执行动作 $a$ 后转移到状态 $s_j$ 的概率：
> $$[P_a]_{ij} = P(s_j \mid s_i, a)$$
>
> 有了这两个定义，标量形式的单状态方程：
> $$V^*(s_i) = \max_a \left( [R_a]_i + \gamma [P_a]_{i,:} V^* \right)$$
> 可以紧凑地写成**向量形式**（对所有状态同时成立）：
> $$\boxed{V^* = \max_a \left( R_a + \gamma P_a V^* \right)}$$
> 其中 $\max_a$ 表示对每个状态分量分别取最大值（逐元素最大化）。

##### Bellman最优方程

- 最优策略对应的即为**Bellman最优方程**：
	- 最优 $V$ 的Bellman方程（标量形式）：
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$
	- 最优 $V$ 的Bellman方程（向量形式）：
$$V^* = \max_a \left( R_a + \gamma P_a V^* \right)$$
		- *遍历所有动作 $a$，计算每个动作带来的即时奖励期望与转移后的最优状态价值之和，然后取最大值。*
	- 最优 $Q$ 的Bellman方程：
$$Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s', a') \right]$$
		- *当前执行了动作 $a$，状态按概率转移到 $s'$ 后，在 $s'$ 处一定会选择那个能带来最大后续 $Q^*$ 值的最优动作 $a'$。*

- 因为 $\max$ 导致的**非线性**，**无法**通过矩阵求逆来一步算出最优价值 $V^*$。因此**必须依赖迭代算法**（如动态规划中的**价值迭代 Value Iteration** 和 **策略迭代 Policy Iteration**，或者无模型的 **Q-Learning**），通过不断重复
$$V_{k+1} = \max_a \left( R_a + \gamma P_a V_k \right)$$
来逼近非线性方程 $V^* = \max_a(R_a + \gamma P_a V^*)$ 的唯一不动点 $V^*$。

- 和Bellman期望方程的区别：
	- **期望方程**描述的是"给定策略 $\pi$，这个状态值多少"：
$$\text{期望方程：} \quad \sum_{a'} \pi(a' \mid s') Q^\pi(s', a') \quad \leftarrow \text{按策略加权平均}$$
	- **最优方程**描述的是"这个状态最多值多少"：
$$\text{最优方程：} \quad \max_{a'} Q^*(s', a') \quad \leftarrow \text{直接取最大值}$$
#### Bellman最优公式求解
##### Bellman 备份算子

为了统一描述迭代收敛的理论基础，先定义两个核心算子：

- **期望算子 $T^\pi$**（对固定策略 $\pi$）：
$$\left(T^\pi V\right)(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V(s') \right]$$
	向量形式：$T^\pi V = R^\pi + \gamma P^\pi V$
	Bellman期望方程就是其不动点：$V^\pi = T^\pi V^\pi$

- **最优算子 $T^*$**：
$$\left(T^* V\right)(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V(s') \right]$$
	向量形式：$T^* V = \max_a \left( R_a + \gamma P_a V \right)$
	Bellman最优方程就是其不动点：$V^* = T^* V^*$

> **两个算子均为压缩映射**（在 $\|\cdot\|_\infty$ 范数下），这保证了不动点的唯一性和迭代的全局收敛性。

##### 压缩映射定理 (Contraction Mapping Theorem)

- **定理（Banach不动点定理）：** 若算子 $T$ 在完备度量空间 $(\mathbb{R}^n, \|\cdot\|_\infty)$ 上满足
$$\| T(V_1) - T(V_2) \|_\infty \leq \gamma \| V_1 - V_2 \|_\infty, \quad 0 \leq \gamma < 1$$
则 $T$ 有**唯一不动点** $V^*$，且对任意初始猜测 $V_0$，迭代序列 $V_{k+1} = T(V_k)$ 以指数速度收敛：
$$\| V_k - V^* \|_\infty \leq \gamma^k \| V_0 - V^* \|_\infty \to 0$$

- **证明 $T^*$ 是 $\gamma$-压缩映射：**

$$\begin{aligned}
\| T^* V_1 - T^* V_2 \|_\infty
&= \left\| \max_a \left(R_a + \gamma P_a V_1\right) - \max_a \left(R_a + \gamma P_a V_2\right) \right\|_\infty \\
&\leq \left\| \max_a \left| \gamma P_a (V_1 - V_2) \right| \right\|_\infty \\
&\leq \gamma \max_a \| P_a (V_1 - V_2) \|_\infty \\
&\leq \gamma \| V_1 - V_2 \|_\infty
\end{aligned}$$

	其中第二行用了 $|\max f - \max g| \leq \max|f - g|$，最后一行用了 $P_a$ 是随机矩阵（每行和为1），故 $\|P_a x\|_\infty \leq \|x\|_\infty$。

- 因此，方程 $V = T^* V$（即 $V^* = \max_a(R_a + \gamma P_a V^*)$）**必然存在唯一解 $V^*$**，且可以通过迭代求解。

- 在已知 $V^*$ 的情况下，最优策略 $\pi^*$ 即为逐状态使价值最大化的确定性动作选择：
$$\pi^*(s) = \arg\max_a \left( [R_a]_s + \gamma [P_a]_{s,:} V^* \right) = \arg\max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

##### 最优策略不变性

> **定理（最优策略不变性）：** 设 $V^*$ 是Bellman最优方程 $V^* = \max_a(R_a + \gamma P_a V^*)$ 的唯一不动点，则贪婪策略
> $$\pi^*(s) = \arg\max_a \left( [R_a]_s + \gamma [P_a]_{s,:} V^* \right)$$
> 是最优策略，即 $V^{\pi^*} = V^*$，且对任意策略 $\pi$ 均有 $V^{\pi^*}(s) \geq V^\pi(s),\ \forall s \in \mathcal{S}$。

**直觉理解：** 若在每个状态下都选择了使"当前奖励 + 折扣后的最优未来价值"最大的动作，就不可能再有策略比它更好——因为 $V^*$ 已经是所有策略中的上确界。

##### 价值迭代 (Value Iteration)
- 先不考虑策略，单纯利用贝尔曼最优方程的"压缩映射"性质，暴力迭代计算出全局最优的 $V^*$。等 $V^*$ 收敛后，最后再做一次贪婪选择提取最优策略 $\pi^*$。
- **价值更新公式 (Value Update)**：在第 $k$ 次迭代，对所有的状态 $s \in \mathcal{S}$ 进行更新：
$$V_{k+1}(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V_k(s') \right]$$

$$V_{k+1} = \max_a \left( R_a + \gamma P_a V_k \right)$$

- **策略提取公式 (Policy Extraction)**：当 $\|V_{k+1} - V_k\|_\infty$ 足够小（收敛），通过 $\arg\max_a$ 提取最优策略：
$$\pi^*(s) = \arg\max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

##### 策略迭代 (Policy Iteration)
- 随机初始化一个确定性策略 $\pi_0$，然后不断交替进行"策略评估（算出当前策略有多好）"和"策略改进（贪婪地寻找更好的动作）"，直到策略不再发生变化。
- **策略评估 (Policy Evaluation)**：给定当前的确定性策略 $\pi_k$，求解线性方程组计算 $V^{\pi_k}$：
$$V^{\pi_k}(s) = \sum_{s'} P\!\left(s' \mid s, \pi_k(s)\right) \left[ R\!\left(s, \pi_k(s), s'\right) + \gamma V^{\pi_k}(s') \right]$$
$$V^{\pi_k} = \left(I - \gamma P_{a_k}\right)^{-1} R_{a_k}$$
	其中 $a_k(s) = \pi_k(s)$ 是固定的确定性动作，$P_{a_k}$ 和 $R_{a_k}$ 即为该确定性动作对应的转移矩阵和奖励向量（与 $P_a, R_a$ 符号一致）。

- **策略改进 (Policy Improvement)**：在每个状态 $s$ 处"贪婪"地遍历所有动作 $a$，寻找比当前策略有更大期望回报的动作：
$$\pi_{k+1}(s) = \arg\max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^{\pi_k}(s') \right]$$

> **策略改进定理：** 若 $\pi_{k+1}$ 由上式通过贪婪改进得到，则对所有状态 $s \in \mathcal{S}$：
> $$V^{\pi_{k+1}}(s) \geq V^{\pi_k}(s)$$
> 即策略改进一定不会变差（单调非递减）。

- 如果对所有的状态 $s$，都有 $\pi_{k+1}(s) = \pi_k(s)$（策略不再变化），则由策略改进定理可知此时满足Bellman最优方程，$\pi_k$ 即为最优策略 $\pi^*$。

##### 价值迭代 vs 策略迭代对比

|   比较维度   |      价值迭代       |      策略迭代       |
| :------: | :-------------: | :-------------: |
| **每轮操作** | 仅更新 $V$，不维护显式策略 |  策略评估 + 策略改进交替  |
| **收敛轮数** |    通常需要更多轮次     | 有限MDP中有限步内精确收敛  |
| **每轮代价** |     低（一次扫描）     | 高（策略评估需求解线性方程组） |
| **适用场景** |   大状态空间，近似求解    |  小/中状态空间，精确求解   |
| **理论基础** |  $T^*$ 的压缩映射性质  |  策略改进定理 + 单调收敛  |

#### 理解Bellman方程
- Bellman方程把一个"无限步的优化问题"变成了"一步决策 + 剩余问题的递归"。你不需要考虑未来的所有可能性，只需要知道下一个状态的价值是多少，就能计算当前状态的价值。
- Bellman方程本质上是一个**不动点方程**——最优价值函数 $V^*$ 满足：
$$V^* = T^* V^*$$
- 其中 $T^*$ 是Bellman最优算子。这意味着 $V^*$ 是这个方程的唯一解，且可以通过反复迭代收敛到它（Value Iteration的理论基础）。
