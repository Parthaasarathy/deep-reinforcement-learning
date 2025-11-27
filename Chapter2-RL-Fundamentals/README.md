# 📘 Chapter 2: Reinforcement Learning Fundamentals  
### *A Complete Beginner Textbook for Absolute Beginners Using GridWorld*

Welcome to **Chapter 2** of your RL journey.  
In Chapter 1, you learned how to build a full custom environment in Gymnasium — GridWorld.  
Now, in Chapter 2, you will learn:

- how agents *think*  
- how agents *choose actions*  
- how agents *learn from experience*  
- how Q-learning works at a fundamental level  

All lessons use your own **GridWorld** environment.

---

# -------------------------------------------
# 🧠 What Is Reinforcement Learning (RL)?
# -------------------------------------------

Reinforcement Learning is a framework where an **agent** learns by interacting with an **environment**.

Every step looks like this:

```
Agent  --action-->  Environment
Environment --reward, new_state--> Agent
```

The agent’s goal is to learn a **policy**:

> A good rule for choosing the best action in each situation.

---

# -------------------------------------------
# 📦 Key Words (Beginner Definitions)
# -------------------------------------------

### **State**  
A numerical representation of “where you are.”  
In GridWorld, the state is:  
```
[agent_x, agent_y, target_x, target_y]
```

---

### **Action**  
What the agent chooses to do.

In GridWorld:
```
0 → move right  
1 → move up  
2 → move left  
3 → move down
```

---

### **Reward**  
A number that tells the agent whether its action was good or bad.

Your environment gives:
- `+10` for reaching the target  
- `-1` per step (to encourage speed)

---

### **Episode**  
One complete run from start → goal (or termination).

---

### **Policy**  
A rule telling the agent:  
> “In state *S*, take action *A*.”

---

### **Q-Value**  
A number that represents:  
> “How good is taking action A in state S?”

Stored inside a **Q-table**.

---

# -------------------------------------------
# 🧪 The Three Agents You Implemented
# -------------------------------------------

In Chapter 2, you built three agents:

1. **Random Agent**  
2. **ε-Greedy Agent**  
3. **Q-Learning Agent**

Each agent is more intelligent than the previous one.

---

# ======================================================
# 1️⃣ RANDOM AGENT — The Baseline (random_agent.py)
# ======================================================

The random agent:

- takes **purely random actions**  
- does **not** learn  
- does **not** think  
- does **not** improve  

Why do we use it?

Because RL always begins with a baseline:
> “What if the agent behaves with zero intelligence?”

A random agent helps us visually understand:
- how the environment behaves
- how movement works
- how rendering works
- how rewards look

### 🔹 How it works
Every step, it simply does:

```python
action = env.action_space.sample()
```

That means:

- pick a random action  
- move  
- repeat  

The agent wanders around until it accidentally reaches the target.

---

# ======================================================
# 2️⃣ ε-GREEDY AGENT — Exploration vs Exploitation (e_greedy_agent.py)
# ======================================================

This is the first time the agent becomes *smart*.

### The key idea:

> Sometimes explore.  
> Sometimes exploit.

---

## 🤔 What is Exploration?

Trying random actions to discover new possibilities.

Example:
- “Maybe left path is shorter?”
- “Let me test a new move.”

---

## 🤔 What is Exploitation?

Using what you currently believe is the best action.

Example:
- Move directly toward the target.

---

## 🧠 ε-Greedy Rule

```
With probability ε     → explore  (random)
With probability 1-ε   → exploit (greedy)
```

In your code:

- ε = 0.3  
- So 30% of the time → random  
- 70% of the time → greedy toward target  

This shows *controlled randomness* — the foundation of learning.

---

# ======================================================
# 3️⃣ Q-LEARNING AGENT — The First Real RL Algorithm (q_learning_agent.py)
# ======================================================

This is where real learning begins.

Q-learning learns a **Q-table**:

```
State × Action → Q-value
```

Each Q-value represents:

> “How good is it to take action A when I am in state S?”

---

# -------------------------------------------
# 🧮 The Q-Learning Formula (Beginner Explanation)
# -------------------------------------------

When the agent takes an action, receives a reward, and sees the next state, we update:

```
Q(s, a) ← Q(s, a) + α * (reward + γ * max(Q(s'), :) - Q(s, a))
```

Where:

- **α (alpha)** = learning rate (how fast we update)
- **γ (gamma)** = discount factor (importance of future rewards)
- **max(Q(s'), :)** = best Q-value in next state  
- **reward + γ * best_next_value** = “target”  

This formula slowly makes Q-values more accurate over time.

---

# -------------------------------------------
# 📌 State Conversion in GridWorld
# -------------------------------------------

Your observation is:

```
[agent_x, agent_y, target_x, target_y]
```

To store Q-values, we convert this into a single number:

```
state_index = function(agent_x, agent_y, target_x, target_y)
```

This lets us create a **Q-table** like:

```
Q = np.zeros((num_states, num_actions))
```

---

# ----------------------------------------------------
# 🏁 What Q-Learning Achieves
# ----------------------------------------------------

After enough episodes:

- the agent learns the fastest path  
- agent moves almost directly toward the target  
- it avoids loops  
- it becomes fully deterministic  
- it behaves intelligently without any hardcoded instructions

You have now implemented **your first true learning agent**.

---

# -------------------------------------------
# 📦 File Overview of Chapter 2
# -------------------------------------------

```
Chapter2-RL-Fundamentals/
│
├── random_agent.py           # agent that moves randomly
├── e_greedy_agent.py         # agent with controlled exploration
├── q_learning_agent.py       # agent that learns optimal policy
│
├── plots/                    # (optional) for graphs later
└── videos/                   # (optional) for saved animations later
```

---

# -------------------------------------------
# ▶ How to Run Each Script
# -------------------------------------------

### 1. Random agent:
```
python Chapter2-RL-Fundamentals/random_agent.py
```

### 2. ε-greedy agent:
```
python Chapter2-RL-Fundamentals/e_greedy_agent.py
```

### 3. Q-learning agent:
```
python Chapter2-RL-Fundamentals/q_learning_agent.py
```

---

# 🎉 End of Chapter 2

You now understand:

- states  
- actions  
- rewards  
- policies  
- exploration  
- exploitation  
- Q-values  
- Q-learning update formula  
- training loops  
- greedy policy testing  

You have officially completed **Reinforcement Learning Basics**.

You are ready for **Chapter 3**:  
> **Deep Q-Networks (DQN)** — Neural Networks + RL.


