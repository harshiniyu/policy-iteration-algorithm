

# POLICY ITERATION ALGORITHM

## AIM
Implement policy iteration algorithm to find optimal policy by iteratively maximizing the value function.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
# Step 1:
Import required libraries.
# Step 2:
Load the frozen lake environment.
# Step 3:
Define the value evaluation, value improvement and value iteration functions.
# Step 4: 
Run the functions and display the results.

## POLICY IMPROVEMENT FUNCTION
##### Name : Harshini Y
##### Register Number : 212223240050
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_pi = lambda s: np.argmax(Q[s])
    return new_pi
```

## POLICY ITERATION FUNCTION
##### Name : Harshini Y
##### Register Number : 212223240050
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
    pi = np.zeros(len(P), dtype=int)
    while True:
        pi_func = lambda s: pi[s]
        V = policy_evaluation(pi_func, P, gamma, theta)
        policy_stable = True
        for s in range(len(P)):
            old_action = pi[s]
            Q = np.zeros(len(P[s]))
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[a] += prob * (reward + gamma * V[next_state] * (not done))
            pi[s] = np.argmax(Q)
            if old_action != pi[s]:
                policy_stable = False
        if policy_stable:
            break

    return V, lambda s: pi[s]
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy

<img width="992" height="217" alt="image" src="https://github.com/user-attachments/assets/65ba826d-6307-42ce-8565-4d38788d05b3" />


### 2. Policy, Value function and success rate for the Improved Policy

<img width="606" height="177" alt="image" src="https://github.com/user-attachments/assets/ea760a55-aaa6-4792-8cc2-9bb2be7bdd6c" />


### 3. Policy, Value function and success rate after policy iteration

<img width="607" height="185" alt="image" src="https://github.com/user-attachments/assets/a91dbe3e-1fd0-499f-9bfe-ed738fccb5bc" />


## RESULT:
Therefore, policy iteration algorithm to find optimal policy by iteratively maximizing the value function is successfully implemented.


### 2. Policy, Value function and success rate for the Improved Policy

<img width="680" height="190" alt="image" src="https://github.com/user-attachments/assets/23232b61-f2b8-4ee9-9515-0c9a426d95ef" />


### 3. Policy, Value function and success rate after policy iteration

<img width="952" height="147" alt="image" src="https://github.com/user-attachments/assets/a47cd57f-add8-4484-9672-96cda315e177" />



## RESULT:
Therefore, policy iteration algorithm to find optimal policy by iteratively maximizing the value function is successfully implemented.
