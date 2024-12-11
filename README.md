# GraphBandits_SimilarArms
This code is for the paper "Graph Feedback Bandits with Similar Arms".
## Stationary settings
1. Compare the performance of UCB-N under standard graph feedback and graph feedback with similar arms.
   
    First modify param.py.
    ```python
    #parammeter setting
    param = {
        'T': 100000, 'K': 10000, 
        'iter':50,  #repeat times
        'arm_type':0, #0: Gaussian, 1: Bernoulli 
        'saved':True # whether the data has been saved for drawing
    }

    ```
    Then run the file:
    ```
    python test_stationary.py
    ```
2.  Compare the performance of UCB-N, Double-UCB, and Conservative-UCB algorithms.
    Modify parameters:
    ```python
    #parammeter setting
    param = {
    'T': 100000, 'K': 10000, 'epsilon':0.01,
    'iter':20,  #repeat times
    'arm_type':0, #0: Gaussian, 1: Bernoulli 
    'saved':True # whether the data has been saved for drawing
    }
    ```
    Then run the file:
    ```bash
    python test_two_sta.py
    ```
## Ballooning settings
 Compare  our proposed four algorithms: Double-UCB, Conservative-UCB, U-Double-UCB and U-Conservative-UCB.
 Modify parameters:
   (a) $\mathcal{P}$ is Gaussian  $\mathcal{N}(0,1)$.
   ```python
    param = {
    'T': 10000, 'K': 10000, 'epsilon':0.3,
    'iter':20,  #repeat times
    'arm_type':0, #0: Gaussian, 1: Bernoulli, 2: Bernoulli (half-triangle distribution for Ballooning settings )
    'saved':True # whether the data has been saved for drawing
    }
```
(b) $\mathcal{P}$ is Uniform distribution  $U(0,1)$.
```python
param = {
'T': 10000, 'K': 10000, 'epsilon':0.05,
'iter':20,  #repeat times
'arm_type':1, #0: Gaussian, 1: Bernoulli, 2: Bernoulli (half-triangle distribution for Ballooning settings )
'saved':True # whether the data has been saved for drawing
}
```
(c) $\mathcal{P}$ is the half-triangle distribution.
```python
param = {
'T': 10000, 'K': 10000, 'epsilon':0.05,
'iter':20,  #repeat times
'arm_type':2, #0: Gaussian, 1: Bernoulli,2: Bernoulli (half-triangle distribution for Ballooning settings)
'saved':True # whether the data has been saved for drawing
}
```
Then run the file:
```bash
python test_four_ball.py
```