import numpy as np
from aux_func import get_reward,deepcopy
from tqdm import tqdm
from numba import jit

@jit(nopython=True)
def UCB_N(T,reward_mat,arm_type,neighbor_init,change_arms_list,seed):
    np.random.seed(seed)
    N = np.zeros(T)
    X_hat = np.zeros(T)
    c = np.zeros(T)
    
    reward = np.zeros(T)
    expect_reward = np.zeros(T)
    ch = np.zeros(T)
    neighbor=deepcopy(neighbor_init,len(neighbor_init))
    arms_num=0
    cons=np.sqrt(np.log(np.sqrt(2)*T**2))
    #for t in tqdm(range(T),desc="UCB_N: "):
    for t in range(T):
        
        if change_arms_list[t]!=0:
            num=change_arms_list[t]
            for _ in range(num):
                arms_num += 1
                N[arms_num-1]=0
                X_hat[arms_num-1]=0
                c[arms_num-1]=cons
                neigh1=neighbor[arms_num-1]
                for arm in set(neigh1):
                    if (arms_num-1) not in neighbor[arm]:
                        neighbor[arm].append(arms_num-1)
        
        j = np.argmax(X_hat[0:arms_num] + c[0:arms_num])
        #neigh=neighbor[j]
        neigh=np.array(list(neighbor[j]))
        ch[t] = j
        expect_reward[t] = reward_mat[j,0]
        
        r_jj=get_reward(reward_mat,neigh,arm_type)
        if len(neigh)==1:
            reward[t]=r_jj[0]
        else:
            reward[t]=r_jj[np.where(neigh==j)][0]
        c[neigh]=cons/np.sqrt(N[neigh]+1)
        X_hat[neigh]= (X_hat[neigh]*N[neigh]+r_jj)/(N[neigh]+1)          
        N[neigh] += 1
        
        a=0
        if t%10000==0:
            a=0
      
    return reward, expect_reward, ch


