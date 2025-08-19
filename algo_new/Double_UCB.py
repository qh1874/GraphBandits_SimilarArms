import numpy as np
from aux_func import get_reward,deepcopy
from tqdm import tqdm
from numba import jit

@jit(nopython=True)
def is_clique(neighbor,neigh):
    
    for i in neigh:
        neigh_i = neighbor[i]
        for k in neigh:
            if k not in neigh_i:
                return False,np.array([i,k])
    return True,np.array([i,k])

# @jit(nopython=True)
# def find_Ijt(neighbor,Njt):
#     flag=False
#     for i in Njt:
#         for j in Njt:
#             if i not in neighbor[j]:
#                 flag=True
#                 # neigh1=set(neighbor[i]).intersection(set(Njt))
#                 # neigh2=set(neighbor[j]).intersection(set(Njt))
#                 # if is_clique(neighbor,neigh1) and is_clique(neighbor,neigh2):
#                 return np.array([i,j])
#     # if flag == False:
#     #     print("Error")
#     return np.array([i,j])
    

            

@jit(nopython=True)
def Double_UCB(T,reward_mat,arm_type,neighbor_init,change_arms_list,seed):
    np.random.seed(seed)
    N=np.zeros(T,dtype='int32')
    X_hat=np.zeros(T)
    c=np.zeros(T)
    c_add_optimal=np.zeros(T)
    c_add=np.zeros(T)
    c_sub=np.zeros(T)
    N_optimal=np.zeros(T,dtype='int32')
    
    arm_ind=np.array([0])
    X_hat_optimal=np.zeros(T)
    
    reward = np.zeros(T)
    expect_reward = np.zeros(T)
    ch = np.zeros(T)
    neighbor=deepcopy(neighbor_init,len(neighbor_init))
    arms_num=0
    cons=np.sqrt(np.log(np.sqrt(2)*T))
    #for t in tqdm(range(T),desc="Double-UCB: "):
    for t in range(T):
        
        if change_arms_list[t]!=0:

            num=change_arms_list[t]
            for _ in range(num):
                arms_num += 1
                N[arms_num-1]=0
                
                flag=1
                neigh1=neighbor[arms_num-1]
                for arm in set(neigh1):
                    if arm in arm_ind:
                        flag = 0                    
                    if (arms_num-1) not in neighbor[arm]:
                        neighbor[arm].append(arms_num-1)
                
                if flag==1:
                    arm_ind=np.append(arm_ind,arms_num-1)
                    X_hat_optimal[arms_num-1]=0
                    c_add_optimal[arms_num-1]=cons
                    N_optimal[arms_num-1] =0
                    
                X_hat[arms_num-1]=0
                c_add[arms_num-1]=cons
                c_sub[arms_num-1]=-cons
                
        if t< len(arm_ind):
            j=arm_ind[t]
        else:
            j1=arm_ind[np.argmax(c_add_optimal[arm_ind])]
            temp_nei=np.array(list(neighbor[j1]))
            j=temp_nei[np.argmax(c_add[temp_nei])]
            #j=neighbor[j1][np.argmax(c_sub[neighbor[j1]])]
            #j=neighbor[j1][np.argmax(X_hat[neighbor[j1]])]
        if t%10000==0:
            a=0
        neigh=np.array(list(neighbor[j]))
        ch[t] = j 
        expect_reward[t] = reward_mat[j,0]
        #update
        r_jj=get_reward(reward_mat,neigh,arm_type)
        if len(neigh)==1:
            reward[t]=r_jj[0]
        else:
            reward[t]=r_jj[np.where(neigh==j)][0]
        
        c[neigh]=cons/np.sqrt(N[neigh]+1)
       
        X_hat[neigh]= (X_hat[neigh]*N[neigh]+r_jj)/(N[neigh]+1)          
        c_add[neigh]=X_hat[neigh]+c[neigh]
        c_sub[neigh]=X_hat[neigh]-c[neigh]
        N[neigh] += 1
        
        X_hat_optimal[arm_ind]= X_hat[arm_ind]
        N_optimal[arm_ind] = N[arm_ind]       
        c_add_optimal[arm_ind]=c_add[arm_ind]
      
    return reward, expect_reward, ch, len(arm_ind)


@jit(nopython=True)
def C_UCB(T,reward_mat,arm_type,neighbor_init,change_arms_list,seed):
    np.random.seed(seed)
    N=np.zeros(T,dtype='int32')
    X_hat=np.zeros(T)
    c=np.zeros(T)
    c_add_optimal=np.zeros(T)
    c_add=np.zeros(T)
    c_sub=np.zeros(T)
    N_optimal=np.zeros(T,dtype='int32')
    
    arm_ind=np.array([0])
    X_hat_optimal=np.zeros(T)
    
    reward = np.zeros(T)
    expect_reward = np.zeros(T)
    ch = np.zeros(T)
    neighbor=deepcopy(neighbor_init,len(neighbor_init))
    arms_num=0
    cons=np.sqrt(np.log(np.sqrt(2)*T))
    #for t in tqdm(range(T),desc="C-UCB: "):
    for t in range(T):
        
        if change_arms_list[t]!=0:

            num=change_arms_list[t]
            for _ in range(num):
                arms_num += 1
                N[arms_num-1]=0
                
                flag=1
                neigh1=neighbor[arms_num-1]
                for arm in set(neigh1):
                    if arm in arm_ind:
                        flag = 0                    
                    if (arms_num-1) not in neighbor[arm]:
                        neighbor[arm].append(arms_num-1)
                
                if flag==1:
                    #arm_ind.append(arms_num-1)
                    arm_ind=np.append(arm_ind,arms_num-1)
                    X_hat_optimal[arms_num-1]=0
                    c_add_optimal[arms_num-1]=cons
                    N_optimal[arms_num-1] =0
                    
                X_hat[arms_num-1]=0
                c_add[arms_num-1]=cons
                c_sub[arms_num-1]=-cons
        
        if t< len(arm_ind):
            j=arm_ind[t]
        else:
            j1=arm_ind[np.argmax(c_add_optimal[arm_ind])]
            temp_nei=np.array(list(neighbor[j1]))
            j=temp_nei[np.argmax(c_sub[temp_nei])]
            #j=neighbor[j1][np.argmax(c_sub[neighbor[j1]])]
            #j=neighbor[j1][np.argmax(X_hat[neighbor[j1]])]
        if t%10000==0:
            a=0
        neigh=np.array(list(neighbor[j]))
        ch[t] = j 
        expect_reward[t] = reward_mat[j,0]
        #update
        r_jj=get_reward(reward_mat,neigh,arm_type)
        if len(neigh)==1:
            reward[t]=r_jj[0]
        else:
            reward[t]=r_jj[np.where(neigh==j)][0]
        
        c[neigh]=cons/np.sqrt(N[neigh]+1)
       
        X_hat[neigh]= (X_hat[neigh]*N[neigh]+r_jj)/(N[neigh]+1)          
        c_add[neigh]=X_hat[neigh]+c[neigh]
        c_sub[neigh]=X_hat[neigh]-c[neigh]
        N[neigh] += 1
        
        X_hat_optimal[arm_ind]= X_hat[arm_ind]
        N_optimal[arm_ind] = N[arm_ind]       
        c_add_optimal[arm_ind]=c_add[arm_ind]
        
    return reward, expect_reward, ch


@jit(nopython=True)
def C_UCB_WithGraph(T,reward_mat,arm_type,neighbor_init,change_arms_list,seed):
    np.random.seed(seed)
    N=np.zeros(T,dtype='int32')
    X_hat=np.zeros(T)
    c=np.zeros(T)
    c_add_optimal=np.zeros(T)
    c_add=np.zeros(T)
    c_sub=np.zeros(T)
    N_optimal=np.zeros(T,dtype='int32')
    
    arm_ind=np.array([0])
    X_hat_optimal=np.zeros(T)
    
    reward = np.zeros(T)
    expect_reward = np.zeros(T)
    ch = np.zeros(T)
    neighbor=deepcopy(neighbor_init,len(neighbor_init))
    arms_num=0
    cons=np.sqrt(np.log(np.sqrt(2)*T))
    #for t in tqdm(range(T),desc="C-UCB: "):
    for t in range(T):
        
        if change_arms_list[t]!=0:

            num=change_arms_list[t]
            for _ in range(num):
                arms_num += 1
                N[arms_num-1]=0
                
                flag=1
                neigh1=neighbor[arms_num-1]
                for arm in set(neigh1):
                    if arm in arm_ind:
                        flag = 0                    
                    if (arms_num-1) not in neighbor[arm]:
                        neighbor[arm].append(arms_num-1)
                
                if flag==1:
                    #arm_ind.append(arms_num-1)
                    arm_ind=np.append(arm_ind,arms_num-1)
                    X_hat_optimal[arms_num-1]=0
                    c_add_optimal[arms_num-1]=cons
                    N_optimal[arms_num-1] =0
                    
                X_hat[arms_num-1]=0
                c_add[arms_num-1]=cons
                c_sub[arms_num-1]=-cons
        
        if t< len(arm_ind):
            j=arm_ind[t]
        else:
            j1=arm_ind[np.argmax(c_add_optimal[arm_ind])]
            isClique,Ijt = is_clique(neighbor,neighbor[j1])
            if isClique:
                temp_nei=np.array(list(neighbor[j1]))
                j=temp_nei[np.argmax(c_sub[temp_nei])]
            else:
                #Ijt = find_Ijt(neighbor,neighbor[j1])
                j2 = Ijt[np.argmax(c_add_optimal[Ijt])]
                temp_set = set(neighbor[j2]).intersection(set(neighbor[j1]))
                temp_nei=np.array(list(temp_set))
                j=temp_nei[np.argmax(c_sub[temp_nei])]
          
        if t%10000==0:
            a=0
        neigh=np.array(list(neighbor[j]))
        ch[t] = j 
        expect_reward[t] = reward_mat[j,0]
        #update
        r_jj=get_reward(reward_mat,neigh,arm_type)
        if len(neigh)==1:
            reward[t]=r_jj[0]
        else:
            reward[t]=r_jj[np.where(neigh==j)][0]
        
        c[neigh]=cons/np.sqrt(N[neigh]+1)
       
        X_hat[neigh]= (X_hat[neigh]*N[neigh]+r_jj)/(N[neigh]+1)          
        c_add[neigh]=X_hat[neigh]+c[neigh]
        c_sub[neigh]=X_hat[neigh]-c[neigh]
        N[neigh] += 1
        
        X_hat_optimal[arm_ind]= X_hat[arm_ind]
        N_optimal[arm_ind] = N[arm_ind]       
        c_add_optimal[arm_ind]=c_add[arm_ind]
        
    return reward, expect_reward, ch