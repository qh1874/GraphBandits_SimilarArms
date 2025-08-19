import numpy as np
from aux_func import get_reward,deepcopy
from tqdm import tqdm
from numba import jit
from numba.typed import List



@jit(nopython=True)
def select_first_geq_zero(L):
    Len=len(L)
    for i in range(Len):
        if L[i] !=0:
            return i
    

tau00=1
@jit(nopython=True)
def Unkown_DUCB_BL(T,reward_mat,arm_type,neighbor_init,change_arms_list,seed):
    np.random.seed(seed)
    tau=int(20*np.sqrt(T))
    N=np.zeros(T,dtype='int64')
    X_hat=np.zeros(T)
    c=np.zeros(T)
    c_add_optimal=np.zeros(T)
    c_add=np.zeros(T)
    c_sub=np.zeros(T)
    N_optimal=np.zeros(T,dtype='int64')
    
    arm_ind=np.array([0])
    X_hat_optimal=np.zeros(T)
    
    reward = np.zeros(T)
    expect_reward = np.zeros(T)
    ch = np.zeros(T)
    neighbor=deepcopy(neighbor_init,len(neighbor_init))
    arms_num=0
    cons=np.sqrt(np.log(np.sqrt(2)*T))
    #for t in tqdm(range(T),desc="Double-UCB: "):
    t=0
    armsnum_temp=0
    flag=-1
    tau0=tau00
    while t<T:
    
        if (t+1)%tau0==0:
            #print("t= ",t)
            tt=arms_num-armsnum_temp  #(t//tau)*tau
            s_temp=np.ones(tt,dtype='int64')
            choose_list=list(arm_ind)
            for i in choose_list:
                neigh_temp1=set([ i_tmp for i_tmp in neighbor[i] if i_tmp>=armsnum_temp])
                #neigh_temp1=set(neighbor[i])
                for k1 in neigh_temp1:
                    if  s_temp[k1-armsnum_temp]!=0:
                        s_temp[k1-armsnum_temp]=0
            while sum(s_temp)!=0:
                index_temp=select_first_geq_zero(s_temp)
                choose_list.append(index_temp+armsnum_temp)
                s_temp[int(index_temp)]=0
                neigh_temp2=set([i_tmp for i_tmp in neighbor[index_temp+armsnum_temp] if i_tmp>=armsnum_temp])
                #neigh_temp2=set(neighbor[index_temp+tt])
                for k2 in neigh_temp2:
                    if s_temp[k2-armsnum_temp]!=0:
                        s_temp[k2-armsnum_temp]=0
            arm_ind=np.array(choose_list)
            flag=len(choose_list)
            armsnum_temp=arms_num
            
            if  tau0*2<tau:
                tau0=tau0*2
            else:
                tau0=tau
        
        if change_arms_list[t]==1:
            arms_num += 1
            N[arms_num-1]=0
            
            neigh1=neighbor[arms_num-1]
            for arm in set(neigh1):                  
                if (arms_num-1) not in neighbor[arm]:
                    neighbor[arm].append(arms_num-1)
                    
            X_hat_optimal[arms_num-1]=0
            c_add_optimal[arms_num-1]=cons
            N_optimal[arms_num-1] =0
            X_hat[arms_num-1]=0
            c_add[arms_num-1]=cons
            c_sub[arms_num-1]=-cons
        
        if t+1<tau00:
            j=min(t,arms_num-1)
        elif flag>=0:
            flag=flag-1
            j=choose_list[flag]
        else:
            j1=arm_ind[np.argmax(c_add_optimal[arm_ind])]
            temp_nei=np.array(list(neighbor[j1]))
            j=temp_nei[np.argmax(c_add[temp_nei])]
        
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
        
        t += 1
      
    return reward, expect_reward, ch


@jit(nopython=True)
def Unkown_CUCB_BL(T,reward_mat,arm_type,neighbor_init,change_arms_list,seed):
    np.random.seed(seed)
    tau=int(20*np.sqrt(T))
    N=np.zeros(T,dtype='int64')
    X_hat=np.zeros(T)
    c=np.zeros(T)
    c_add_optimal=np.zeros(T)
    c_add=np.zeros(T)
    c_sub=np.zeros(T)
    N_optimal=np.zeros(T,dtype='int64')
    
    arm_ind=np.array([0])
    X_hat_optimal=np.zeros(T)
    
    reward = np.zeros(T)
    expect_reward = np.zeros(T)
    ch = np.zeros(T)
    neighbor=deepcopy(neighbor_init,len(neighbor_init))
    arms_num=0
    cons=np.sqrt(np.log(np.sqrt(2)*T))
    #for t in tqdm(range(T),desc="Double-UCB: "):
    t=0
    armsnum_temp=0
    flag=-1
    tau0=tau00
    while t<T:
    
        if (t+1)%tau0==0:
            #print("t= ",t)
            tt=arms_num-armsnum_temp  #(t//tau)*tau
            s_temp=np.ones(tt,dtype='int64')
            choose_list=list(arm_ind)
            for i in choose_list:
                neigh_temp1=set([ i_tmp for i_tmp in neighbor[i] if i_tmp>=armsnum_temp])
                #neigh_temp1=set(neighbor[i])
                for k1 in neigh_temp1:
                    if  s_temp[k1-armsnum_temp]!=0:
                        s_temp[k1-armsnum_temp]=0
            while sum(s_temp)!=0:
                index_temp=select_first_geq_zero(s_temp)
                choose_list.append(index_temp+armsnum_temp)
                s_temp[int(index_temp)]=0
                neigh_temp2=set([i_tmp for i_tmp in neighbor[index_temp+armsnum_temp] if i_tmp>=armsnum_temp])
                #neigh_temp2=set(neighbor[index_temp+tt])
                for k2 in neigh_temp2:
                    if s_temp[k2-armsnum_temp]!=0:
                        s_temp[k2-armsnum_temp]=0
            arm_ind=np.array(choose_list)
            flag=len(choose_list)
            armsnum_temp=arms_num
            
            if  tau0*2<tau:
                tau0=tau0*2
            else:
                tau0=tau
        
        if change_arms_list[t]==1:
            arms_num += 1
            N[arms_num-1]=0
            
            neigh1=neighbor[arms_num-1]
            for arm in set(neigh1):                  
                if (arms_num-1) not in neighbor[arm]:
                    neighbor[arm].append(arms_num-1)
                    
            X_hat_optimal[arms_num-1]=0
            c_add_optimal[arms_num-1]=cons
            N_optimal[arms_num-1] =0
            X_hat[arms_num-1]=0
            c_add[arms_num-1]=cons
            c_sub[arms_num-1]=-cons
            
        if t+1<tau00:
            j=min(t,arms_num-1)
        elif flag>=0:
            flag=flag-1
            j=choose_list[flag]
        else:
            j1=arm_ind[np.argmax(c_add_optimal[arm_ind])]
            temp_nei=np.array(list(neighbor[j1]))
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
        
        t += 1
    
           
      
    return reward, expect_reward, ch