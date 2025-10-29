import numpy as np
from aux_func import get_reward,deepcopy,neighbor_to_adj_symmetric
from tqdm import tqdm
from numba import jit,njit,int32,boolean

# @jit(nopython=True)
# def is_clique(neighbor,neigh):
    
#     for i in neigh:
#         neigh_i = neighbor[i]
#         for k in neigh:
#             if k not in neigh_i:
#                 return False
#     return True

# @jit(nopython=True)
# def find_Ijt(neighbor,Njt):
#     flag=False
#     for i in Njt:
#         for j in Njt:
#             if i not in neighbor[j]:
#                 flag=True
#                 neigh1=set(neighbor[i]).intersection(set(Njt))
#                 neigh2=set(neighbor[j]).intersection(set(Njt))
#                 if is_clique(neighbor,np.array(list(neigh1))) and is_clique(neighbor,np.array(list(neigh2))):
#                     return np.array([i,j])
#     if flag == False:
#         print("Error")
#     #return np.array([i,j])
    

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
    cons=np.sqrt(np.log(np.sqrt(2)*T**2))
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
    cons=np.sqrt(np.log(np.sqrt(2)*T**2))
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
def is_clique_adj(adj, neigh):
    # neigh: int32 array of vertex indices
    m = len(neigh)
    for a in range(m):
        i = neigh[a]
        for b in range(m):
            k = neigh[b]
            if not adj[i, k]:
                return False
    return True

@jit(nopython=True)
def filter_neighbors_in_N(adj, v, Njt):
    # 返回 v 在 Njt 中的邻居列表（包括 v 自身）
    # 为了 numba 友好，先用最大长度申请，再切片返回
    m = len(Njt)
    buf = np.empty(m, dtype=np.int32)
    cnt = 0
    for t in range(m):
        u = Njt[t]
        if adj[v, u]:
            buf[cnt] = u
            cnt += 1
    return buf[:cnt]

@jit(nopython=True)
def find_Ijt_adj(adj, Njt):
    m = len(Njt)
    found_non_edge = False

    for a in range(m):
        i = Njt[a]
        for b in range(m):
            j = Njt[b]
            # 注意：如果 Njt 可能是完全图，这里永远 True；我们需要找一对“不相邻”
            if not adj[i, j]:
                found_non_edge = True

                neigh1 = filter_neighbors_in_N(adj, i, Njt)
                neigh2 = filter_neighbors_in_N(adj, j, Njt)

                if is_clique_adj(adj, neigh1) and is_clique_adj(adj, neigh2):
                    out = np.empty(2, dtype=np.int32)
                    out[0] = i
                    out[1] = j
                    return out

    if not found_non_edge:
        # Njt 本身是完全图
        # 为与原逻辑一致，这里不返回对；可以返回一个空数组或用约定值
        # 你要 print 也可以，但 njit 下 print 可用但不建议频繁
        # print("Error")
        return np.empty(0, dtype=np.int32)

    # 找到了不相邻对，但不满足“邻域交 Njt 是完全图”的约束
    return np.empty(0, dtype=np.int32)

@jit(nopython=True)
def C_UCB_WithGraph(T,reward_mat,arm_type,neighbor_init,change_arms_list,adj,seed):
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
    cons=np.sqrt(np.log(np.sqrt(2)*T**2))
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
            #adj=neighbor_to_adj_symmetric(neighbor,num)
            
        if t< len(arm_ind):
            j=arm_ind[t]
        else:
            j1=arm_ind[np.argmax(c_add_optimal[arm_ind])]
            isClique = is_clique_adj(adj,neighbor[j1])
            if isClique:
                temp_nei=np.array(list(neighbor[j1]))
                j=temp_nei[np.argmax(c_sub[temp_nei])]
            else:
                Ijt = find_Ijt_adj(adj,neighbor[j1])
                j2 = Ijt[np.argmax(c_add_optimal[Ijt])]
                #temp_set = set(neighbor[j2]).intersection(set(neighbor[j1]))
                temp_neigh = filter_neighbors_in_N(adj,j2,neighbor[j1])
                temp_nei=np.array(list(temp_neigh))
                j=temp_neigh[np.argmax(c_sub[temp_neigh])]
          
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