import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
from scipy.stats import truncnorm
from tqdm import tqdm
from numba import jit
from numba.typed import List
import numba

def generate_graph(K):
    g=nx.Graph()
    g.clear()
    g.add_nodes_from(range(K))
    neighbor=[[] for i in range(K)]
    for i in range(0,22,7):
        for ii in range(i,i+5):
            neighbor[ii].append(i+5)
            neighbor[ii].append(i+6)
            if ii==i:
                neighbor[ii].append(ii+1)
                neighbor[ii].append(ii+4)
            elif ii==i+4:
                neighbor[ii].append(ii-1)
                neighbor[ii].append(ii-4)
            else:
                neighbor[ii].append(ii+1)
                neighbor[ii].append(ii-1)
    for i in range(5,27,7):
        for ii in range(i-5,i):
            neighbor[i].append(ii)
            neighbor[i+1].append(ii)
            
        
    neighbor[6].append(12)
    neighbor[6].append(28)
    neighbor[29].append(12)
    neighbor[29].append(26)
    neighbor[28].append(20)
    neighbor[28].append(6)
    neighbor[12].append(6)
    neighbor[12].append(29)
    neighbor[20].append(26)
    neighbor[20].append(28)
    neighbor[26].append(29)
    neighbor[26].append(20)
    edges=[]
    for i in range(K):
        for ii in neighbor[i]:
            edges.append((i,ii))
    g.add_edges_from(edges)
    
    return g,neighbor
    
def generate_graph1(T,K,m):
    
    num=int(T/m)
    neighbor=[]
    for _ in range(num):
        neigh=[]
        for _ in range(K):
            x=np.random.randint(5)
            xlist=list(np.random.randint(0,K,x)) ##################
            neigh.append(xlist)
        for i in range(K):
            neigh[i].append(i)
            for j in range(K):
                if i in neigh[j]:
                    neigh[i].append(j)
        for i in range(K):
            neigh[i]=list(set(neigh[i]))
        neighbor.append(neigh)
        
    return neighbor
    
    
@jit(nopython=True)
def generate_data(arm_type,K,seed):
    np.random.seed(seed+1)
        
    reward_mat = np.zeros((K,2))
    if arm_type==0: #Gaussian
        reward_mat[:,1] = 1/2#np.random.uniform(0,1,K)
        reward_mat[:,0]=np.random.randn(K)
    elif arm_type==1: #Bernoulli
        reward_mat[:,1] = np.random.uniform(0,1,K)
        reward_mat[:,0]=np.random.uniform(0,1,K)
    else:   
        #### p(x)=2*x-x^2 F(x)=1-(1-x)^2  ####
        x1=np.random.uniform(0,1,K)
        reward_mat[:,0]=1-np.sqrt(1-x1) 
    
    
    return reward_mat


@jit(nopython=True)
def get_reward_distribution_ball(arm_type,T,K,ep,seed):
    
    np.random.seed(seed)
    change_arms_list=np.zeros(T,dtype='int64')
    #change_arms_list[0]=1
    #change_arms_list[0:10]=1
    change_arms_list[np.arange(0,T,int(T/K))]=1
    
    #generate optimal rewards
    Ksum=int(np.sum(change_arms_list))
    reward_mat=generate_data(arm_type,Ksum,seed)
    r_opt=np.zeros(T)
    arm_num_cumu=np.cumsum(change_arms_list)
    max_test=max(reward_mat[0:arm_num_cumu[0],0])
    r_opt[0]=max_test
    for ii in range(1,T):
        if arm_num_cumu[ii-1]<arm_num_cumu[ii]:    
            max_test=max(max_test,np.max(reward_mat[arm_num_cumu[ii-1]:arm_num_cumu[ii],0]))        
            r_opt[ii]=max_test
        else:
            r_opt[ii]=r_opt[ii-1]
    
    #r_opt=np.max(reward_mat[0:Ksum,0])
    #generate neighbors
    Ksum=np.sum(change_arms_list)
    #neighbor=[ [i] for i in range(Ksum)]
    #neighbor= List( List([i]) for i in range(Ksum) )
    neighbor=List()
    for i in range(Ksum):
        neighbor.append(List([i]))
    #neighbor[0].append(0)
    xr=reward_mat[:,0]
    #for i in tqdm(range(1,Ksum),desc="construct graph: "):
    for i in range(1,Ksum):
        x1=np.abs(xr[i]-xr[0:i])
        index=(x1<ep)*np.arange(1,i+1)
        for j in index:
            if j!=0:
                neighbor[i].append(j-1)
        #neighbor[i].append(i)
    
    
    # Ksum=np.sum(change_arms_list)
    # for i in range(Ksum):
    #     for j in range(Ksum):
    #         if i!=j and i in neighbor[j]:
    #             neighbor[i].append(j)
    # print("world")
    '''
    g=nx.Graph()
    g.clear()
    g.add_nodes_from(range(K))
    edges=[]
    for i in range(101):
        for ii in neighbor[i]:
            edges.append((i,ii))
    g.add_edges_from(edges)
    
    #g, neighbor =generate_graph(K)
    plt.figure(1)
    pos=nx.spring_layout(g)
    nx.draw(g,pos,with_labels=True)
    #nx.draw_circular(g,with_labels=True)
    plt.show()
    '''
    # for i in range(Ksum):
    #     neighbor[i]=list(set(neighbor[i]))
    

    return reward_mat,r_opt,neighbor,change_arms_list

@jit(nopython=True)
def get_reward_distribution(arm_type,T,K,ep,seed):
    
    np.random.seed(seed)
    change_arms_list=np.zeros(T,dtype='int32')
    change_arms_list[0]=K
    #generate optimal rewards
    reward_mat=generate_data(arm_type,K,seed)
    r_opt=np.zeros(T)
    r_opt=np.max(reward_mat[:,0])
    #generate neighbors
  
    #neighbor=[ [i] for i in range(Ksum)]
    #neighbor= List( List([i]) for i in range(Ksum) )
    neighbor=List()
    for i in range(K):
        neighbor.append(List([i]))
    #neighbor[0].append(0)
    xr=reward_mat[:,0]
    #for i in tqdm(range(1,Ksum),desc="construct graph: "):
    for i in range(1,K):
        x1=np.abs(xr[i]-xr[0:i])
        index=(x1<ep)*np.arange(1,i+1)
        for j in index:
            if j!=0:
                neighbor[i].append(j-1)
    
    # for i in range(K):
    #     for j in range(K):
    #         if i!=j and i in neighbor[j]:
    #             neighbor[i].append(j)
    
    '''
    g=nx.Graph()
    g.clear()
    g.add_nodes_from(range(K))
    edges=[]
    for i in range(101):
        for ii in neighbor[i]:
            edges.append((i,ii))
    g.add_edges_from(edges)
    
    #g, neighbor =generate_graph(K)
    plt.figure(1)
    pos=nx.spring_layout(g)
    nx.draw(g,pos,with_labels=True)
    #nx.draw_circular(g,with_labels=True)
    plt.show()
    '''
    # for i in range(Ksum):
    #     neighbor[i]=list(set(neighbor[i]))
    

    return reward_mat,r_opt,neighbor,change_arms_list

@jit(nopython=True)
def swap_rows(A, i, j):
    temp = A[i].copy()  # 临时存储一行
    A[i] = A[j]
    A[j] = temp
    return A

@jit(nopython=True)
def get_reward_distribution_native(arm_type,T,K,seed):
    
    np.random.seed(seed)
    change_arms_list=np.zeros(T,dtype='int32')
    change_arms_list[0]=K
    #generate optimal rewards
    reward_mat=generate_data(arm_type,K,seed)
    r_opt=np.zeros(T)
    r_opt=np.max(reward_mat[:,0])
    
    
    # 随机生成 K 对不同的行索引
    num_swaps = 100
    rows = np.random.choice(100, size=(num_swaps, 2), replace=True)

    # 执行行交换
    for i, j in rows:
        swap_rows(reward_mat,i,j)
    
    #generate neighbors
  
    # neighbor=List()
    # for i in range(K):
    #     neighbor.append(List([i]))
  
   
    # for i in range(1,K):
    #     index=np.arange(1,i+1)*(np.random.uniform(0,1,i)<p)
    #     for j in index:
    #         if j!=0:
    #             neighbor[i].append(j-1)
                
    return reward_mat,r_opt,change_arms_list

# @jit(nopython=True)
# def get_reward(r_mat,j,arm_type):
#     if arm_type==0:
#         n=len(j)
#         return r_mat[np.array(j),0] + r_mat[np.array(j),1]*np.random.randn(n)
#         #return np.random.normal(r_mat[np.array(j),0],r_mat[np.array(j),1])
#     else:
#         return (np.random.uniform(0,1,len(j))<r_mat[np.array(j),0])*np.ones(len(j))

@jit(nopython=True)
def get_reward(r_mat,j,arm_type):
    if arm_type==0:
        n=len(j)
        return r_mat[j,0] + r_mat[j,1]*np.random.randn(n)
        #return np.random.normal(r_mat[j,0],r_mat[j,1])
    else:
        return (np.random.uniform(0,1,len(j))<r_mat[j,0])*np.ones(len(j))
   
    
@jit(nopython=True)
def deepcopy(A_list,n):
    B_list=List()
    for i in range(n):
        B_list.append(List.empty_list(numba.types.int32))
        m=len(A_list[i])
        for j in range(m): 
            B_list[i].append(A_list[i][j])
    return B_list

def get_reward0(r_mat,j):
   
    #return trunc(np.random.normal(r_mat[t,j,0],r_mat[t,j,1]))
    return 1 if np.random.uniform()<r_mat[j,0] else 0
   