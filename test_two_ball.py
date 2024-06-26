import numpy as np
import matplotlib.pyplot as plt

from scipy import stats as st
import statsmodels.stats.api as sms
import time

from aux_func import get_reward_distribution_ball
from param import param

from algo_new.Double_UCB_BL import Double_UCB_BL,C_UCB_BL


if __name__ == '__main__':

    T = param['T']
    K = param['K']
    epsilon=param['epsilon']
    iter = param['iter']
    arm_type=param['arm_type'] #0: Gaussian, 1: Bernoulli
    
    color=['m','c','g']
    color_index=0
    t1 = time.time()
    for epsilon in [0.01,0.02,0.05]:
        
        r1 = 0
        r2 = 0

        dpr1 = np.zeros((iter, T))
        dpr2 = np.zeros((iter, T))

        seed = 0
        
        for i in range(iter):
            print("iter = {}".format(i))
            reward_mat, r_opt, neighbor_init,change_arms_list = get_reward_distribution_ball(arm_type,T, K, epsilon, i+seed)
        
            print("arms num: {} ,epsilon = {}".format(K,epsilon))
            if arm_type==0:
                print("Gaussian arms")
            else:
                print("Bernoulli arms")
                
            rr1, e_r1, ch_p1 =  Double_UCB_BL(T, reward_mat, arm_type,neighbor_init, change_arms_list,i+seed)
            rr2, e_r2, ch_p2 = C_UCB_BL(T, reward_mat, arm_type,neighbor_init, change_arms_list,i+seed)
            
            dpr1[i, :] = (r_opt - e_r1).cumsum()
            dpr2[i, :] = (r_opt - e_r2).cumsum()
        
            r1 += e_r1
            r2 += e_r2
            
        
        r1 = r1 / iter
        r2 = r2 / iter
    
        cr1 = np.mean(dpr1, 0)
        cr2 = np.mean(dpr2, 0)
    
        print("Double-UCB: {}\nC-UCB: {}".format(cr1[-1],cr2[-1]))
        
        #plt.figure(2)
        xx = np.arange(0, T)
        d1 = int(param['T'] / 20)
        xx1 = np.arange(0, T, d1)
        
        d2=int(T/1000)
        xx2=np.arange(1,T,d2)

        alpha = 0.05
        #low_bound,high_bound=st.t.interval(0.95,T-1,loc=np.mean(dpr1,0),scale=st.sem(dpr1))
        low_bound, high_bound = sms.DescrStatsW(dpr1).tconfint_mean(alpha=alpha)
        plt.plot(xx1, cr1[xx1], '--'+color[color_index] , markerfacecolor='none', label=r'D-UCB $(\epsilon={})$'.format(epsilon))
        plt.fill_between(xx2, low_bound[xx2], high_bound[xx2], alpha=0.5)

        # low_bound, high_bound = st.t.interval(0.95, T - 1, loc=np.mean(dpr7, 0), scale=st.sem(dpr7))
        low_bound, high_bound = sms.DescrStatsW(dpr2).tconfint_mean(alpha=alpha)
        plt.plot(xx1, cr2[xx1], '-' + color[color_index]+ 'o', markerfacecolor='none', label=r'C-UCB $(\epsilon={})$'.format(epsilon))
        plt.fill_between(xx2, low_bound[xx2], high_bound[xx2], alpha=0.5)
    
        color_index += 1 
    t2 = time.time()
    print("time  = ", t2 - t1)
    plt.legend()
    #plt.title("T : {}, arms : {}".format(T, K))
    plt.xlabel("Rounds")
    plt.ylabel("Regret")
    plt.show()
