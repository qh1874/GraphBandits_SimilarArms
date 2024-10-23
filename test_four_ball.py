import numpy as np
import matplotlib.pyplot as plt

from scipy import stats as st
import statsmodels.stats.api as sms
import time

from aux_func import get_reward_distribution_ball
from param import param

from algo_new.Double_UCB_BL import Double_UCB_BL,C_UCB_BL
from algo_new.Unkown_UCB_BL import Unkown_DUCB_BL,Unkown_CUCB_BL

import os
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2.0'


if __name__ == '__main__':

    T = param['T']
    K = param['K']
    epsilon=param['epsilon']
    iter = param['iter']
    arm_type=param['arm_type'] #0: Gaussian, 1: Bernoulli
    
    t1 = time.time()
  
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0

    dpr1 = np.zeros((iter, T))
    dpr2 = np.zeros((iter, T))
    dpr3 = np.zeros((iter, T))
    dpr4 = np.zeros((iter, T))

    seed = 0
    
    for i in range(iter):
        print("iter = {}".format(i))
        reward_mat, r_opt, neighbor_init,change_arms_list = get_reward_distribution_ball(arm_type,T, K, epsilon, i+seed)
    
        print("arms num: {}, epsilon = {}".format(K,epsilon))
        if arm_type==0:
            print("Gaussian arms")
        else:
            print("Bernoulli arms")
        
        rr3, e_r3, ch_p3 = Unkown_DUCB_BL(T, reward_mat, arm_type,neighbor_init, change_arms_list,i+seed)
        rr4, e_r4, ch_p4 = Unkown_CUCB_BL(T, reward_mat, arm_type,neighbor_init, change_arms_list,i+seed)
        rr1, e_r1, ch_p1,num_ind =  Double_UCB_BL(T, reward_mat, arm_type,neighbor_init, change_arms_list,i+seed)
        print("num_ind:",num_ind)
        rr2, e_r2, ch_p2 = C_UCB_BL(T, reward_mat, arm_type,neighbor_init, change_arms_list,i+seed)
        
        dpr1[i, :] = (r_opt - e_r1).cumsum()
        dpr2[i, :] = (r_opt - e_r2).cumsum()
        dpr3[i, :] = (r_opt - e_r3).cumsum()
        dpr4[i, :] = (r_opt - e_r4).cumsum()
    
        r1 += e_r1
        r2 += e_r2
        r3 += e_r3
        r4 += e_r4
        
    r1 = r1 / iter
    r2 = r2 / iter
    r3 = r3 / iter
    r4 = r4 / iter

    cr1 = np.mean(dpr1, 0)
    cr2 = np.mean(dpr2, 0)
    cr3 = np.mean(dpr3, 0)
    cr4 = np.mean(dpr4, 0)

    print("Double-UCB: {}\nC-UCB: {}\nU-DUCB: {}\nU-CUCB: {}".format(cr1[-1],cr2[-1],cr3[-1],cr4[-1]))
    
    #plt.figure(2)
    xx = np.arange(0, T)
    d1 = int(param['T'] / 20)
    xx1 = np.arange(0, T, d1)
    
    d2=int(T/1000)
    xx2=np.arange(1,T,d2)

    alpha = 0.05
    #low_bound,high_bound=st.t.interval(0.95,T-1,loc=np.mean(dpr1,0),scale=st.sem(dpr1))
    low_bound, high_bound = sms.DescrStatsW(dpr1).tconfint_mean(alpha=alpha)
    plt.plot(xx1, cr1[xx1], '-m^' , markerfacecolor='none', label='D-UCB')
    plt.fill_between(xx2, low_bound[xx2], high_bound[xx2], alpha=0.5)

    # low_bound, high_bound = st.t.interval(0.95, T - 1, loc=np.mean(dpr7, 0), scale=st.sem(dpr7))
    low_bound, high_bound = sms.DescrStatsW(dpr2).tconfint_mean(alpha=alpha)
    plt.plot(xx1, cr2[xx1], '-co', markerfacecolor='none', label='C-UCB')
    plt.fill_between(xx2, low_bound[xx2], high_bound[xx2], alpha=0.5)
    
    low_bound, high_bound = sms.DescrStatsW(dpr3).tconfint_mean(alpha=alpha)
    plt.plot(xx1, cr3[xx1], '-g*', markerfacecolor='none', label='U-DUCB')
    plt.fill_between(xx2, low_bound[xx2], high_bound[xx2], alpha=0.5)
    
    low_bound, high_bound = sms.DescrStatsW(dpr4).tconfint_mean(alpha=alpha)
    plt.plot(xx1, cr4[xx1], '-bd', markerfacecolor='none', label='U-CUCB')
    plt.fill_between(xx2, low_bound[xx2], high_bound[xx2], alpha=0.5)

    t2 = time.time()
    print("time  = ", t2 - t1)
    plt.legend()
    #plt.title("T : {}, arms : {}".format(T, K))
    plt.xlabel("Rounds")
    plt.ylabel("Regret")
    plt.show()
