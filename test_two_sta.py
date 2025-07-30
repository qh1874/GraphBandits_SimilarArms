import numpy as np
import matplotlib.pyplot as plt

from scipy import stats as st
import statsmodels.stats.api as sms
import time

from aux_func import get_reward_distribution,get_reward_distribution_native
from param import param

from algo_new.UCB_N import UCB_N
from algo_new.Double_UCB import Double_UCB


if __name__ == '__main__':

    T = param['T']
    K = param['K']
    epsilon=param['epsilon']
    iter = param['iter']
    arm_type=param['arm_type'] #0: Gaussian, 1: Bernoulli
    
    #color=['m','c','g']
    color=['m','k','b']
    color_index=0
    saved=True
    if arm_type == 1:
        epsilon_list= [0.02,0.05,0.1]
    else:
        epsilon_list= [0.05,0.1,0.2]
    t1 = time.time()
    plt.figure(figsize=(8,6))
    for epsilon in epsilon_list:
        
        r1 = 0
        r2 = 0

        dpr1 = np.zeros((iter, T))
        dpr2 = np.zeros((iter, T))

        seed = 0
        if saved==False:
            for i in range(iter):
                print("iter = {}".format(i))
                reward_mat, r_opt, neighbor_init,change_arms_list = get_reward_distribution(arm_type,T, K, epsilon, i+seed,0)

                print("arms num: {} ,epsilon = {}".format(K,epsilon))
                if arm_type==0:
                    print("Gaussian arms")
                    p=np.sqrt(2)*(2*st.norm.cdf(epsilon/np.sqrt(2))-1)
                else:
                    print("Bernoulli arms")
                    p=1-(1-epsilon)**2
                if i==0:
                    _,_,_,ind_num=Double_UCB(T, reward_mat, arm_type,neighbor_init, change_arms_list,i+seed)
                    print("ind_num = ", ind_num)
                rr1, e_r1, ch_p1 =  UCB_N(T, reward_mat, arm_type,neighbor_init, change_arms_list,i+seed)
                
                reward_mat1, r_opt,change_arms_list = get_reward_distribution_native(arm_type,T, K, i+seed,0)
                #reward_mat1, r_opt,neighbor_init,change_arms_list = get_reward_distribution_native(arm_type,T, p,K, i+seed)
                # if i==0:
                #     _,_,_,ind_num=Double_UCB(T, reward_mat1, arm_type,neighbor_init, change_arms_list,i+seed)
                #     print("ind_num1 = ", ind_num)
                rr2, e_r2, ch_p2 = UCB_N(T, reward_mat1, arm_type,neighbor_init, change_arms_list,i+seed)
               
                
                dpr1[i, :] = (r_opt - e_r1).cumsum()
                dpr2[i, :] = (r_opt - e_r2).cumsum()
            
                r1 += e_r1
                r2 += e_r2
                
            r1 = r1 / iter
            r2 = r2 / iter
        
            cr1 = np.mean(dpr1, 0)
            cr2 = np.mean(dpr2, 0)
        
            print("UCB-N-native: {}\nUCB-N-similar: {}".format(
                cr2[-1],cr1[-1]))
        
        #save data
        if saved==False:
            np.save("data_test_two_sta/cr1_eps_{}_armtype_{}.npy".format(epsilon,arm_type),cr1)
            np.save("data_test_two_sta/cr2_eps_{}_armtype_{}.npy".format(epsilon,arm_type),cr2)
            np.save("data_test_two_sta/dpr1_eps_{}_armtype_{}.npy".format(epsilon,arm_type),dpr1)
            np.save("data_test_two_sta/dpr2_eps_{}_armtype_{}.npy".format(epsilon,arm_type),dpr2)
        #load data
        if saved==True:
            cr1=np.load("data_test_two_sta/cr1_eps_{}_armtype_{}.npy".format(epsilon,arm_type))
            cr2=np.load("data_test_two_sta/cr2_eps_{}_armtype_{}.npy".format(epsilon,arm_type))
            dpr1=np.load("data_test_two_sta/dpr1_eps_{}_armtype_{}.npy".format(epsilon,arm_type))
            dpr2=np.load("data_test_two_sta/dpr2_eps_{}_armtype_{}.npy".format(epsilon,arm_type))
        
       
        xx = np.arange(0, T)
        d = int(param['T'] / 20)
        xx1 = np.arange(0, T, d)

        d2=int(T/1000)
        xx2=np.arange(1,T,d2)
        
        alpha = 0.05
        #low_bound,high_bound=st.t.interval(0.95,T-1,loc=np.mean(dpr1,0),scale=st.sem(dpr1))
        low_bound, high_bound = sms.DescrStatsW(dpr1).tconfint_mean(alpha=alpha)
        plt.plot(xx1, cr1[xx1], '-'+color[color_index] + '*', markersize = 8, markerfacecolor='none', label=r'UCB-N $(\epsilon={})$'.format(epsilon))
        plt.fill_between(xx2, low_bound[xx2], high_bound[xx2], alpha=0.5)

        # low_bound, high_bound = st.t.interval(0.95, T - 1, loc=np.mean(dpr7, 0), scale=st.sem(dpr7))
        low_bound, high_bound = sms.DescrStatsW(dpr2).tconfint_mean(alpha=alpha)
        plt.plot(xx1, cr2[xx1], '-' + color[color_index]+ 'o',markersize = 8, markerfacecolor='none', label='UCB-N-Standard')
        plt.fill_between(xx2, low_bound[xx2], high_bound[xx2], alpha=0.5)
    
        color_index += 1 
    t2 = time.time()
    print("time  = ", t2 - t1)
    plt.legend(fontsize=10)
    #plt.title("T : {}, arms : {}".format(T, K))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.tick_params(axis='both', which='major', labelsize=16)
    plt.xlabel("Rounds",fontsize=16)
    plt.ylabel("Regret",fontsize=16)
    plt.show()
