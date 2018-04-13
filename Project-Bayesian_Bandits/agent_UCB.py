from agent import agent
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt


class Agent_UCB(agent):
    # Basic UCB algorithm

    def __init__(self,K):
        self.N_arms=K # Nb of arms
        self.cumReward = np.zeros(self.N_arms) # Cum reward
        self.I_hat = np.zeros(self.N_arms) # Cum nb of pulls for each arm
        self.Mu_hat=np.zeros(self.N_arms) # Estimated mu in the UCB algorithm


    # Agent strategy is based on the frequentist UCB
    def Action(self,**kwargs):
        
        alpha = kwargs['alpha'] # Paramter of the algotihm, alpha>=2
        
        # Go through all arms ones, before start estimating the UCB
        if 0 in self.I_hat:
            I_t = np.argmin(self.I_hat)
            self.UCB = np.ones(self.N_arms)*np.inf
        else:
            # Estimate the UCB and get the best action
            t=self.I_hat.sum()
            self.UCB = self.Mu_hat + np.sqrt(alpha*np.log(t+1)/(2*self.I_hat))
            I_t=np.argmax(self.UCB)
            
        return I_t
        
    # Agent Update it's belief in the frequentist UCB approach after taking action I_t and getting reward R 
    def Update(self,**kwargs):
        
        I_t = kwargs['I_t']
        R = kwargs['R']
        
        self.Mu_hat[I_t]=(self.Mu_hat[I_t]*self.I_hat[I_t] + R)/(self.I_hat[I_t]+1)
        self.I_hat[I_t]+=1
        self.cumReward[I_t] += R
         
    # Useful plots
    def plot_arm(self,K):
        plt.plot([self.Mu_hat[K],self.Mu_hat[K]],[0,1],label="Mu_hat - Arm"+str(K))
        plt.plot([self.UCB[K],self.UCB[K]],[0,1],'--',label="UCB - Arm"+str(K))
        plt.legend()
    