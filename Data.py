import numpy as np
from scipy.integrate import quad

import Para 
# All numpy vectorization transformation

# sequence
def my_list(Num = 100, alpha = 5, alpha1 = None):
    if alpha1 is None: # The data range is: -alpha to +alpha
        return np.linspace(-alpha,alpha,Num)
    else:              # The data range is: alpha1 to +alpha
        return np.linspace(alpha1,alpha,Num)

#Phy model ​Initial data
def rho(omega = 0, N_f = Para.N_f, Delta = Para.Delta, eps = Para.eps):
    # input numpy array 'omega'
    rho = N_f * np.abs(omega) / np.sqrt(np.abs(np.power(omega,2) - np.power(Delta,2)) + eps)
    flag = (omega < -Delta-eps) | (omega > Delta+eps)
    return np.where(flag, rho, 0)
    
#Temp result Temperature broadening function
def cosh(ev=0, omega=0, beta = Para.beta, a = Para.a, eps = Para.eps):
    # input float 'ev' and numpy array 'omega'
    return np.power(np.cosh(beta * 0.5 * (ev - omega)) + eps, -2) ## ++ --

# quad Function integral
def sigma(ev = 0, omega = None, a = Para.a):
    # input float 'ev'
    #y = quad(lambda omega:(cosh(ev, omega)), -np.inf, np.inf)/4 
    y = quad(lambda omega:(rho(omega) * cosh(ev, omega) * 0.25), -np.inf, np.inf)
    return a * y[0]

# trapz: numerical integration
def sigma0(ev = 0, omega = my_list(Num = 1000, alpha = 5), a = Para.a):
    # input float 'ev'
    r_c_ = rho(omega) * cosh(ev, omega) * 0.25
    y = np.trapz(r_c_, omega)
    return a * y

def get_list(func=lambda x:np.sin(x), omega=0, omega2 = None):
    if omega2 is None:
        return func(omega)
    else:
        return np.array([func(omega_, omega2) for omega_ in omega])

class Datasets:
    def __init__(self, Para = Para): 
        #Define data format
        self.omega = my_list(Num = Para.Num_omega, alpha = Para.alpha_omega)
        self.d_omega = Para.d_omega
        self.ev = my_list(Num = Para.Num, alpha = Para.alpha)
        self.d_ev = Para.d_ev
        self.rho = get_list(rho, self.omega)
        if Para.quad_trapz:
            self.sigma = get_list(sigma, self.ev, self.omega)
        else:
            self.sigma = get_list(sigma0, self.ev, self.omega)
        if Para.Noise:
            noise = np.random.normal(Para.mean,Para.sigma, size=(Para.Num))
            self.sigma = self.sigma + noise

        if Para.dataset == True:
            Para.Num_omega = 1171  #282
            self.d_ev = 0.01 #0.1019
            self.ev = np.loadtxt('exp2_data1.txt')
            self.sigma = np.loadtxt('exp2_data2.txt') * 1e12
            Para.Num = Para.Num_omega
            self.d_omega = self.d_ev *1.2
            self.omega = self.ev *1.2
            self.rho = self.sigma *1.2

        self.omega = np.reshape(self.omega, [1,Para.Num_omega,1])
        self.ev = np.reshape(self.ev, [1,Para.Num,1])
        self.rho = np.reshape(self.rho, [1,Para.Num_omega,1])
        self.sigma = np.reshape(self.sigma, [1,Para.Num,1])

    def __str__(self):
        return 'Construction dataset finished:\n' + ' '.join(('%s' % item for item in self.__dict__.keys())) # items = keys + values

import matplotlib.pyplot as plt
if __name__ == '__main__':
    data = Datasets()
    print(data)
    print('X_train: ', data.omega.shape)
    print('Y_train: ', data.ev.shape)
    print('X_test: ', data.rho.shape)
    print('Y_test: ', data.sigma.shape)

    plt.figure(1)
    plt.plot(data.omega[0],data.rho[0], label='rho')
    plt.plot(data.ev[0],data.sigma[0], label='sigma')
    #plt.ylim(0, 5)
    plt.show()

#Time test,
#Before vectorization:
#More than 10 times slower
#After vectorization:
#Num = 10000 data volume, time-consuming 9s
#Num = 1000, time < 1s