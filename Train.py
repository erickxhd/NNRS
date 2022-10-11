import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras

import Para

from myclass import ClassNN 
def train(env):
    if Para.Temp_if:
        out = env.m0.predict(input = env.data.omega)
        X_train = [env.data.omega, env.data.ev, out[0],out[1],out[2]]
        env.m1.model_int.fit(X_train, env.data.sigma,
                            epochs = Para.nb_epochs)   
        for epoch in range(Para.nb_update):
            print(epoch)
            out = env.m0.predict(input = env.data.omega)
            print(out[0].shape, out[1].shape, out[2].shape)
            """
            X_train = [env.data.omega, env.data.ev, env.data.sigma, out[0],out[1],out[2]]
            env.m1.model_loss.train_on_batch(X_train, env.data.sigma)
            env.m1.model_loss.evaluate(X_train, env.data.sigma)
            # model_loss
            X_train = [out[0],out[1],out[2]]
            env.m1.model.train_on_batch(X_train, env.data.rho)
            env.m1.model.evaluate(X_train, env.data.rho)
            """
            X_train = [env.data.omega, env.data.ev, out[0],out[1],out[2]]
            env.m1.model_int.train_on_batch(X_train, env.data.sigma)
            env.m1.model_int.evaluate(X_train, env.data.sigma)
            #for epoch in range(Para.nb_update):
            env.update(print_=True)
    else:
        """
        X_train = [env.data.omega,env.data.rho]
        env.m1.model_loss.fit(X_train, env.data.rho,
                            epochs = Para.nb_epochs)      
        """
        for epoch in range(Para.nb_epochs):
            print(epoch)
            X_train = [env.data.omega,env.data.rho]
            env.m1.model_loss.train_on_batch(X_train, env.data.rho)
            env.m1.model_loss.evaluate(X_train, env.data.rho)
            env.m1.model.evaluate(env.data.omega,env.data.rho)  

    #env.model.save('omega.h5')
    #env.model.evaluate(env.data.omega, env.data.rho)
    
from NN import get_int
def predict(env, figure = 1):
    if Para.Temp_if:
        delta1, delta2 = env.m0.get_delta()
        print(delta1, delta2)

        out = env.m0.predict(input = env.data.omega)
        X_train = [env.data.omega, env.data.ev, out[0],out[1],out[2]]
        result_sigma = env.m1.model_int.predict(X_train)
        X_train = [out[0],out[1],out[2]]
        result_rho = env.m1.model.predict(X_train)
        env.m1.model.evaluate(X_train, env.data.rho)
        result_rho1 = env.m1.model1.predict(env.data.omega)
        result_rho2 = env.m1.model2.predict(env.data.omega)
        result_rho3 = env.m1.model3.predict(env.data.omega)
        print_plot([result_rho1[0], result_rho2[0], result_rho3[0]], env.data.omega[0], figure = 0)

        print_plot3([result_rho[0]], env.data.omega[0], delta1, delta2, figure = 1)

        #print_plot3([env.data.rho[0]], env.data.omega[0], figure = 1)

        print_plot([result_sigma[0], env.data.sigma[0]], env.data.ev[0], figure = 2)

        np.savetxt('data1.txt', (np.array(env.data.omega[0])))  #x,y,z One dimensional array of the same size
        np.savetxt('data21.txt', (np.array(env.data.rho[0])))
        np.savetxt('data2.txt', (np.array(result_rho[0])))  #x,y,z One dimensional array of the same size
        np.savetxt('data3.txt', (np.array(env.data.ev[0])))  #x,y,z One dimensional array of the same size
        np.savetxt('data43.txt', (np.array(env.data.sigma[0])))
        np.savetxt('data4.txt', (np.array(result_sigma[0])))  #x,y,z One dimensional array of the same size
        plt.show()
        
    else:
        result_rho = env.m1.model.predict(env.data.omega)
        print_plot([result_rho[0], env.data.rho[0]], env.data.omega[0], figure = 1)
        plt.show()
    
import matplotlib.pyplot as plt 
def print_plot(List_data, X_train, figure = 1):
    plt.figure(figure)
    for list in List_data:
        plt.plot(X_train,list)
    #plt.show()
  
import numpy as np
def print_plot3(List_data, X_train, delta1 = -1, delta2 = 1, figure = 1):
    plt.figure(figure)
    X_train = np.array(X_train)
    for list in List_data:
        list0 = np.array(list)
        ind1 = X_train < delta1
        ind2 = np.logical_and(X_train > delta1, X_train < delta2)
        ind3 = X_train > delta2
        plt.plot(X_train[ind1],list0[ind1],'g')
        plt.plot(X_train[ind2],list0[ind2],'r')
        plt.plot(X_train[ind3],list0[ind3],'b')
    #plt.show()

if __name__ == '__main__':
    env = ClassNN()
    if True: # True False
        train(env)
    else:
        env.m1.model_loss.load_weights('omega_Ndata1000_Node100_Nlayer2_loss003.h5')
    predict(env)
