import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras

import Para
from Data import Datasets
#from NN import int_model, int_model2, get_delta
from NN import Model0, Model1, Model2, grad, diff

tf.compat.v1.disable_eager_execution()

class ClassNN:
    def __init__(self, Para = Para): #Define data format
        with tf.variable_scope('model'):
            self.sess = tf.Session()
        
            # Dataset
            self.data = Datasets() # omega,d_omega, ev,d_ev, rho,sigma


            # Model 0
            #input: omega, delta1,2
            #output: omega1,2,3
            self.m0 = Model0(Para = Para, sess = self.sess)
            #out = m0.predict(input = data.omega)
            
            # Model 1
            #imput: 
            #output:
            if Para.Temp_if:
                self.m1 = Model1(Para = Para)
            else:
                self.m1 = Model2(Para = Para)
                                           
            # set optimizers
            self.opt = keras.optimizers.Adam(lr = Para.learning_rate)
            self.loss0 = keras.losses.mean_squared_error
            self.loss = lambda y_true,y_pred: y_pred

            self.m1.model_loss.compile(loss = self.loss,
                                optimizer = self.opt)     
            self.m1.model_int.compile(loss = self.loss0,
                                optimizer = self.opt)   
            self.m1.model.compile(loss = self.loss0,
                                optimizer = self.opt)    

            self.grad = grad
            self.diff = diff
            self.sess.run(tf.global_variables_initializer())

    def __str__(self):
        return 'Construction graph finished:\n' + ' '.join(('%s' % item for item in self.__dict__.keys())) # items = keys + values

    def update(self, print_ = False):
        delta1, delta2 = self.m0.get_delta()
        #g1 = self.grad(self.m1.model1, delta1, self.sess)
        #g2 = self.grad(self.m1.model3, delta2, self.sess)
        g11 = self.diff(self.m1.model1, delta1, self.sess)
        g22 = self.diff(self.m1.model3, delta2, self.sess)
        if print_:
            #print(g1, g2)
            print(g11, g22)
        self.m0.update(g11, g22)
        if print_:
            print(self.m0.get_delta())

if __name__ == "__main__":
    env = ClassNN()
    print(env)
    #print(env.model.input)
    #print(env.model.output)
    #print(env.model_int.input)
    #print(env.model_int.output)
    print(env.m1.model_loss.output)
    out = env.m0.predict(input = env.data.omega)
