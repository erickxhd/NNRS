import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
import numpy as np

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
#Activation = keras.layers.Activation

import Para

tf.compat.v1.disable_eager_execution()

class Model0:
    def __init__(self, Para = Para, sess = None):
        # Default breakpoint initial value
        self.alpha1 = -10.0 #-Para.alpha
        self.alpha2 = 10.0 #Para.alpha
        self.shift = 0.0 #Para.alpha/2
        # input shape
        self.input_shape = (None, 1)
        self.inputs = keras.layers.Input(shape=self.input_shape, name='omega_input')
        # Init delta
        self.delta1 = tf.Variable(self.alpha1 + self.shift, name = 'delta1')
        self.delta2 = tf.Variable(self.alpha2 - self.shift, name = 'delta2')
        # Cal output
        self.output = self.cut3()
        # self.o1,self.o2,self.o3 = self.output # 3 parts
        if sess == None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def cut3(self):
        inp = self.inputs
        
        d1 = self.delta1 * tf.ones_like(inp, dtype = tf.float32)
        d2 = self.delta2 * tf.ones_like(inp, dtype = tf.float32)
        # find indix
        inputs1 = tf.where(inp <= d1)
        inputs2 = tf.where(tf.logical_and(inp > d1, inp < d2))
        inputs3 = tf.where(inp >= d2)
        # Select subsequence
        inputs11 = tf.gather_nd(inp,inputs1)
        inputs22 = tf.gather_nd(inp,inputs2)
        inputs33 = tf.gather_nd(inp,inputs3)
        # Restore shape
        inputs111 = tf.reshape(inputs11,[1,-1,1])
        inputs222 = tf.reshape(inputs22,[1,-1,1])
        inputs333 = tf.reshape(inputs33,[1,-1,1])
        return inputs111, inputs222, inputs333

    def predict(self, input = None):
        print(self.get_delta())
        return self.sess.run(self.output, feed_dict={self.inputs:input})

    def get_delta(self):
        delta1 = self.sess.run(self.delta1)
        delta2 = self.sess.run(self.delta2)
        delta1 = np.reshape(delta1, [1])
        delta2 = np.reshape(delta2, [1])
        return delta1, delta2

    def update(self, g1, g2):
        # Approaching the maximum value of delta
        lr = 1
        self.delta1 = tf.stop_gradient(self.delta1 + lr * g1)
        self.delta2 = tf.stop_gradient(self.delta2 + lr * g2)

def grad(model, input, sess):
    input1 = tf.reshape(input, [1,1,1])
    Input1 = sess.run(input1)
    print(Input1)
    #output = model(input1)
    output = model.output
    print(sess.run(output, feed_dict={model.input:Input1}))
    print(model.predict(input1, batch_size = 1))
    grad = tf.gradients(output, model.input) # stop_gradients=[input1]
    Grad = sess.run(grad, feed_dict={model.input:Input1})
    Grad = np.reshape(Grad, [1])
    return Grad

def diff(model, input, sess):
    input1 = np.reshape(input, [1,1,1])
    input2 = input1 + 1e-3
    output1 = model.predict(input1, batch_size = 1)
    output2 = model.predict(input2, batch_size = 1)
    diff = (output2 - output1)/(input2 - input1)
    return diff

#Full connection n-layer construction
def layers(inputs = keras.layers.Input(shape=(1)), nb_classes= 1, Para = None, name = None):
    if Para is not None:
        node = Para.node
        N_layers = Para.N_layers
        activation2 = Para.activation2
    else:
        node = 10
        N_layers = 1
        activation2 = None

    for i in range(N_layers):
        if i == 0:
            rho = Dense(node, activation ='relu')(inputs)
        else:
            rho = Dense(node, activation ='relu')(rho)

    if name is None:
        rho = Dense(nb_classes, activation =activation2)(rho) 
    else:
        rho = Dense(nb_classes, activation =activation2, name=name)(rho)
    return rho

def model_layers(inputs, nb_classes= 1, Para = None, name = None):
    outputs = layers(inputs = inputs, nb_classes = nb_classes, Para = Para, name = name)
    outputs = tf.reshape(outputs,[1,-1,1]) # restore shape
    if name is None:
        return keras.models.Model(inputs=inputs, outputs=outputs)
    else:
        return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

class Model1:
    def __init__(self, Para = Para):
        self.input_shape = (None, 1)
        self.output_shape0 = (1)
        #The model inputs must be defined separately, otherwise, two definitions are equal to one
        self.inputs = keras.layers.Input(shape=self.input_shape, name='omega_input') # model_int,
        self.inputs_ev = keras.layers.Input(shape=self.input_shape, name='ev_input') # model_int,
        self.inputs_sigma = keras.layers.Input(shape=self.input_shape, name='sigma_input') # model_loss

        self.inputs1 = keras.layers.Input(shape=self.input_shape, name='omega_input1')
        self.model1 = model_layers(inputs=self.inputs1,nb_classes=self.output_shape0,Para = Para)
        
        self.inputs2 = keras.layers.Input(shape=self.input_shape, name='omega_input2')
        self.model2 = model_layers(inputs=self.inputs2,nb_classes=self.output_shape0,Para = Para)

        self.inputs3 = keras.layers.Input(shape=self.input_shape, name='omega_input3')
        self.model3 = model_layers(inputs=self.inputs3,nb_classes=self.output_shape0,Para = Para)

        self.rho = keras.layers.concatenate([self.model1.output,self.model2.output,self.model3.output], axis=1)
        input_model = [self.inputs1,self.inputs2,self.inputs3]
        self.model = keras.models.Model(inputs=input_model, outputs=self.rho)

        self.the_int = get_int(inputs=self.inputs, inputs2=self.inputs_ev, rho=self.rho)
        self.the_int = tf.reshape(self.the_int,[1,-1,1]) # Restore the original shape to ensure successful model construction
        input_model_int = [self.inputs,self.inputs_ev, self.inputs1,self.inputs2,self.inputs3]
        self.model_int = keras.models.Model(inputs=input_model_int, outputs=self.the_int)

        self.inputs_sigma0 = tf.reshape(self.inputs_sigma,[-1]) # Change to 1D
        self.the_int = tf.reshape(self.the_int,[-1]) # Change to 1D
        self.loss = keras.losses.mean_squared_error(self.the_int, self.inputs_sigma)
        self.loss = tf.reshape(self.loss, shape =[1,-1,1]) # 1 number should be output
        inputs_model_loss=[self.inputs,self.inputs_ev,self.inputs_sigma,
                        self.inputs1,self.inputs2,self.inputs3]
        self.model_loss = keras.models.Model(inputs=inputs_model_loss, outputs=self.loss)

class Model2:
    def __init__(self, Para = Para):
        self.input_shape = (None, 1)
        self.output_shape0 = (1)
        self.inputs = keras.layers.Input(shape=self.input_shape, name='omega_input')
        self.inputs_rho = keras.layers.Input(shape=self.input_shape, name='rho_input')

        self.model = model_layers(inputs=self.inputs,nb_classes=self.output_shape0,Para = Para)

        self.inputs_rho0 = tf.reshape(self.inputs_rho,[-1,1]) # Change to 1D
        self.model_output0 = tf.reshape(self.model.output,[-1,1]) # Change to 1D
        self.loss = keras.losses.mean_squared_error(self.model_output0, self.inputs_rho0)
        self.loss = tf.reshape(self.loss, shape =[1,-1,1]) # 1 number should be output
        inputs_model_loss=[self.inputs,self.inputs_rho]
        self.model_loss = keras.models.Model(inputs=inputs_model_loss, outputs=self.loss)

###################################################
#Temperature broadening function
def cosh(ev, omega, beta = Para.beta, eps = Para.eps):
    return tf.pow(tf.cosh(beta * 0.5 * (ev - omega)) + eps, (-2))  ## ++ --

def my_int(rho, ev, omega, d_omega = Para.d_omega, a = Para.a): 
    quad = tf.reduce_sum(tf.multiply(rho, cosh(ev,omega)), axis = 1, keepdims=True) # reduce_mean
    return a * 0.25 * quad * d_omega  # Integration interval / Para.Num

def get_int(inputs,inputs2,rho):
    # Dimension 1 * n * 1 will be automatically processed as one dimension without additional operation
    inputs = tf.reshape(inputs,[-1]) # Change to 1D
    inputs2 = tf.reshape(inputs2,[-1]) # Change to 1D
    rho = tf.reshape(rho,[-1]) # Change to 1D
    # omega & ev to meshgrid
    Omega, Ev= tf.meshgrid(inputs,inputs2)
    Rho, Ev= tf.meshgrid(rho,inputs2)
    # Rho & mesh to int
    return my_int(Rho, Ev, Omega, d_omega = Para.d_omega, a = Para.a) # shape: 1-Dim

"""

#######################################################################
def loss_2(model):

    #input=tf.constant([-Para.Delta-Para.eps, Para.Delta+Para.eps], shape=[2, 1])
    #output = model(input) + 0.01

    Delta = model.get_layer('delta')
    Delta = Delta.output * tf.pow(tf.cosh( model.input ) , (-2))
    rho1 = model.get_layer('rho1')
    rho2 = model.get_layer('rho2')
    rho3 = model.get_layer('rho3')
    #rho2_ = rho2.output * 0 #tf.constant(0.0)
    rho13 = tf.concat([rho1.output, rho2.output * 0, rho3.output], 1)
    predictions = tf.reduce_sum(tf.multiply(Delta,rho13), 1, keepdims=True)
    loss1 = 1 / tf.reduce_sum(predictions) * Para.Num

    rho22 = tf.concat([rho1.output * 0, rho2.output, rho3.output * 0], 1)
    predictions = tf.reduce_sum(tf.multiply(Delta,rho22), 1, keepdims=True)
    loss2 = tf.reduce_sum(predictions) / Para.Num

    return 10.00 * (loss1 + loss2)
"""
import sys
from Data import Datasets
if __name__ == '__main__':
    data = Datasets()
    m0 = Model0(Para = Para)
    out = m0.predict(input = data.omega)
    print(out[0].shape, out[1].shape, out[2].shape)
    #print(out)
    delta1, delta2 = m0.get_delta()
    print(delta1, delta2)
    m1 = Model1(Para = Para)
    #m1.model1.summary()
    
    out1 = m1.model1.predict(out[0], batch_size = max(Para.Num, Para.Num_omega))
    #print(out1)
    out2 = m1.model2.predict(out[1], batch_size = max(Para.Num, Para.Num_omega))
    #print(out2)
    out3 = m1.model3.predict(out[2], batch_size = max(Para.Num, Para.Num_omega))
    #print(out3)
    print(out1.shape,out2.shape,out3.shape)
    print(m1.model_int.input)
    print(m1.model_int.output)
    out_int = m1.model_int.predict([data.omega,data.ev,out[0],out[1],out[2]])
    #print(out_int)

    print(m1.model_loss.input)
    print(m1.model_loss.output)
    loss = m1.model_loss.predict([data.omega,data.ev,data.sigma,out[0],out[1],out[2]])
    print(loss.shape)
    