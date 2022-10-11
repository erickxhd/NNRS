#!/usr/bin/python
#coding:utf-8
# ==============================================================================
# Copyright 2022 All Rights Reserved.
# This open source code is uploaded to show the algorithm process of "Neural network replacing spectrum method", 
# For articles, please refer to: "https://arxiv.org/abs/2109.08861", 
# If you have any questions or suggestions, please contact me: xiehaidong@aliyun.com. 
# ==============================================================================
"""
Dependency libraries:
This code is writing in Python, 
dependence on tensorflow: https://github.com/tensorflow/tensorflow 
"""

# List of files of this code

1. Para.py
Set all parameters for calculation

2. Train.py
The main program. Realize training function
# If you intend to run the program, you can run this file directly

3. myclass.py
All variable type declarations

4. Data.py
Dataset construction. Contain: omega, d_omega, ev, d_ev, rho, sigma

5. NN.py
Model module, including neural network, integration, loss, etc
input:     omega:input1,    ev:input2,    rho:input3,    sigma:input4
Weights:    rho1:input1,    rho2:input1,    rho3:input1,    delta1,delta2,    rho:rho123
    int:rho,input2
    loss1:int,input4 For update rho123
    loss2:rho1,rho3  For update delta12

