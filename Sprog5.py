import numpy as np
X=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
X=X/np.amax(X,axis=0)
y=y/100

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return x*(1-x)

epoch=1

lr=0.1
inputlayer_neurons=2
hiddenlayer_neurons=3
output_neurons=1

wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
    
    hinp1=np.dot(X,wh)
    hinp=hinp1+bh
    hlayer_act=sigmoid(hinp)
    outinp1=np.dot(hlayer_act,wout)
    outinp=outinp1+bout
    output=sigmoid(outinp)
    
    EO=y-output
    outgrad=derivatives_sigmoid(output)
    d_output=EO*outgrad
    EH=np.dot(d_output,wout.T)
    
    hiddengrad=derivatives_sigmoid(hlayer_act)
    
    d_hiddenlayer=EH*hiddengrad
    
    wout+=np.dot(hlayer_act.T,d_output)*lr
    
    bout+=np.sum(d_output,axis=0)*lr
    wh+=X.T.dot(d_hiddenlayer)*lr
    bh+=np.sum(d_hiddenlayer,axis=0)*lr
print("Input:\n"+ str(X))
print("Actual output:\n"+ str(y))
print("predicted output:\n",output)
print("Final error in predicted output:\n",str(y-output))