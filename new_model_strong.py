# Imports
import numpy as np
import pandas as pd
import os
import pennylane as qml
import matplotlib.pyplot as plt
from dataset16 import load_dataset as ld_full
from dataset_muon import load_dataset as ld_muon
from functools import partial
import jax
from jax.example_libraries.optimizers import adam
import sklearn
from sklearn.metrics import roc_curve, roc_auc_score 

# ------ Constants ------#

SEED=0      
TRAIN_SIZE = 10000 
TEST_SIZE = 5000
N_QUBITS = 16   
LR=1e-2 
N_EPOCHS = 3000
BATCH_SIZE = 200

#------------------------#

# Definiton of the Pennylane device using JAX
device = qml.device("default.qubit.jax", wires=N_QUBITS,prng_key = jax.random.PRNGKey(SEED))

# Definition of the quantum circuit
# x : features from the jet structure
# w : weights of the model
# The qml.qnode decorator transforms the python function into a Pennylane QNode
# i.e. a circuit to be run on a specified device.
# The partial(jax.vmap) decorator creates a vectorized version of the function
# This way I can process multiple jets at one time, passing a vector of features.
# in_axes = [0,None] specifies that I want to vectorize the function with respect
# to the first parameter only (x), since I want the weights (w) to be the same for
# each jet.
@partial(jax.vmap,in_axes=[0,None]) # Vectorized version of the function
@qml.qnode(device,interface='jax')  # Create a Pennylane QNode
def Circuit(x,w):
  qml.AngleEmbedding(x,wires=range(N_QUBITS))   # Features x are embedded in rotation angles
  qml.StronglyEntanglingLayers(w,wires=range(N_QUBITS)) # Variational layer
  return qml.expval(qml.PauliZ(0)) # Expectation value of the \sigma_z operator on the 1st qubit

# Simple MSE loss function
def Loss(w,x,y):
  pred = Circuit(x,w)
  return jax.numpy.mean((pred - y) ** 2)

# Simple binary accuracy function
def Accuracy(w,x,y):
  pred = Circuit(x,w)
  return jax.numpy.mean(jax.numpy.sign(pred) == y)

# The ADAM optimizer is initialized
opt_init, opt_update, get_params = adam(LR)

# Training step
# This function is compiled Just-In-Time on the GPU
@jax.jit
def Train_Step(stepid, opt_state,train_f,train_t):
  current_w = get_params(opt_state)
  loss_value, grads = jax.value_and_grad(Loss,argnums=0)(current_w,train_f,train_t)
  acc_value = Accuracy(current_w,train_f,train_t)
  opt_state = opt_update(stepid, grads, opt_state)
  return loss_value,acc_value, opt_state

@jax.jit
def Test_Step(current_w,test_f,test_t):
  loss_value, grads = jax.value_and_grad(Loss,argnums=0)(current_w,test_f,test_t)
  acc_value = Accuracy(current_w,test_f,test_t)
  return loss_value, acc_value

def Batch_and_Shuffle(x,y):
  z = int(len(x) / BATCH_SIZE)
  data = np.column_stack([x,y])
  np.random.shuffle(data)
  return np.split(data[:,0:N_QUBITS],z), np.split(data[:,-1],z),z

def Train_Model(opt_state,x, y):
  loss_data = np.zeros(N_EPOCHS)
  acc_data = np.zeros(N_EPOCHS)
  print("Training...")
  print("Epoch\tLoss\tAccuracy")
  step=0
  for i in range(N_EPOCHS):
    
    # Batch and shuffle the data for ever epoch
    train_f, train_t, chunks = Batch_and_Shuffle(x, y)
    loss_temp = np.zeros(chunks)
    acc_temp = np.zeros(chunks)

    for j in range(chunks):
      loss_temp[j],acc_temp[j], opt_state = Train_Step(step, opt_state, train_f[j], train_t[j])
      step+=1

    loss_data[i] = np.average(loss_temp)
    acc_data[i] = np.average(acc_temp)

    if (i+1) % 100 == 0:
      print(f"{i+1}\t{loss_data[i]:.3f}\t{acc_data[i]*100:.2f}%")
      np.save("strong_w/strong_weights_epcoh_"+ str(i+1) +".npy", get_params(opt_state))
  
  weights = np.array(get_params(opt_state))
  file_weights = "strong_w/final_strong_weights_with_"+str(weights.shape[0])+".npy"
  np.save(file_weights, weights)

  return weights, loss_data, acc_data

def Test_Model(w, x, y):
  print("Testing...")  
  print("\tLoss\tAccuracy")
  # Batch and shuffle the data for ever epoch
  test_f, test_t, chunks = Batch_and_Shuffle(x, y)
  loss_temp = np.zeros(chunks)
  acc_temp = np.zeros(chunks)

  for j in range(chunks):
    loss_temp[j],acc_temp[j] = Test_Step(w, test_f[j], test_t[j])

  loss_data = np.average(loss_temp)
  acc_data = np.average(acc_temp)

  print(f"\t{loss_data:.3f}\t{acc_data*100:.2f}%")

  return loss_data, acc_data

def Plot_ROC(w,x,y,layer):
  depth = int(len(x) / BATCH_SIZE)
  new_x = np.split(x,depth)
  ps = np.array(TEST_SIZE)
  for i in range(depth):
    ps[i] = Circuit(new_x[i],w)
  predictions = np.reshape(ps, (ps.shape[0]*ps.shape[1], ps.shape[2])) # Convert 3D array to 2D array  
  fpr, tpr, threshold = roc_curve(y,predictions)
  auc = roc_auc_score(y,predictions)
  
  plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
  plt.plot(fpr,tpr,label="ROC QML,Strong(area = %0.2f)" % auc)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver Operating Characteristic")
  plt.legend(loc="lower right")
  fname = 'ROC_strong_training' +str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(fname)
  plt.clf()
  
  roc_d = {'FPR': fpr, 'TPR': tpr, 'Threshold': threshold, 'Area': auc}
  frame = pd.DataFrame(roc_d)
  file_name = 'strong_roc_data_with_'+str(layer)+'layers.csv'
  frame.to_csv(file_name, index=False)

def Plot_Loss_and_Acc(ep,loss,acc,layer):
  fig, ax1 = plt.subplots() 
  ax1.set_xlabel('# of Epochs') 
  ax1.set_ylabel('Loss', color = 'black') 
  plot_1 = ax1.plot(ep, loss, color = 'black') 
  ax1.tick_params(axis ='Loss', labelcolor = 'black')
  ax2 = ax1.twinx() 
  ax2.set_ylabel('Accuracy', color = 'green') 
  plot_2 = ax2.plot(ep, acc, color = 'green') 
  ax2.tick_params(axis ='Accuracy', labelcolor = 'green')
  plt.title("Strongly Entangling Layers("+str(layer)+") Architecture Loss and Accuracy")
  file_name = 'strong_full_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(file_name) 

def Run_Model():
  # Loads the dataset (already preprocessed... see dataset.py)
  train_features,train_target,test_features,test_target = ld_full(TRAIN_SIZE,TEST_SIZE,SEED)

  path = "/home/leonidas/example-qml4btag/strong_w"
  weights_files = os.scandir(path) # Get the .npy weight files
  with weights_files as entries:
    for entry in entries:
        print(entry.name)
  print("Please Choose a file from above to load the weights for the Strong model, otherwise press the space bar, then enter to pass this stage.")
  w_f = input("Enter file name:  ")
  if (w_f != " "):
    weights = np.load(path+"/"+w_f)
    test_loss, test_acc = Test_Model(weights, test_features, test_target)
    Plot_ROC(weights,test_features,test_target, 0)
  else:
    max_layers = 8
    train_loss_data = np.zeros([max_layers,TRAIN_SIZE,16])
    train_acc_data = np.zeros([max_layers,TRAIN_SIZE,16])
    test_loss = np.zeros([max_layers])
    test_acc = np.zeros([max_layers])
    for i in range(max_layers):
      # Weights are initialized randomly
      initial_weights = jax.random.uniform(jax.random.PRNGKey(SEED), ((i+1), N_QUBITS, 3))*jax.numpy.pi
      opt_state = opt_init(initial_weights)

      print("Training with "+str(i+1)+" layer(s)")
      weights, train_loss_data[i], train_acc_data[i] = Train_Model(opt_state, train_features, train_target)
      ep = np.linspace(1,N_EPOCHS,num=N_EPOCHS)
      Plot_Loss_and_Acc(ep,train_loss_data,train_acc_data,(i+1))

      test_loss[i], test_acc[i] = Test_Model(weights, test_features, test_target)
      Plot_ROC(weights,test_features,test_target,(i+1))

      d = {'Epochs': ep, 'Train Loss': train_loss_data[i], 'Train Accuracy':train_acc_data[i], 'Test Loss':test_loss[i], 'Test Accuracy':test_acc[i]}
      frame = pd.DataFrame(d)
      file_name = 'strong_loss_accuracy_data_with'+str(i+1)+'layers.csv'
      frame.to_csv(file_name, index=False)
  
