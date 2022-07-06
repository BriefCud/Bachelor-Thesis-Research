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
TRAIN_SIZE = 300*400 
TEST_SIZE = 300*500
N_QUBITS = 16   
LR=1e-3 
N_EPOCHS = 200
BATCH_SIZE = 300

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
def Test_Step(w,test_f,test_t):
  loss_value = Loss(w,test_f,test_t)
  acc_value = Accuracy(w,test_f,test_t)
  return loss_value, acc_value

def Batch_and_Shuffle(x,y):
  z = int(len(x) / BATCH_SIZE)
  data = np.column_stack([x,y])
  np.random.shuffle(data)
  return np.split(data[:,0:N_QUBITS],z), np.split(data[:,-1],z),z

def Plot_ROC(w,x,y,layer):
  depth = int(len(x) / BATCH_SIZE)
  new_x = np.split(x,depth)
  ps = np.array(depth,BATCH_SIZE)
  for i in range(depth):
    ps[i] = Circuit(new_x[i],w)
  predictions = np.reshape(ps, (ps.shape[0]*ps.shape[1])) # Convert 2D array to 1D array  
  fpr, tpr, threshold = roc_curve(y,predictions)
  auc = roc_auc_score(y,predictions)
  df_auc = np.ones(len(fpr))*auc
  
  # Get data predictions from the XGBoost to compare ROC curves
  xgb_csv =  pd.read_csv('/data/test_withxgb.csv')
  xgb_pred = xgb_csv['XGB_PRED'] 
  xgb_target = xgb_csv['Jet_LABEL']*2-1
  xgb_fpr,xgb_tpr,xgb_threshold = roc_curve(xgb_target,xgb_pred)
  xgb_auc = roc_auc_score(xgb_target,xgb_pred)
  
  plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
  plt.plot(fpr,tpr,label="QML,Strong(area = %0.2f)" % auc)
  plt.plot(xgb_fpr,xgb_tpr,label="XGBoost(area = %0.2f)" % xgb_auc)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver Operating Characteristic")
  plt.legend(loc="lower right")
  fname = 'ROC_strong_training' +str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(fname)
  plt.clf()
  
  roc_d = {'FPR': fpr, 'TPR': tpr, 'Threshold': threshold, 'Area': df_auc}
  frame = pd.DataFrame(roc_d)
  file_name = 'strong_data/strong_roc_data_'+str(layer)+'layers_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.csv'
  frame.to_csv(file_name, index=False)
  
  # Plot the distribution
  pb = predictions[y==1]
  pb_bar = predictions[y==-1]
  plt.hist(pb,bins=np.linspace(-1, 1, 100),alpha=0.5,label='Pb')
  plt.hist(pb_bar,bins=np.linspace(-1, 1, 100),alpha=0.5,label='Pb-bar')
  plt.xlim([-1,1])
  plt.legend(loc='upper right')
  fname = 'strong_data/strong_prob_dist_layer'+str(layer)+'_training' +str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(fname)
  plt.clf()

def Plot_Loss_and_Acc(ep,loss,acc,title,file_name,xlabel):
  figure, axis = plt.subplots(2, 1)

  axis[0].plot(ep, acc)
  axis[0].set_xlabel(xlabel, size=14)
  axis[0].set_ylabel('Accuracy', size=14)

  axis[1].plot(ep,loss)
  axis[1].set_xlabel(xlabel, size=14)
  axis[1].set_ylabel('Loss', size=14)

  figure.suptitle(title)
  plt.figure(figsize=(10,8))
  plt.savefig(file_name)
  plt.clf()

def Train_Model(opt_state,x, y,layer):
  z = int(len(x) / BATCH_SIZE)
  loss_step_data = np.zeros(N_EPOCHS*z)
  acc_step_data = np.zeros(N_EPOCHS*z)
  loss_epoch_data = np.zeros(N_EPOCHS)
  acc_epoch_data = np.zeros(N_EPOCHS)
  print("Training...")
  print("Epoch\tLoss\tAccuracy")
  step=0
  for i in range(N_EPOCHS):
    
    # Batch and shuffle the data for ever epoch
    train_f, train_t, chunks = Batch_and_Shuffle(x, y)

    for j in range(chunks):
      loss_step_data[step],acc_step_data[step], opt_state = Train_Step(step, opt_state, train_f[j], train_t[j])
      step+=1
    
    loss_epoch_data[i] = np.mean(loss_step_data[step-chunks:step]) # get the mean of the loss for the chunk size
    acc_epoch_data[i] = np.mean(acc_step_data[step-chunks:step])

    if (i+1) % 100 == 0:
      print(f"{i+1}\t{loss_epoch_data[i]:.3f}\t{acc_epoch_data[i]*100:.2f}%")
  
  fname = "strong_w/final_strong_weights_layer'+str(layer)+'_training"+str(TRAIN_SIZE)+"_testing"+str(TEST_SIZE)+".npy"
  np.save(fname, get_params(opt_state))
  
  title = 'Accuracy and Loss vs Steps for MPS'
  fname = 'strong_data/strong_step_acc_loss_layer'+str(layer)+'_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  Plot_Loss_and_Acc(np.linspace(1,N_EPOCHS*z,num=N_EPOCHS*z),loss_step_data,acc_step_data,title,fname,'Step')
  
  d = {'Steps': np.linspace(1,step,step), 'Train Loss': loss_step_data[i], 'Train Accuracy':acc_step_data[i]}
  frame = pd.DataFrame(d)
  fname = 'strong_data/strong_step_loss_accuracy_data_with'+str(layer)+'layers_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.csv'
  frame.to_csv(fname, index=False)
  
  return opt_state, loss_epoch_data, acc_epoch_data

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
    z_train = int(len(train_features)/BATCH_SIZE)
    z_test = int(len(test_features)/BATCH_SIZE)    
    train_loss_data = np.zeros([max_layers,N_EPOCHS])
    train_acc_data = np.zeros([max_layers,N_EPOCHS])
    test_loss = np.zeros([max_layers])
    test_acc = np.zeros([max_layers])
    for i in range(max_layers):
      # Weights are initialized randomly
      initial_weights = jax.random.uniform(jax.random.PRNGKey(SEED), ((i+1), N_QUBITS, 3))*jax.numpy.pi
      init_state = opt_init(initial_weights)

      print("Training with "+str(i+1)+" layer(s)")
      weights, train_loss_data[i], train_acc_data[i] = Train_Model(init_state, train_features, train_target,(i+1))
      fig_name = 'strong_data/strong_full_layer'+str(i+1)+'_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
      fig_title = "Strongly Entangling Layers("+str(i+1)+") Architecture Loss and Accuracy"
      Plot_Loss_and_Acc(np.linspace(1,N_EPOCHS,num=N_EPOCHS),train_loss_data,train_acc_data,fig_title,fig_name,'Epoch')

      test_loss[i], test_acc[i] = Test_Model(weights, test_features, test_target)
      Plot_ROC(weights,test_features,test_target,(i+1))

      d = {'Epochs': ep, 'Train Loss': train_loss_data[i], 'Train Accuracy':train_acc_data[i]}
      frame = pd.DataFrame(d)
      file_name = 'strong_data/strong_epoch_loss_accuracy_data_with'+str(i+1)+'layers_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.csv'
      frame.to_csv(file_name, index=False)
    
    fig_name = 'strong_data/strong_full_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
    fig_title = "Strongly Entangling Layers Architecture Loss and Accuracy"
    Plot_Loss_and_Acc(np.linspace(1,max_layers,num=max_layers),test_loss,test_acc,fig_title,fig_name,'Layer')
      
Run_Model()  
