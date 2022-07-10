# Imports
import numpy as np
import pandas as pd
import os
from psutil import virtual_memory
import pennylane as qml
import matplotlib.pyplot as plt
from functools import partial
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
import jax
import jax.numpy as jnp
import optax
import sklearn
from sklearn.metrics import roc_curve, roc_auc_score 

# ------ Constants ------#

SEED=0      
TRAIN_SIZE = 1000*20 
TEST_SIZE = 1000*30
N_QUBITS = 4   
LR=0.001 
N_EPOCHS = 1000
BATCH_SIZE = 1000
HOME_PATH = '/home/leonidas/example-qml4btag/'

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
  return jnp.mean((pred - y) ** 2)

# Simple binary accuracy function
def Accuracy(w,x,y):
  pred = Circuit(x,w)
  return jnp.mean(jax.numpy.sign(pred) == y)

# The ADAM optimizer is initialized
optimizer = optax.adam(LR)

# Training step
# This function is compiled Just-In-Time on the GPU
@jax.jit
def Train_Step(opt_state,w,x,y):
  loss_value, grads = jax.value_and_grad(Loss,argnums=0)(w,x,y)
  acc_value = Accuracy(w,x,y)
  updates, opt_state = optimizer.update(grads, opt_state, w)
  w = optax.apply_updates(w, updates)
  return loss_value,acc_value, opt_state,w

@jax.jit
def Test_Step(w,x,y):
  return Loss(w,x,y), Accuracy(w,x,y)

def Batch_and_Shuffle(x,y):
  z = int(len(x) / BATCH_SIZE)
  data = jnp.column_stack([x,y])
  jax.random.permutation(jax.random.PRNGKey(SEED),data)
  return jnp.split(data[:,0:N_QUBITS],z), jnp.split(data[:,-1],z),z

def Train_Model(w:optax.Params,x, y, layer):
  opt_state = optimizer.init(w)
  z = int(len(x)/BATCH_SIZE)
  loss_data = np.zeros(N_EPOCHS)
  acc_data = np.zeros(N_EPOCHS)
  print("Training...")
  print("Epoch\tLoss\tAccuracy")
  step=0
  for i in range(N_EPOCHS):
    
    # Batch and shuffle the data for ever epoch
    train_f, train_t, chunks = Batch_and_Shuffle(x, y)
    loss_temp = np.zeros((chunks))
    acc_temp = np.zeros((chunks))
    for j in range(chunks):
      loss_temp[j],acc_temp[j], opt_state,w = Train_Step(opt_state,w,train_f[j], train_t[j])
      step+=1

    loss_data[i] = np.mean(loss_temp)
    acc_data[i] = np.mean(acc_temp)

    if (i+1) % 100 == 0:
      print(f"{i+1}\t{loss_data[i]:.3f}\t{acc_data[i]*100:.2f}%")
  
  fweights = HOME_PATH+'strong_w/final_strong_weights_with_'+str(layer)+'layers'+'_training' +str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.npy'
  np.save(fweights, w)

  return w, loss_data, acc_data

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
  z = int(len(x) / BATCH_SIZE)
  new_x = np.split(x,z)
  ps = np.zeros([z,BATCH_SIZE])
  for i in range(z):
    ps[i] = Circuit(new_x[i],w)
  predictions = np.reshape(ps, (z*BATCH_SIZE))
  fpr, tpr, threshold = roc_curve(y,predictions)
  auc = roc_auc_score(y,predictions)
  df_auc = np.ones(len(fpr))*auc

  # Get data predictions from the XGBoost to compare ROC curves
  xgb_csv =  pd.read_csv(HOME_PATH+'data/test_withxgb.csv')
  xgb_csv = xgb_csv[xgb_csv['mu_Q'] != 0] # only if using muon dataset include this code
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
  fig_name = HOME_PATH+'strong_data/ROC_strong_layer'+str(layer)+'training' +str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(fig_name)
  plt.clf()
  
  roc_d = {'FPR': fpr, 'TPR': tpr, 'Threshold': threshold, 'Area': df_auc}
  frame = pd.DataFrame(roc_d)
  file_name = HOME_PATH+'strong_data/strong_roc_data_'+str(layer)+'layers_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.csv'
  frame.to_csv(file_name, index=False)

  # Plot the distribution
  pb = predictions[y==1]
  pb_bar = predictions[y==-1]
  plt.hist(pb,bins=np.linspace(-1, 1, 100),alpha=0.5,label='Pb')
  plt.hist(pb_bar,bins=np.linspace(-1, 1, 100),alpha=0.5,label='Pb-bar')
  plt.xlim([-1,1])
  plt.legend(loc='upper right')
  fname = HOME_PATH+'strong_data/strong_prob_dist_'+str(layer)+'layers_training' +str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(fname)
  plt.clf()

def Plot_Loss_and_Acc(ep,loss,acc,layer):
  fig, axis = plt.subplots(2,1) 
  fig.suptitle("Strongly Entangling "+str(layer)+" Layers")
  axis[0].set_xlabel('# of Epochs') 
  axis[0].set_ylabel('Loss') 
  axis[0].plot(ep, loss) 
  axis[1].set_xlabel('# of Epochs')
  axis[1].set_ylabel('Accuracy') 
  axis[1].plot(ep, acc) 
  file_name = HOME_PATH+'strong_data/strong_full_layer'+str(layer)+'training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(file_name) 
  plt.clf()

def Run_Model(train_features,train_target,test_features,test_target,pre_trained="file name"):
  max_layers = 8
  z = int(len(train_features)/BATCH_SIZE)
  train_loss_data = np.zeros([max_layers,N_EPOCHS])
  train_acc_data = np.zeros([max_layers,N_EPOCHS])
  test_loss = np.zeros([max_layers])
  test_acc = np.zeros([max_layers])
  for i in range(max_layers):
    # Weights are initialized randomly
    init_w = jax.random.uniform(jax.random.PRNGKey(SEED), ((i+1), N_QUBITS, 3))*jax.numpy.pi

    print("Training with "+str(i+1)+" layer(s)")
    weights, train_loss_data[i], train_acc_data[i] = Train_Model(init_w, train_features, train_target, i+1)
    ep = np.linspace(1,N_EPOCHS,num=N_EPOCHS)
    Plot_Loss_and_Acc(ep,train_loss_data[i],train_acc_data[i],(i+1))

    test_loss[i], test_acc[i] = Test_Model(weights, test_features, test_target)
    Plot_ROC(weights,test_features,test_target,(i+1))

    d = {'Epochs': ep, 'Train Loss': train_loss_data[i], 'Train Accuracy':train_acc_data[i]}
    frame = pd.DataFrame(d)
    file_name = HOME_PATH+'strong_data/strong_loss_accuracy_data_with'+str(i+1)+'layers.csv'
    frame.to_csv(file_name, index=False)


def Menu():
  # Loads the dataset (already preprocessed... see dataset.py)
  train_features,train_target,test_features,test_target = load_dataset(TRAIN_SIZE,TEST_SIZE,SEED,True)

  path = HOME_PATH+'strong_w/'
  weights_files = os.scandir(path) # Get the .npy weight files
  with weights_files as entries:
    for entry in entries:
        print(entry.name)
  print("Please choose a file from above to get the ROC curve for the Strong model, otherwise press the space bar, then enter to pass this stage.")
  w_f = input("Enter file name including the extension:  ")

  if (w_f != " "):
    weights = np.load(path+w_f)
    test_loss, test_acc = Test_Model(weights, test_features, test_target)
    Plot_ROC(weights,test_features,test_target, 0)
  else:
    ram_gb = virtual_memory().total / 1e9
    if ram_gb > 20:
      Run_Model(train_features,train_target,test_features,test_target)
    else:
      print("not using high-RAM runtime")
