# Imports
from load_dataset import load_dataset
import numpy as np
import pandas as pd
import os
import optax
import pennylane as qml
import matplotlib.pyplot as plt
from functools import partial
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
import jax
import jax.numpy as jnp
import sklearn
from sklearn.metrics import roc_curve, roc_auc_score 

# ------ Constants ------#

SEED=7
TRAIN_SIZE = 1000*20 
TEST_SIZE = 1000*30
N_QUBITS = 16 
N_PARAMS_B = 3
LR=0.001 
N_EPOCHS = 2000
BATCH_SIZE = 1000
HOME_PATH = '/home/leonidas/example-qml4btag/'

#------------------------#

device = qml.device("default.qubit.jax", wires=N_QUBITS,prng_key = jax.random.PRNGKey(SEED))

def Block(weights,wires):
  qml.RY(weights[0], wires=wires[0])
  qml.RX(weights[1], wires=wires[0])
  qml.RZ(weights[2], wires=wires[1])
  qml.CNOT(wires=wires)

@partial(jax.vmap,in_axes=[0,None]) # Vectorized version of the function
@qml.qnode(device,interface='jax')  # Create a Pennylane QNode
def Circuit(x,w):
  qml.AngleEmbedding(x,wires=range(N_QUBITS))   # Features x are embedded in rotation angles
  qml.TTN(wires=range(N_QUBITS), n_block_wires=2,block=Block, n_params_block=N_PARAMS_B, template_weights=w) # Variational layer
  return qml.expval(qml.PauliZ(N_QUBITS-1)) # Expectation value of the \sigma_z operator on the last qubit

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
def Train_Step(w, opt_state,x,y):
  loss_value, grads = jax.value_and_grad(Loss,argnums=0)(w,x,y)
  acc_value = Accuracy(w,x,y)
  updates, opt_state = optimizer.update(grads, opt_state, w)
  w = optax.apply_updates(w, updates)
  return loss_value,acc_value, opt_state, w

@jax.jit
def Test_Step(w,x,y):
  loss_value = Loss(w,x,y)
  acc_value = Accuracy(w,x,y)
  return loss_value, acc_value

def Batch_and_Shuffle(x,y):
  z = int(len(x) / BATCH_SIZE)
  data = jnp.column_stack([x,y])
  jax.random.permutation(jax.random.PRNGKey(SEED),data)
  return jnp.split(data[:,0:N_QUBITS],z), jnp.split(data[:,-1],z),z

def Train_Model(w:optax.Params,x, y):
  opt_state = optimizer.init(w)
  z = int(len(x)/BATCH_SIZE)
  loss_data = np.zeros(N_EPOCHS*z)
  acc_data = np.zeros(N_EPOCHS*z)
  print("Training...")
  print("Epoch\tLoss\tAccuracy")
  step=0
  for i in range(N_EPOCHS):
    
    # Batch and shuffle the data for ever epoch
    train_f, train_t, chunks = Batch_and_Shuffle(x, y)
    # loss_temp = jnp.zeros(chunks)
    # acc_temp = jnp.zeros(chunks)

    for j in range(chunks):
      # loss_temp[j],acc_temp[j], opt_state = Train_Step(step, opt_state, train_f[j], train_t[j])
      loss_data[step],acc_data[step], opt_state, w = Train_Step(w, opt_state, train_f[j], train_t[j])
      step+=1
    
    # loss_data[i] = jnp.average(loss_temp)
    # acc_data[i] = jnp.average(acc_temp)

    
    if (i+1) % 100 == 0:
      print(f"{i+1}\t{loss_data[i]:.3f}\t{acc_data[i]*100:.2f}%")
   
  file_weights = HOME_PATH+'ttn_w/final_ttn_weights_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.npy'
  np.save(file_weights, w)

  return w, loss_data, acc_data

def Test_Model(w,x,y):
  print("Testing...")  
  print("\tLoss\tAccuracy")
  # Batch and shuffle the data for ever epoch
  test_f, test_t, chunks = Batch_and_Shuffle(x, y)
  loss_temp = np.zeros(chunks)
  acc_temp = np.zeros(chunks)

  for j in range(chunks):
    loss_temp[j],acc_temp[j] = Test_Step(w, test_f[j], test_t[j])

  loss_data = jnp.average(loss_temp)
  acc_data = jnp.average(acc_temp)
  print(f"\t{loss_data:.3f}\t{acc_data*100:.2f}%")

  return loss_data, acc_data

def Plot_ROC(w,x,y):
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
  # xgb_csv = xgb_csv[xgb_csv['mu_Q'] != 0] # only if using muon dataset include this code
  xgb_pred = xgb_csv['XGB_PRED'] 
  xgb_target = xgb_csv['Jet_LABEL']*2-1
  xgb_fpr,xgb_tpr,xgb_threshold = roc_curve(xgb_target,xgb_pred)
  xgb_auc = roc_auc_score(xgb_target,xgb_pred)
  
  plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
  plt.plot(fpr,tpr,label="TTN(area = %0.2f)" % auc)
  plt.plot(xgb_fpr,xgb_tpr,label="XGBoost(area = %0.2f)" % xgb_auc)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver Operating Characteristic")
  plt.legend(loc="lower right")
  fig_name = HOME_PATH+'ttn_data/ROC_ttn_training' +str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(fig_name)
  plt.clf()
  
  roc_d = {'FPR': fpr, 'TPR': tpr, 'Threshold': threshold, 'Area': df_auc}
  frame = pd.DataFrame(roc_d)
  file_name = HOME_PATH+'ttn_data/ttn_roc_data_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.csv'
  frame.to_csv(file_name, index=False)

  # Plot the distribution
  pb = predictions[y==1]
  pb_bar = predictions[y==-1]
  plt.hist(pb,bins=np.linspace(-1, 1, 100),alpha=0.5,label='Pb')
  plt.hist(pb_bar,bins=np.linspace(-1, 1, 100),alpha=0.5,label='Pb-bar')
  plt.xlim([-1,1])
  plt.legend(loc='upper right')
  fname = HOME_PATH+'ttn_data/ttn_prob_dist_training' +str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(fname)
  plt.clf()


def Plot_Loss_and_Acc(ep,loss,acc):
  fig, axis = plt.subplots(2,1) 
  fig.suptitle("TTN")
  axis[0].set_xlabel('Step') 
  axis[0].set_ylabel('Loss') 
  axis[0].plot(ep, loss) 
  axis[1].set_xlabel('Step')
  axis[1].set_ylabel('Accuracy') 
  axis[1].plot(ep, acc) 
  file_name = HOME_PATH+'ttn_data/ttn_muon_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(file_name) 
  plt.clf()

def Run_Model():
  # Loads the dataset (already preprocessed... see dataset.py)
  train_features,train_target,test_features,test_target = load_dataset(TRAIN_SIZE,TEST_SIZE,SEED)
  z = int(len(train_features) / BATCH_SIZE)

  path = HOME_PATH+'ttn_w/'
  weights_files = os.scandir(path) # Get the .npy weight files
  with weights_files as entries:
    for entry in entries:
        print(entry.name)
  print("Please Choose a file from above to load the weights for the TTN model, otherwise press the space bar, then enter to pass this stage.")
  w_f = input("Enter file name:  ")
  if (w_f != " "):
    weights = np.load(path+w_f)
    test_loss, test_acc = Test_Model(weights, test_features, test_target)
    Plot_ROC(weights,test_features,test_target)
  else:
    # Weights are initialized randomly
    init_w = jax.random.uniform(jax.random.PRNGKey(SEED), (N_QUBITS-1, N_PARAMS_B))*jax.numpy.pi
    weights, train_loss, train_acc = Train_Model(init_w,train_features, train_target)
    ep = np.linspace(1,N_EPOCHS*z,num=N_EPOCHS*z)
    Plot_Loss_and_Acc(ep,train_loss,train_acc)

    test_loss, test_acc = Test_Model(weights, test_features, test_target)
    Plot_ROC(weights,test_features,test_target)

    d = {'Steps': ep, 'Train Loss': train_loss, 'Train Accuracy':train_acc}
    frame = pd.DataFrame(d)
    frame.to_csv(HOME_PATH+'ttn_data/ttn_loss_accuracy_data_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.csv', index=False) 
