# Imports
import pandas as pd
import numpy as np
from dataset16 import load_dataset
from sklearn.metrics import roc_curve, roc_auc_score 
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml
from functools import partial
import jax
#import optax
from jax.example_libraries.optimizers import adam
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import pennylane_qiskit
from qiskit import IBMQ

# Constants
SEED=0              # Fix it for reproducibility (kind of)
TRAIN_SIZE = 0
TEST_SIZE = 300*5  # Number of jets (has to be even) for testing
N_QUBITS = 16       # One qubit per feature
N_LAYERS = 2        # Add more layers for extra complexity
LR=1e-3           # Learning rate of the ADAM optimizer
N_EPOCHS = 10     # Number of training epochs
BATCH_SIZE = 300
N_PARAMS_B = 3

token=""
prov = IBMQ.get_provider(hub='ibm-q-cern',group='internal', project='qml4btag')
ibm_device = qml.device('qiskit.ibmq', provider=prov,backend='ibm_hanoi',wires=N_QUBITS,imbqx_token = token)

opt_init, opt_update, get_params = adam(LR)

def Block(weights,wires):
  qml.RZ(weights[0], wires=wires[0])
  qml.RY(weights[1], wires=wires[1])
  qml.U1(weights[2],wires=wires[0])
  qml.CZ(wires=wires)
    
@partial(jax.vmap,in_axes=[0,None,None])
@qml.batch_param(all_operations=true)
@qml.qnode(ibm_device,interface='jax')
def Strong_Circuit(x,w):
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(w,wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

@partial(jax.vmap,in_axes=[0,None,None])
@qml.batch_param(all_operations=true)
@qml.qnode(ibm_device,interface='jax')
def Mps_Circuit(x,w):
  qml.AngleEmbedding(x, wires=range(N_QUBITS))
  qml.MPS(wires=range(N_QUBITS), n_block_wires=2,block=Block, n_params_block=N_PARAMS_B, template_weights=w) 
  return qml.expval(qml.PauliZ(N_QUBITS-1))

@partial(jax.vmap,in_axes=[0,None,None])
@qml.batch_param(all_operations=true)
@qml.qnode(ibm_device,interface='jax')
def Ttn_Circuit(x,w):
  qml.AngleEmbedding(x, wires=range(N_QUBITS))
  qml.TTN(wires=range(N_QUBITS), n_block_wires=2,block=Block, n_params_block=N_PARAMS_B, template_weights=w)
  return qml.expval(qml.PauliZ(N_QUBITS-1))

# Simple MSE loss function
def Strong_Loss(w,x,y):
  pred = Strong_Circuit(x,w)
  return jnp.mean((pred - y) ** 2)

# Simple binary accuracy function
def Strong_Accuracy(w,x,y):
  pred = Strong_Circuit(x,w)
  return jnp.mean(jax.numpy.sign(pred) == y)

def Mps_Loss(w,x,y):
  pred = Mps_Circuit(x,w)
  return jnp.mean((pred - y) ** 2)

def Mps_Accuracy(w,x,y):
  pred = Mps_Circuit(x,w)
  return jnp.mean(jax.numpy.sign(pred) == y)

def Ttn_Loss(w,x,y):
  pred = Ttn_Circuit(x,w)
  return jnp.mean((pred - y) ** 2)

def Ttn_Accuracy(w,x,y):
  pred = Ttn_Circuit(x,w)
  return jnp.mean(jax.numpy.sign(pred) == y)

def Batch(x,y):
  z = int(len(x) / BATCH_SIZE)
  data = np.column_stack([x,y])
  return np.split(data[:,0:N_QUBITS],z), np.split(data[:,-1],z),z

def Test_Step(w,x,y,n):
  if(n==0):
    loss_value = jax.jit(Strong_Loss(w,x,y))
    acc_value = jax.jit(Strong_Accuracy(w,x,y))
  elif(n==1):
    loss_value = jax.jit(Mps_Loss(w,x,y))
    acc_value = jax.jit(Mps_Accuracy(w,x,y))
  elif(n==2):
    loss_value = jax.jit(Ttn_Loss(w,x,y))
    acc_value = jax.jit(Ttn_Accuracy(w,x,y))

  return loss_value, acc_value

def Test_Model(w,x,y,n):
  print("Testing...")  
  print("\tLoss\tAccuracy")
  # Batch and shuffle the data for ever epoch
  test_f, test_t, chunks = Batch(x, y)
  loss_temp = np.zeros(chunks)
  acc_temp = np.zeros(chunks)

  for j in range(chunks):
    loss_temp[j],acc_temp[j] = Test_Step(w, test_f[j], test_t[j],n)

  loss_data = np.mean(loss_temp)
  acc_data = np.mean(acc_temp)

  print(f"\t{loss_data:.3f}\t{acc_data*100:.2f}%")

  return loss_data, acc_data

def Run_Model():
  
  train_features,train_target,test_features,test_target = load_dataset(TRAIN_SIZE,TEST_SIZE,SEED)
  
  input_model = input("Which model would you like to choose from, Strong=0, MPS=1, TTN=2? Enter a number:")
    path=""
    n=0
    file_name = ''
    if(input_model=='0'):
      path = "strong_w/"
      n=0
      file_name = 'strong_data/strong_loss_accuracy_data_hardware_testsize'+str(TESTSIZE)+'.csv'
    elif(input_model=='1'):
      path = "mps_w/"
      n=1
      file_name = 'mps_data/mps_loss_accuracy_data_hardware_testsize'+str(TESTSIZE)+'.csv'
    elif(input_model=='2'):
      path = "ttn_w/"
      n=2
      file_name = 'ttn_data/ttn_loss_accuracy_data_hardware_testsize'+str(TESTSIZE)+'.csv'
    else:
      print("Model does not exist")
      exit()

    weights_files = os.scandir(path) # Get the .npy weight files
    with weights_files as entries:
      for entry in entries:
        print(entry.name)
    print("Please Choose a file from above to load the weights for the Strong model, otherwise press the space bar, then enter to pass this stage.")
    w_f = input("Enter file name:  ")
    
    weights = np.load(path+"/"+w_f)
    test_loss, test_acc = Test_Model(weights, test_features, test_target,n)
    
    d = {'Test Loss': test_loss, 'Test Accuracy':test_acc}
    frame = pd.DataFrame(d)
    frame.to_csv(file_name, index=False)
    
    
