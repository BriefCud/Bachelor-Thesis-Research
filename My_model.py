import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml
import matplotlib.pyplot as plt
from dataset16 import load_dataset as ld_full
from dataset_muon import load_dataset as ld_muon
from functools import partial
import jax
from jax.example_libraries.optimizers import adam

# Function that splits a dataset for batching where n is the size of data chunk
def Split(data, rows):
    depth = len(data) // rows
    dataframes = np.split(data, depth)
    return dataframes, depth

def QuantumModel(SEED, TRAIN_SIZE, TEST_SIZE, N_QUBITS, N_LAYERS, LR, N_EPOCHS):
  device = qml.device("default.qubit.jax", wires=N_QUBITS,prng_key = jax.random.PRNGKey(SEED))
  train_features,train_target,test_features,test_target = ld_full(TRAIN_SIZE,TEST_SIZE,SEED)
  train_features = np.array(train_features)
  train_target = np.array(train_target)
  test_features = np.array(test_features)
  test_target = np.array(test_target)

  @partial(jax.vmap,in_axes=[0,None])
  @qml.qnode(device,interface='jax')
  def circuit(x,w):
    qml.AngleEmbedding(x,wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(w,wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

  def loss_fn(w,x,y):
    pred = circuit(x,w)
    return jax.numpy.mean((pred - y) ** 2)
  
  def acc_fn(w,x,y):
    pred = circuit(x,w)
    return jax.numpy.mean(jax.numpy.sign(pred) == y)
  
  # Split the training dataet 
  train_dataframe, chunks = Split(train_features, 1000)
  train_target_dataframe, chunks = Split(train_target, 1000)
  test_dataframe, chunks = Split(test_features, 1000)
  test_target_dataframe, chunks = Split(test_target, 1000)

  #loss_train = partial(loss_fn,x=train_features,y=train_target.to_numpy())
  #acc_train = partial(acc_fn,x=train_features,y=train_target.to_numpy())

  #loss_test = partial(loss_fn,x=test_features,y=test_target.to_numpy())
  #acc_test = partial(acc_fn,x=test_features,y=test_target.to_numpy())

  weights = jax.random.uniform(jax.random.PRNGKey(SEED), (N_LAYERS, N_QUBITS, 3))*jax.numpy.pi

  opt_init, opt_update, get_params = adam(LR)
  opt_state = opt_init(weights)

  #----- Training ------#
  @jax.jit
  def train_step(stepid, opt_state,x,y):
    current_w = get_params(opt_state)
    loss_value, grads = jax.value_and_grad(loss_fn, argnums=0)(current_w,x,y)
    acc_value = acc_fn(current_w,x,y)
    opt_state = opt_update(stepid, grads, opt_state)
    return loss_value,acc_value, opt_state

  train_loss_data = np.zeros(N_EPOCHS)
  train_acc_data = np.zeros(N_EPOCHS)
  ep = np.linspace(0,N_EPOCHS, num=N_EPOCHS)

  print("Epoch\tLoss\tAccuracy")
  for i in range(N_EPOCHS):
    for j in range(chunks):
      loss_temp = np.zeros(chunks)
      acc_temp = np.zeros(chunks)
      loss_value,acc_value, opt_state = train_step(i,opt_state,train_dataframe[j],train_target_dataframe[j])
      loss_temp[j] = loss_value
      acc_temp[j] = acc_value
    loss_avg = np.average(loss_temp)
    acc_avg = np.average(acc_temp)
    train_loss_data[i] = loss_avg
    train_acc_data[i] = acc_avg
    if (i+1) % 100 == 0:
        print(f"{i+1}\t{loss_value:.3f}\t{acc_value*100:.2f}%")
  final_state = opt_state
  
  #------- Testing -------#
  @jax.jit
  def test_step(stepid, opt_state, x, y):
    weights = get_params(opt_state)
    loss_value, grads = jax.value_and_grad(loss_fn, argnums=0)(weights, x, y)
    acc_value = acc_fn(weights,x,y)
    return loss_value, acc_value

  test_loss_data = np.zeros(N_EPOCHS)
  test_acc_data = np.zeros(N_EPOCHS)

  for j in range(chunks):
    loss_temp = np.zeros(chunks)
    acc_temp = np.zeros(chunks)
    loss_value,acc_value = test_step(i,final_state,test_dataframe[j], test_target_dataframe[j])
    loss_temp[j] = loss_value
    acc_temp[j] = acc_value
  loss_avg = np.average(loss_temp)
  acc_avg = np.average(acc_temp)
  test_loss_data[i] = loss_avg
  test_acc_data[i] = acc_avg
  print(f"\t{loss_value:.3f}\t{acc_value*100:.2f}%")
  return train_loss_data, train_acc_data, test_loss_data, test_acc_data, ep

SEED=0      
TRAIN_SIZE = 2000 
TEST_SIZE = 2000
N_QUBITS = 16   
N_LAYERS = 2
LR=1e-3 
N_EPOCHS = 1000

train_layers_data = np.zeros(10)
test_layers_data = np.zeros(10)
num_layer = np.linspace(1,10, num =10)
for i in range(10):
  train_ld, train_ad, test_ld, test_ad, ep = QuantumModel(SEED, TRAIN_SIZE, TEST_SIZE, N_QUBITS, i, LR, N_EPOCHS)
  train_layers_data[i] = train_ad[-1]
  test_layers_data[i] = test_ad[-1]
plt.title('Accuracy vs Layers')
plt.xlabel("# of layers", sie=14)
plt.ylabel('Accuracy', size=14)
plt.plot(num_layer,train_layers_data,'r',label='Training')
plt.plot(num_layer,test_layers_data,'b', label='Testing')
plt.legend(loc='lower right')
file_name = 'full_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
plt.savefig(file_name)
