import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from dataset16 import load_dataset as ld_full
from dataset_muon import load_dataset as ld_muon
from functools import partial
import jax
from jax.example_libraries.optimizers import adam

def my_model(SEED, TRAIN_SIZE, TEST_SIZE, N_QUBITS, N_LAYERS, LR, N_EPOCHS):
  
  # Definiton of the Pennylane device using JAX
  device = qml.device("default.qubit.jax", wires=N_QUBITS,prng_key = jax.random.PRNGKey(SEED))

  # Loads the dataset (already preprocessed... see dataset.py)
  train_features,train_target,test_features,test_target = ld_full(TRAIN_SIZE,TEST_SIZE,SEED)
  
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
  def circuit(x,w):
      qml.AngleEmbedding(x,wires=range(N_QUBITS))   # Features x are embedded in rotation angles
      qml.StronglyEntanglingLayers(w,wires=range(N_QUBITS)) # Variational layer
      return qml.expval(qml.PauliZ(0)) # Expectation value of the \sigma_z operator on the 1st qubit

  # Simple MSE loss function
  def loss_fn(w,x,y):
      pred = circuit(x,w)
      return jax.numpy.mean((pred - y) ** 2)

  # Simple binary accuracy function
  def acc_fn(w,x,y):
      pred = circuit(x,w)
      return jax.numpy.mean(jax.numpy.sign(pred) == y)
    
  # Weights are initialized randomly
  weights = jax.random.uniform(jax.random.PRNGKey(SEED), (N_LAYERS, N_QUBITS, 3))*jax.numpy.pi

  # The ADAM optimizer is initialized
  opt_init, opt_update, get_params = adam(LR)
  opt_state = opt_init(weights)
  
  # Training step
  # This function is compiled Just-In-Time on the GPU
  @jax.jit
  def step(stepid, opt_state,train_f,train_t,test_f,test_t):
    current_w = get_params(opt_state)
    loss_value, grads = jax.value_and_grad(loss_fn,argnums=0)(current_w,train_f,train_t)
    acc_value = acc_fn(current_w,test_f,test_t)
    opt_state = opt_update(stepid, grads, opt_state)
    return loss_value,acc_value, opt_state
  
  def batch_and_shuffle(x,y,batch_size):
    assert (x % batch_size) == 0 and (y % batch_size) == 0
    z = int(len(x) / batch_size)
    data = np.comulmn_stack(x,y)
    data = np.random.shuffle(data)
    new_x = data[:,0:N_QUBITS-1]
    new_y = data[:,-1]
    return np.split(new_x,z), np.split(new_y,z),z
  
  loss_data = np.zeros(N_EPOCHS)
  acc_data = np.zeros(N_EPOCHS)
  batch_size = 250
  print("Epoch\tLoss\tAccuracy")
  for i in range(N_EPOCHS):
    # Batch and shuffle the data for ever epoch
    train_f, train_t, chunks = batch_and_shuffle(np.array(train_features), np.array(train_target), batch_size)
    test_f, test_t, chunks = batch_and_shuffle(np.array(test_features), np.array(test_target), batch_size)
    loss_temp = np.zeros(chunks)
    acc_temp = np.zeros(chunks)

    for j in range(chunks):
      loss_temp[j],acc_temp[j], opt_state = step(i, opt_state, train_f[j], train_t[j], test_f[j], test_t[j])

    loss_data[i] = np.average(loss_temp)
    acc_data[i] = np.average(acc_temp)

    if (i+1) % 100 == 0:
      print(f"{i+1}\t{loss_value:.3f}\t{acc_value*100:.2f}%")
  
  return loss_data, acc_data
  
 def run_model():
  
  SEED=0      
  TRAIN_SIZE = 10000 
  TEST_SIZE = 10000
  N_QUBITS = 16   
  N_LAYERS = 2
  LR=1e-2 
  N_EPOCHS = 1000
  
  loss_data = np.zeros([N_EPOCHS, 10])
  acc_data = np.zeros([N_EPOCHS, 10])
  num_layer = np.linspace(1,10, num =10)
  y_error = np.zeros(10)
  
  for i in range(10):
    
    loss_temp, acc_temp = my_model(SEED, TRAIN_SIZE, TEST_SIZE, N_QUBITS, i, LR, N_EPOCHS)
    loss_data[:,i] = loss_temp
    acc_data[:,i] = acc_temp
    y_error[i] = np.std(acc_temp)
    
  plt.title('Accuracy vs Layers')
  plt.xlabel("# of layers", sie=14)
  plt.ylabel('Accuracy', size=14)
  plt.errorbar(num_layer,acc_data[-1,:],yerr=y_error)
  file_name = 'full_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(file_name)
  
   
run_model()

