import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from dataset16 import load_dataset as ld_full
from dataset_muon import load_dataset as ld_muon
from functools import partial
import jax
from jax.example_libraries.optimizers import adam
import sklearn
from sklearn.metrics import roc_curve, roc_auc_score 

def my_model(SEED, TRAIN_SIZE, TEST_SIZE, N_QUBITS, N_LAYERS, LR, N_EPOCHS,train_features,train_target,test_features,test_target):
  
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
  def train_step(stepid, opt_state,train_f,train_t):
    current_w = get_params(opt_state)
    loss_value, grads = jax.value_and_grad(loss_fn,argnums=0)(current_w,train_f,train_t)
    acc_value = acc_fn(current_w,train_f,train_t)
    opt_state = opt_update(stepid, grads, opt_state)
    return loss_value,acc_value, opt_state
  
  def test_step(final_state,test_f,test_t):
    current_w = get_params(final_state)
    loss_value, grads = jax.value_and_grad(loss_fn,argnums=0)(current_w,test_f,test_t)
    acc_value = acc_fn(current_w,test_f,test_t)
    return loss_value, acc_value
  
  def batch_and_shuffle(x,y,batch_size):
    z = int(len(x) / batch_size)
    data = np.column_stack([x,y])
    np.random.shuffle(data)
    return np.split(data[:,0:N_QUBITS],z), np.split(data[:,-1],z),z
  
  # -------------------------- Training -------------------------- #
  
  train_loss_data = np.zeros(N_EPOCHS)
  train_acc_data = np.zeros(N_EPOCHS)
  batch_size = 200
  print("Training...")
  print("Epoch\tLoss\tAccuracy")
  for i in range(N_EPOCHS):
    
    # Batch and shuffle the data for ever epoch
    train_f, train_t, chunks = batch_and_shuffle(np.array(train_features), np.array(train_target), batch_size)
    loss_temp = np.zeros(chunks)
    acc_temp = np.zeros(chunks)

    for j in range(chunks):
      loss_temp[j],acc_temp[j], opt_state = train_step(i, opt_state, train_f[j], train_t[j])

    train_loss_data[i] = np.average(loss_temp)
    train_acc_data[i] = np.average(acc_temp)

    if (i+1) % 100 == 0:
      print(f"{i+1}\t{ train_loss_data[i]:.3f}\t{train_acc_data[i]*100:.2f}%")
   
  final_state = opt_state
  file_weigths = "strong_weights_with_" + str(N_LAYERS) + "layers.npy"
  np.save(file_weights,get_params(weights))
  
  # -------------------------- TESTING -------------------------- #
  
  print("Testing...")  
  print("\tLoss\tAccuracy")
  loss_temp = np.zeros(chunks)
  acc_temp = np.zeros(chunks)
  # Batch and shuffle the data for ever epoch
  test_f, test_t, chunks = batch_and_shuffle(np.array(test_features), np.array(test_target), batch_size)
  for j in range(chunks):
    loss_temp[j],acc_temp[j] = test_step(final_state, test_f[j], test_t[j])

  test_loss_data = np.average(loss_temp)
  test_acc_data = np.average(acc_temp)

  print(f"{i+1}\t{test_loss_data:.3f}\t{test_acc_data*100:.2f}%")
  
  # -------------------------- ROC curve -------------------------- #
  predictions = circuit(test_features,get_params(final_state))
  fpr, tpr, threshold = roc_curve(test_target,predictions)
  auc = roc_auc_score(test_target,predictions)
  
  plt.plot(fpr,tpr,label="ROC QML (layers = " + str(N_LAYERS) + ")(area = %0.2f)" % auc)
  fname = 'ROC_' + str(N_LAYERS) + 'layers_full_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(fname)
  
  return train_loss_data, train_acc_data, test_loss_data, test_acc_data
  
def run_model():
  
  SEED=0      
  TRAIN_SIZE = 65000 
  TEST_SIZE = 40000
  N_QUBITS = 16   
  N_LAYERS = 2
  LR=1e-2 
  N_EPOCHS = 1500
  
  # Loads the dataset (already preprocessed... see dataset.py)
  train_features,train_target,test_features,test_target = ld_full(TRAIN_SIZE,TEST_SIZE,SEED)
  
  max_layers = 8
  
  train_loss_data = np.zeros([N_EPOCHS, max_layers])
  train_acc_data = np.zeros([N_EPOCHS, max_layers])
  test_loss_data = np.zeros([N_EPOCHS, max_layers])
  test_acc_data = np.zeros([N_EPOCHS, max_layers])
  num_layer = np.linspace(1,max_layers, num =max_layers)
  y_error = np.zeros(max_layers)
  
  for i in range(max_layers):
    
    train_loss_temp, train_acc_temp, test_loss_temp, test_acc_temp = my_model(SEED, TRAIN_SIZE, TEST_SIZE, N_QUBITS, (i+1), LR, N_EPOCHS, train_features,train_target,test_features,test_target)
    
    train_loss_data[:,i] = train_loss_temp
    train_acc_data[:,i] = train_acc_temp
    
    test_loss_data[i] = test_loss_temp
    test_acc_data[i] = test_acc_temp
    
    y_error[i] = np.std(train_acc_temp)
  
  np.savetxt('train_loss.csv',train_loss_data, delimiter=',')
  np.savetxt('train_acc.csv',train_acc_data, delimiter=',')
  np.savetxt('test_loss.csv',test_acc_data, delimiter=',')
  np.savetxt('test_acc.csv',test_acc_data, delimiter=',')
  
  plt.title('Accuracy vs Layers')
  plt.xlabel("# of layers", sie=14)
  plt.ylabel('Accuracy', size=14)
  plt.errorbar(num_layer,train_acc_data[-1,:],yerr=y_error)
  plt.plot(num_layer,test_acc_data)
  file_name = 'full_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(file_name) 
   
run_model()

