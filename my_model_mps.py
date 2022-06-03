import numpy as np
import pandas as pd
import pennylane as qml
import matplotlib.pyplot as plt
from dataset16 import load_dataset as ld_full
from dataset_muon import load_dataset as ld_muon
from functools import partial
import jax
from jax.example_libraries.optimizers import adam
import sklearn
from sklearn.metrics import roc_curve, roc_auc_score 

def my_model(SEED, TRAIN_SIZE, TEST_SIZE, N_QUBITS, N_PARAMS_B, LR, N_EPOCHS,train_features,train_target,test_features,test_target):
  
  # Definiton of the Pennylane device using JAX
  device = qml.device("default.qubit.jax", wires=N_QUBITS,prng_key = jax.random.PRNGKey(SEED))
  
  # The block defines a variational quantum circuit that takes the position of tensors in the circuit
  def block(weights,wires):
    qml.RX(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.CNOT(wires=wires)
  
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
      qml.MPS(wires=range(N_QUBITS), n_block_wires=2,block=block, n_params_block=N_PARAMS_B, template_weights=w) # Variational layer
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
  weights = jax.random.uniform(jax.random.PRNGKey(SEED), (N_QUBITS-1, N_PARAMS_B))*jax.numpy.pi

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
      np.save("mps_w\mps_weights_epcoh_"+ str(i+1) +".npy", get_params(opt_state))
   
  final_state = opt_state
  file_weights = "mps_w\final_mps_weights.npy"
  np.save(file_weights, get_params(final_state))
  
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
  
  plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
  plt.plot(fpr,tpr,label="ROC QML,MPS")(area = %0.2f)" % auc)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver Operating Characteristic")
  plt.legend(loc="lower right")
  fname = 'ROC_' + str(N_LAYERS) + 'layers_full_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(fname)
  plt.clf()
  
  return train_loss_data, train_acc_data, test_loss_data, test_acc_data
  
def run_model():
  
  SEED=0      
  TRAIN_SIZE = 20000 
  TEST_SIZE = 10000
  N_QUBITS = 16   
  N_PARAMS_B = 2
  LR=1e-2 
  N_EPOCHS = 3000
  
  # Loads the dataset (already preprocessed... see dataset.py)
  train_features,train_target,test_features,test_target = ld_muon(TRAIN_SIZE,TEST_SIZE,SEED)
  
  train_loss_data = np.zeros(N_EPOCHS)
  train_acc_data = np.zeros(N_EPOCHS)
  test_loss_data = np.zeros(N_EPOCHS)
  test_acc_data = np.zeros(N_EPOCHS)  
   
  train_loss, train_acc, test_loss, test_acc = my_model(SEED, TRAIN_SIZE, TEST_SIZE, N_QUBITS, N_PARAMS_B, LR, N_EPOCHS, train_features,train_target,test_features,test_target)
  
  ep = np.linspace(1,N_EPOCHS,num=N_EPOCHS)
  
  fig, ax1 = plt.subplots() 
  ax1.set_xlabel('# of Epochs') 
  ax1.set_ylabel('Loss', color = 'black') 
  plot_1 = ax1.plot(ep, train_loss, color = 'black') 
  ax1.tick_params(axis ='Loss', labelcolor = 'black')
  ax2 = ax1.twinx() 
  ax2.set_ylabel('Accuracy', color = 'green') 
  plot_2 = ax2.plot(ep, train_acc, color = 'green') 
  ax2.tick_params(axis ='Accuracy', labelcolor = 'green')
  plt.title("Matrix Product State Architecture Loss and Accuracy")
  file_name = 'mps_full_training'+str(TRAIN_SIZE)+'_testing'+str(TEST_SIZE)+'.png'
  plt.savefig(file_name) 
  
  d = {'Epochs': ep, 'Train Loss': train_loss, 'Train Accuracy':train_acc, 'Test Loss':test_loss, 'Test Accuracy':test_acc}
  frame = pd.DataFrame(d)
  frame.to_csv('mps_loss_accuracy_data', index=False)
  
run_model()

