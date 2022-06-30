# Imports
import pandas as pd
import numpy as np
from dataset16 import load_dataset
from sklearn.metrics import roc_curve, roc_auc_score 
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml
from functools import partial
import jax
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import pennylane_qiskit
from qiskit import IBMQ

# Constants
SEED=0              # Fix it for reproducibility (kind of)
TRAIN_SIZE = 200*5  # Number of jets (has to be even) for training
TEST_SIZE = 200*2   # ------------------------------- for testing
N_QUBITS = 16       # One qubit per feature
N_LAYERS = 2        # Add more layers for extra complexity
LR=1e-3           # Learning rate of the ADAM optimizer
N_EPOCHS = 10     # Number of training epochs
BATCH_SIZE = 200
N_PARAMS_B = 3
