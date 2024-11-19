# QNNLib: A Python Library for Simplifying Quantum Neural Network Development

QNNLib is an open-source Python library designed to simplify the construction and implementation of Quantum Neural Networks (QNNs). Built on the PennyLane framework, it bridges the gap between quantum computing and machine learning by providing an intuitive interface for building QNNs based on the ZZFeatureMap and TwoLocal architectures. QNNLib is suitable for data scientists and researchers looking to explore quantum-enhanced solutions with minimal technical hurdles.

## Features
- **Simplified Ansatz Configuration**: Easily configure the number of ZZFeatureMap and TwoLocal layers (reps) without manually constructing circuits.
- **CSV Dataset Integration**: Effortlessly import CSV datasets and run QNN experiments, mimicking workflows of traditional machine learning libraries.
- **Pre-trained Model Loading**: Save and reuse QNN models for predictions, similar to classical frameworks like Keras.
- **Quantum Backend Support**: Compatible with PennyLane-supported simulators (`default.qubit`, `lightning.qubit`, etc.) and physical quantum backends such as IBMQ through Qiskit.
- **Hyperparameter Tuning**: Adjust batch size, epochs, learning rates, optimizers, and random seeds for fine-tuned experiments.
- **Results Export**: Save training progress, accuracy, and loss in CSV or graphical formats for easy monitoring.

## Installation
Install QNNLib with a single command:
```bash
pip install qnnlib
```

Note: QNNLib currently supports Python 3.11 due to dependencies on PennyLane 0.37 and TensorFlow 2.15. Updates for Python 3.12 compatibility are under development.

## Example Usage 

### Running QNN on a Simulator

from qnnlib import qnnlib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

qnn = qnnlib(nqubits=8, device_name="lightning.qubit")
qnn.run_experiment(
    data_path='diabetes.csv',
    target='Outcome',
    test_size=0.3,
    model_output_path='qnn_model.h5',
    csv_output_path='training_progress.csv',
    loss_plot_file='loss.png',
    accuracy_plot_file='acc.png',
    batch_size=10,
    epochs=100,
    reps=30,
    scaler=MinMaxScaler(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    seed=1234
)

### Using QNN on IBMQ Quantum Hardware
from qnnlib import qnnlib
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='<IBMQ_API_TOKEN>'
)
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=8)

qnn = qnnlib(nqubits=8, device_name="qiskit.remote", backend=backend)
qnn.run_experiment(
    data_path='diabetes.csv',
    target='Outcome',
    test_size=0.3,
    model_output_path='qnn_model.h5',
    batch_size=10,
    epochs=100,
    reps=30
)
### Loading Pre-trained Models
from qnnlib import qnnlib
import numpy as np

qnn = qnnlib(nqubits=8, device_name="lightning.qubit")
qnn.load_pretrained_qnn(model_file="qnn_model.h5", nqubits=8, reps=30)

sample_input = np.array([[6,148,72,35,0,33.6,0.627,50]])
result = qnn.predict(sample_input)
print(result)