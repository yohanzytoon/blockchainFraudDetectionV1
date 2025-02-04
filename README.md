# Fraud Detection using Graph Neural Networks

## Overview

This project implements an advanced Graph Convolutional Network (GCN) model to detect fraudulent Bitcoin transactions using the Elliptic dataset. The dataset represents Bitcoin transactions as nodes and their connections as edges, allowing for a graph-based approach to fraud detection.

## Features

- **Data Loading & Cleaning:**
  - Load transaction and edge data from CSV files.
  - Remove low-variance features.
  - Visualize the target class distribution.

- **Graph Splitting Strategies:**
  - **Time-step Split:** Creates subgraphs for each time step.
  - **Time-group Split:** Merges consecutive time steps into a single graph.
  - **Random Split (Transductive & Inductive):** Randomly assigns nodes to training, validation, and test sets.
  - **Community-based Split:** Uses spectral clustering and K-Means for community-based partitioning.

- **Advanced GCN Model:**
  - Multi-layer GCN with residual connections.
  - Batch normalization, dropout, and a learning rate scheduler.
  - Handles different graph splitting strategies.

- **Training & Evaluation:**
  - Early stopping and model checkpointing.
  - Evaluation metrics including classification reports, AUC-ROC scores, and confusion matrices.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yohanzytoon/blockchainFraudDetection.git
   cd blockchainFraudDetection
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure PyTorch Geometric dependencies are correctly installed by following [PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Data Preparation

The dataset consists of:
- `nodes.csv`: Node features and labels (fraudulent or legitimate transactions).
- `edges.csv`: Edges representing Bitcoin flows between transactions.

To use the dataset:
1. Place `nodes.csv` and `edges.csv` inside the `data/` directory.
2. Run the notebook to preprocess and split the data.

## Running the Project

Run the training and evaluation notebook:
```bash
jupyter notebook fraud_detection_GCN.ipynb
```

## Model Architecture

The advanced GCN model consists of:
- Input layer mapping raw features to a hidden dimension.
- Multiple GCN layers with batch normalization and dropout.
- Residual connections for deeper learning.
- An output layer with LogSoftmax activation for classification.

## Evaluation Metrics

The model is evaluated using:
- **Classification Report:** Precision, recall, F1-score.
- **ROC-AUC Score:** Measures classification performance.
- **Confusion Matrix:** Provides insights into misclassification.

## Results and Visualization

### Training Results

```
Starting training...
Epoch 1: Loss = 0.7763, Val AUC = 0.5353
Epoch 2: Loss = 0.6223, Val AUC = 0.5039
Epoch 3: Loss = 0.6442, Val AUC = 0.5978
Epoch 4: Loss = 0.5372, Val AUC = 0.7079
Epoch 5: Loss = 0.4380, Val AUC = 0.7292
Epoch 6: Loss = 0.3466, Val AUC = 0.7643
Epoch 7: Loss = 0.3006, Val AUC = 0.7880
Epoch 8: Loss = 0.2924, Val AUC = 0.7531
Epoch 9: Loss = 0.2920, Val AUC = 0.7044
Epoch 10: Loss = 0.2778, Val AUC = 0.6594
Epoch 11: Loss = 0.2564, Val AUC = 0.5916
Epoch 12: Loss = 0.2469, Val AUC = 0.5218
Epoch 13: Loss = 0.2507, Val AUC = 0.5017
Epoch 14: Loss = 0.2546, Val AUC = 0.4998
Epoch 15: Loss = 0.2425, Val AUC = 0.5008
Epoch 16: Loss = 0.2401, Val AUC = 0.5133
Epoch 17: Loss = 0.2287, Val AUC = 0.5577
Early stopping triggered.
```

## Future Work
- Implementing a more complex dataset for fraud detection.
- Exploring advanced GNN architectures such as GAT, GraphSAGE, and Transformer-based models.
- Conducting extensive hyperparameter tuning and experimenting with larger datasets.

## Saving and Loading Models

The best-performing model is automatically saved:
```bash
best_advanced_gcn_model.pt
```
To load the saved model:
```python
model.load_state_dict(torch.load('best_advanced_gcn_model.pt'))
```


## License
This project is open-source and available under the MIT License.

## Acknowledgments
This project is built using PyTorch Geometric and inspired by research in blockchain fraud detection.

---
For questions or collaborations, contact yohanze@icloud.com

