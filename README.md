
# J-BOND Algorithm Hypothesis Testing

This project tests the hypothesis of the J-BOND Algorithm by DeepMind. It involves setting up the environment, optimizing a large language model (LLM), and evaluating the hypothesis by comparing pre- and post-optimization results using the `facebook/bart-large-cnn` model for text summarization.

## Purpose

The purpose of this project is to:
1. Evaluate the performance of the J-BOND Algorithm.
2. Optimize a large language model using the J-BOND Algorithm.
3. Compare the performance of the model before and after optimization.

## Requirements

Before you begin, ensure you have met the following requirements:
- You have installed Python 3.6 or higher.
- You have basic knowledge of Python and programming concepts.
- You have access to a machine with GPU capabilities for training the model efficiently.

## Setup Instructions

Follow these steps to set up and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/j-bond-hypothesis-testing.git
cd j-bond-hypothesis-testing
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Uninstall Incompatible Packages

```bash
pip uninstall -y pyarrow requests datasets
```

### 5. Install Compatible Versions of Required Packages

```bash
pip install pyarrow==14.0.1 requests==2.31.0 datasets==2.1.0
```

### 6. Install Remaining Required Libraries

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers numpy rouge-score tensorboard optuna pytorch-lightning sacrebleu evaluate nltk bert-score
```

### 7. Verify the Installed Versions

```python
import pyarrow
import requests
import datasets
import torch
import transformers
import numpy
import rouge_score
import tensorboard
import optuna
import pytorch_lightning
import sacrebleu
import evaluate
import nltk
import bert_score

print(f"pyarrow version: {pyarrow.__version__}")
print(f"requests version: {requests.__version__}")
print(f"datasets version: {datasets.__version__}")
print(f"torch version: {torch.__version__}")
print(f"transformers version: {transformers.__version__}")
print(f"numpy version: {numpy.__version__}")
print(f"rouge-score version: {rouge_score.__version__}")
print(f"tensorboard version: {tensorboard.__version__}")
print(f"optuna version: {optuna.__version__}")
print(f"pytorch-lightning version: {pytorch_lightning.__version__}")
print(f"sacrebleu version: {sacrebleu.__version__}")
print(f"evaluate version: {evaluate.__version__}")
print(f"nltk version: {nltk.__version__}")
print(f"bert-score version: {bert_score.__version__}")
```

## Running the Project

### 1. Set Up Device Configuration

Ensure CUDA is properly configured for GPU usage.

### 2. Load Model and Tokenizer

The script loads the `facebook/bart-large-cnn` model and its associated tokenizer from the Hugging Face library. This model is used for the text summarization task.

### 3. Preprocess Data

The script includes functions to preprocess the input text data by tokenizing it and ensuring it is truncated and padded to a fixed length.

### 4. Define PyTorch Lightning Module

The `SummarizationModel` class, a custom Lightning module, encapsulates the model, training, and evaluation logic.

### 5. Hyperparameter Optimization

Optuna is used for hyperparameter optimization, defining the search space and training the model with different hyperparameters.

### 6. Training and Evaluation Workflow

The script sets up the device configuration for CUDA, evaluates the baseline model performance, trains the model using the J-BOND Algorithm, and evaluates the model performance after training to measure improvements.

### 7. Ablation Study

The script conducts an ablation study by training a baseline model without the J-BOND Algorithm and comparing the performance metrics.

### 8. Logging and Metrics

TensorBoard is used for logging and visualizing metrics. The script logs and prints detailed evaluation metrics for both pre- and post-training to track improvements.

## Running the Main Script

```bash
python main.py
```

## Project Structure

```
j-bond-hypothesis-testing/
│
├── main.py                  # Main script to run the project
├── requirements.txt         # List of required libraries
├── README.md                # Project documentation
└── ...                      # Other project files and directories
```

## Comments Explaining the Code's Functionality

The code includes detailed comments and docstrings explaining the purpose and logic of each function, making it easy to follow and understand.

## Contributing

To contribute to this project, fork the repository and create a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

By following these instructions, you will be able to set up, run, and understand the project's workflow, ensuring you can effectively test the J-BOND Algorithm hypothesis and evaluate its performance.
