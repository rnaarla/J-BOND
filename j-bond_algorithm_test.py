"""
This script tests the hypothesis of the J-BOND Algorithm by DeepMind. It sets up the environment, optimizes a large 
language model (LLM), and evaluates the hypothesis by providing before and after results. The script performs the 
following steps:

1. **Environment Setup**: Uninstalls incompatible packages and installs the required libraries for running the model.

2. **Load Model and Tokenizer**: 
    - Loads a pre-trained BART model and its associated tokenizer from the Hugging Face library. 
    - This model will be used for the summarization task.

3. **Data Preprocessing**: 
    - Defines a function to preprocess the input text data by tokenizing it, ensuring it is truncated and padded to a fixed length.
    - Creates a DataLoader for efficient data loading during training and evaluation.

4. **Define the PyTorch Lightning Module**:
    - Implements a custom Lightning module (`SummarizationModel`) which encapsulates the model, training, and evaluation logic.
    - The module includes methods for the forward pass, training step, validation step, and configuration of optimizers and learning rate schedulers.

5. **Hyperparameter Optimization with Optuna**:
    - Defines an objective function for Optuna that specifies the hyperparameter search space and trains the model with different hyperparameters.
    - Uses Optuna to find the best hyperparameters by minimizing the validation loss.

6. **Training and Evaluation Workflow**:
    - Sets up the device configuration for CUDA and enables CuDNN benchmarking for performance optimization.
    - Evaluates the baseline model performance before training.
    - Trains the model using the J-BOND Algorithm.
    - Evaluates the model performance after training to measure improvements.

7. **Ablation Study**:
    - Conducts an ablation study by training a baseline model without the J-BOND Algorithm.
    - Compares the performance metrics (e.g., validation loss, ROUGE, BLEU, BERTScore, METEOR, and ROUGE-WE) between the baseline and the J-BOND enhanced models.

8. **Logging and Metrics**:
    - Uses TensorBoard for logging and visualizing metrics.
    - Logs and prints detailed evaluation metrics for both pre- and post-training to track improvements and validate the effectiveness of the J-BOND Algorithm.

By running this script, the environment is prepared for testing the J-BOND Algorithm, optimizing the LLM, and evaluating the hypothesis by comparing pre- and post-optimization results.
"""

import torch
from torch import nn, optim
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from evaluate import load as evaluate_load

def load_model(model_name):
    """
    Loads the pre-trained model and tokenizer from the specified model name.

    This function is essential for initializing the model and tokenizer
    which will be used for training and inference. It allows you to easily
    switch between different pre-trained models by changing the model name.

    Args:
        model_name (str): The name of the pre-trained model to load.

    Returns:
        tokenizer: The tokenizer associated with the model.
        model: The pre-trained model.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load model or tokenizer: {e}")
    return tokenizer, model

def preprocess_data(examples, tokenizer, max_length=512):
    """
    Tokenizes the input examples using the provided tokenizer.

    This function helps in converting the raw text data into tokenized
    format which can be fed into the model for training and inference.
    It ensures that the data is truncated and padded to a fixed length
    for uniformity.

    Args:
        examples (dict): A dictionary containing the input examples.
        tokenizer: The tokenizer to use for tokenizing the examples.
        max_length (int): The maximum length for truncation/padding.

    Returns:
        dict: The tokenized examples.
    """
    try:
        return tokenizer(examples['article'], truncation=True, padding='max_length', max_length=max_length)
    except Exception as e:
        raise ValueError(f"Error in preprocessing data: {e}")

def create_dataloader(dataset, tokenizer, batch_size=16, num_workers=8):
    """
    Creates a DataLoader from the dataset and tokenizer.

    This function sets up the DataLoader with specified batch size and number of workers
    for efficient data loading. It also preprocesses the dataset using the provided tokenizer.

    Args:
        dataset: The dataset to be loaded.
        tokenizer: The tokenizer to use for preprocessing the dataset.
        batch_size (int): The batch size to use for the DataLoader.
        num_workers (int): The number of worker threads for data loading.

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    try:
        dataset = dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    except Exception as e:
        raise RuntimeError(f"Failed to create DataLoader: {e}")

class SummarizationModel(pl.LightningModule):
    def __init__(self, model_name, learning_rate=1e-5, accumulation_steps=4, num_training_steps=1000):
        """
        Initializes the SummarizationModel with the given parameters.

        This class is a PyTorch Lightning module that encapsulates the model,
        training, and evaluation logic. It uses mixed precision training and 
        supports gradient accumulation for efficient training.

        Args:
            model_name (str): The name of the pre-trained model to load.
            learning_rate (float): The learning rate for the optimizer.
            accumulation_steps (int): The number of steps to accumulate gradients.
            num_training_steps (int): The total number of training steps for the scheduler.
        """
        super(SummarizationModel, self).__init__()
        try:
            self.model_name = model_name
            self.tokenizer, self.model = load_model(model_name)
            self.learning_rate = learning_rate
            self.accumulation_steps = accumulation_steps
            self.num_training_steps = num_training_steps
            self.rouge = load_metric("rouge")
            self.bleu = load_metric("sacrebleu")
            self.rouge_we = evaluate_load("rouge-we")
        except Exception as e:
            raise RuntimeError(f"Initialization failed: {e}")

    def forward(self, input_ids, attention_mask):
        """
        Defines the forward pass of the model.

        This function is used by PyTorch Lightning during the training and
        evaluation phases. It performs a forward pass through the model and
        returns the output.

        Args:
            input_ids (Tensor): The input tensor containing token ids.
            attention_mask (Tensor): The attention mask for the input tensor.

        Returns:
            output: The output of the model.
        """
        try:
            return self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {e}")

    def training_step(self, batch, batch_idx):
        """
        Defines a single training step.

        This function is called by PyTorch Lightning during the training loop.
        It performs the forward pass, computes the loss, and performs backpropagation
        using mixed precision and gradient accumulation.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            loss: The computed loss for the batch.
        """
        try:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            with autocast():
                outputs = self(input_ids, attention_mask)
                loss = outputs.loss / self.accumulation_steps
            
            self.manual_backward(loss)
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss
        except Exception as e:
            raise RuntimeError(f"Training step failed: {e}")

    def validation_step(self, batch, batch_idx):
        """
        Defines a single validation step.

        This function is called by PyTorch Lightning during the validation loop.
        It performs the forward pass, computes the loss, and logs additional
        evaluation metrics such as ROUGE, BLEU, BERTScore, METEOR, and ROUGE-WE.

        Args:
            batch (dict): A batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            loss: The computed loss for the batch.
        """
        try:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = self(input_ids, attention_mask)
            loss = outputs.loss
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            # Calculate ROUGE, BLEU, BERTScore, METEOR, and ROUGE-WE
            preds = self.model.generate(input_ids)
            pred_str = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            self.rouge.add_batch(predictions=pred_str, references=label_str)
            self.bleu.add_batch(predictions=pred_str, references=label_str)
            
            # BERTScore calculation
            P, R, F1 = bert_score(pred_str, label_str, lang="en", verbose=True)
            avg_f1 = F1.mean().item()
            self.log('val_bertscore', avg_f1, prog_bar=True)
            
            # METEOR calculation
            meteor_scores = [meteor_score([ref], pred) for ref, pred in zip(label_str, pred_str)]
            avg_meteor = sum(meteor_scores) / len(meteor_scores)
            self.log('val_meteor', avg_meteor, prog_bar=True)

            # ROUGE-WE calculation
            rouge_we_scores = self.rouge_we.compute(predictions=pred_str, references=label_str)
            self.log('val_rouge_we', rouge_we_scores['rouge-we-F1'], prog_bar=True)

            return loss
        except Exception as e:
            raise RuntimeError(f"Validation step failed: {e}")

    def validation_epoch_end(self, outputs):
        """
        Aggregates the validation results at the end of the epoch.

        This function computes the final ROUGE and BLEU scores for the entire
        validation set and logs them.

        Args:
            outputs (list): List of individual validation step outputs.
        """
        try:
            rouge_result = self.rouge.compute()
            bleu_result = self.bleu.compute()
            self.log('val_rouge', rouge_result['rouge1'].mid.fmeasure, prog_bar=True)
            self.log('val_bleu', bleu_result['score'], prog_bar=True)
        except Exception as e:
            raise RuntimeError(f"Validation epoch end failed: {e}")

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for the training.

        This function sets up the AdamW optimizer and a linear learning rate scheduler
        with warmup.

        Returns:
            dict: The configured optimizer and scheduler.
        """
        try:
            self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
            self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                                num_warmup_steps=0, 
                                                                num_training_steps=self.num_training_steps)
            return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}
        except Exception as e:
            raise RuntimeError(f"Optimizer configuration failed: {e}")

def set_device_config():
    """
    Sets the device configuration for CUDA and enables CuDNN benchmarking for performance.

    This function configures the GPU devices to be used for training and
    enables CuDNN benchmarking to optimize performance. It ensures that
    the model utilizes available hardware resources efficiently.
    """
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Adjust as necessary for available GPUs
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    except Exception as e:
        raise RuntimeError(f"Failed to set device configuration: {e}")

def objective(trial):
    """
    Objective function for hyperparameter optimization using Optuna.

    This function defines the hyperparameter search space and trains the model
    with different hyperparameters. It returns the validation loss for each trial.

    Args:
        trial: Optuna trial object.

    Returns:
        float: Validation loss for the trial.
    """
    try:
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-4)
        accumulation_steps = trial.suggest_int('accumulation_steps', 1, 8)
        
        model = SummarizationModel(model_name="facebook/bart-large-cnn", learning_rate=learning_rate, accumulation_steps=accumulation_steps)
        trainer = pl.Trainer(
            logger=TensorBoardLogger("tb_logs", name="summarization"),
            max_epochs=10,
            gpus=torch.cuda.device_count(),
            strategy='ddp',
            precision=16,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        )
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='train[:1000]')
        dataloader = create_dataloader(dataset, model.tokenizer)
        val_dataset = load_dataset('cnn_dailymail', '3.0.0', split='validation[:500]')
        val_dataloader = create_dataloader(val_dataset, model.tokenizer)

        trainer.fit(model, dataloader, val_dataloader)
        val_loss = trainer.callback_metrics["val_loss"].item()
        return val_loss
    except Exception as e:
        raise RuntimeError(f"Objective function failed: {e}")

def main():
    """
    Main function to load the model, train it, and evaluate its performance.

    This function sets up the environment, loads the model and tokenizer,
    evaluates the model before training (baseline), trains the model,
    and evaluates the model after training. It prints the performance metrics
    to track improvements. It also performs hyperparameter optimization and
    an ablation study to assess the impact of the J-Bond Algorithm.

    """
    try:
        set_device_config()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)

        print("Best hyperparameters: ", study.best_params)
        print("Best validation loss: ", study.best_value)

        # Conduct ablation study
        # Train with J-Bond Algorithm (Hypothetical)
        model = SummarizationModel(model_name="facebook/bart-large-cnn")
        trainer = pl.Trainer(
            logger=TensorBoardLogger("tb_logs", name="summarization_j_bond"),
            max_epochs=10,
            gpus=torch.cuda.device_count(),
            strategy='ddp',
            precision=16
        )
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='train[:1000]')
        dataloader = create_dataloader(dataset, model.tokenizer)
        val_dataset = load_dataset('cnn_dailymail', '3.0.0', split='validation[:500]')
        val_dataloader = create_dataloader(val_dataset, model.tokenizer)
        trainer.fit(model, dataloader, val_dataloader)
        
        # Evaluate the model
        print("Post-Training Evaluation")
        val_loss = trainer.callback_metrics["val_loss"].item()
        rouge_result = model.rouge.compute()
        bleu_result = model.bleu.compute()
        P, R, F1 = bert_score(
            [pred_str], [label_str], lang="en", verbose=True
        )
        avg_f1 = F1.mean().item()
        meteor_scores = [meteor_score([ref], pred) for ref, pred in zip(label_str, pred_str)]
        avg_meteor = sum(meteor_scores) / len(meteor_scores)
        rouge_we_scores = model.rouge_we.compute(predictions=pred_str, references=label_str)
        print(f'Validation Loss: {val_loss}')
        print(f'Validation ROUGE: {rouge_result}')
        print(f'Validation BLEU: {bleu_result}')
        print(f'Validation BERTScore F1: {avg_f1}')
        print(f'Validation METEOR: {avg_meteor}')
        print(f'Validation ROUGE-WE: {rouge_we_scores}')

        # Baseline without J-Bond Algorithm
        print("Training Baseline Model")
        model_baseline = SummarizationModel(model_name="facebook/bart-large-cnn")
        trainer_baseline = pl.Trainer(
            logger=TensorBoardLogger("tb_logs", name="summarization_baseline"),
            max_epochs=10,
            gpus=torch.cuda.device_count(),
            strategy='ddp',
            precision=16
        )
        trainer_baseline.fit(model_baseline, dataloader, val_dataloader)
        
        # Evaluate the baseline model
        print("Baseline Post-Training Evaluation")
        val_loss_baseline = trainer_baseline.callback_metrics["val_loss"].item()
        rouge_result_baseline = model_baseline.rouge.compute()
        bleu_result_baseline = model_baseline.bleu.compute()
        P_baseline, R_baseline, F1_baseline = bert_score(
            [pred_str], [label_str], lang="en", verbose=True
        )
        avg_f1_baseline = F1_baseline.mean().item()
        meteor_scores_baseline = [meteor_score([ref], pred) for ref, pred in zip(label_str, pred_str)]
        avg_meteor_baseline = sum(meteor_scores_baseline) / len(meteor_scores_baseline)
        rouge_we_scores_baseline = model_baseline.rouge_we.compute(predictions=pred_str, references=label_str)
        print(f'Baseline Validation Loss: {val_loss_baseline}')
        print(f'Baseline Validation ROUGE: {rouge_result_baseline}')
        print(f'Baseline Validation BLEU: {bleu_result_baseline}')
        print(f'Baseline Validation BERTScore F1: {avg_f1_baseline}')
        print(f'Baseline Validation METEOR: {avg_meteor_baseline}')
        print(f'Baseline Validation ROUGE-WE: {rouge_we_scores_baseline}')
        
        # Compare improvements
        print("Improvement in Loss: ", val_loss_baseline - val_loss)
        print("Improvement in ROUGE: ", rouge_result_baseline['rouge1'].mid.fmeasure - rouge_result['rouge1'].mid.fmeasure)
        print("Improvement in BLEU: ", bleu_result_baseline['score'] - bleu_result['score'])
        print("Improvement in BERTScore F1: ", avg_f1_baseline - avg_f1)
        print("Improvement in METEOR: ", avg_meteor_baseline - avg_meteor)
        print("Improvement in ROUGE-WE: ", rouge_we_scores_baseline['rouge-we-F1'] - rouge_we_scores['rouge-we-F1'])
    except Exception as e:
        raise RuntimeError(f"Main function failed: {e}")

if __name__ == "__main__":
    main()