# Caltech 101 Classification using CLIP and PyTorch Lightning

This project classifies the Caltech 101 dataset using the CLIP model for feature extraction and a lightweight neural network for classification. The task involves:
- Utilizing CLIP embeddings for classification.
- Training the classifier using PyTorch Lightning.
- Logging metrics with wandb.
- Running experiments on Google Colab.

## Link to wandb.ai training result
https://wandb.ai/taylortianluwang-cornell-university/clip-caltech101?nw=nwusertaylortianluwang

## Summary of the Project
### Process:
- Dataset: Used the Caltech 101 dataset, splitting it into training and validation sets.
- CLIP Model: Leveraged the pre-trained CLIP model to extract 512-dimensional embeddings for each image. The CLIP model remained frozen throughout the training process.
- Classifier: Designed a lightweight neural network (an MLP) to classify the extracted embeddings into 101 categories.

### Model Architecture:
- Input Layer: 512-dimensional feature vectors (embeddings from CLIP).
- Hidden Layer: One fully connected layer with 256 neurons, ReLU activation, and dropout for regularization.
- Output Layer: 101 neurons with softmax activation to predict class probabilities.

### Training Results:
- Train Loss: The training loss showed high variance initially but generally decreased over epochs, indicating learning.
- Validation Loss: The validation loss fluctuated but stabilized near 0.12, showing moderate generalization.

### Future Improvements:
For this simulation, max_epochs is set to 10 due to GPU limitation, which negatively affected the training results. From the results, we see that raining could benefit from hyperparameter tuning (e.g., learning rate adjustment or additional epochs). Minor fluctuations in validation loss suggest possible underfitting or sensitivity to the dataset size.