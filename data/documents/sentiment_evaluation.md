
# Sentiment Model Evaluation Guide

Evaluate a sentiment classifier using accuracy, precision, recall, F1-score, and a confusion matrix.

Accuracy is the fraction of predictions that match the true labels. Precision measures how often predictions for a class are correct. Recall measures how many examples of a class are identified. The F1-score combines precision and recall.

Macro F1 computes the F1-score independently for each class and averages the scores equally. It is useful when class balance matters.

A confusion matrix compares true labels with predicted labels. Evaluation should include class-level metrics, dataset limitations, and examples of incorrect predictions.

A held-out test set should not be used to select model parameters. Synthetic data can be easier than real customer feedback and may contain artificial patterns. Performance on synthetic examples is not evidence of real-world generalization.
