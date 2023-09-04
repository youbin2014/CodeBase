import torch


def accuracy(outputs, labels):
    """
    Compute the accuracy.

    Parameters:
    - outputs (torch.Tensor): Model logits. Shape: [batch_size, num_classes].
    - labels (torch.Tensor): Ground truth labels. Shape: [batch_size].

    Returns:
    - accuracy (float): Accuracy value.
    """
    y_pred = outputs.argmax(dim=1)
    correct = (y_pred == labels).float().sum().item()
    return correct / len(labels)*100


def top_k_accuracy(outputs, labels, k=5):
    """
    Compute the top-k accuracy.

    Parameters:
    - outputs (torch.Tensor): Model logits. Shape: [batch_size, num_classes].
    - labels (torch.Tensor): Ground truth labels. Shape: [batch_size].
    - k (int): The top-k predictions to consider.

    Returns:
    - accuracy (float): Top-k accuracy value.
    """
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True).item()
    return correct_k / len(labels)*100


def f1_score(outputs, labels, threshold=0.5):
    """
    Compute the F1 score for binary classification.

    Note: Use this only if your classification task is binary.

    Parameters:
    - outputs (torch.Tensor): Model logits. Shape: [batch_size, 1].
    - labels (torch.Tensor): Ground truth labels. Shape: [batch_size].
    - threshold (float): Threshold to classify between the two classes.

    Returns:
    - f1 (float): F1 score.
    """
    y_pred = torch.sigmoid(outputs)
    y_pred_bin = (y_pred > threshold).float()

    tp = (y_pred_bin * labels).sum().item()
    tn = ((1 - y_pred_bin) * (1 - labels)).sum().item()
    fp = (y_pred_bin * (1 - labels)).sum().item()
    fn = ((1 - y_pred_bin) * labels).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1

# Add more metrics as needed.
