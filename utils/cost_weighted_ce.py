import torch
import torch.nn as nn

def assign_class_weights(df):
    freq_dist = df["label"].value_counts().sort_index()
    
    total_samples = freq_dist.sum()
    n_classes = len(freq_dist)
    class_weights = {}
    
    # Compute weight for each class
    for cls, count in freq_dist.items():
        # Avoid division by zero (if a class count is 0, though it should not occur in this context)
        if count > 0:
            class_weights[cls] = total_samples / (n_classes * count)
        else:
            class_weights[cls] = 0.0
    return class_weights

def compute_applied_weights(
          cost_matrix, 
          inputs,
          targets
          ):
        assert inputs.shape[1] == cost_matrix.shape[0], \
            "Number of classes in inputs must match cost matrix."
        
        # Expand the cost matrix to match the batch size
        batch_size = inputs.shape[0]

        cost_matrix_expanded = cost_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        batch_indices = torch.arange(batch_size, device=targets.device)

        # Compute weights to be applied
        applied_weights = cost_matrix_expanded[batch_indices, targets]

        return batch_indices, applied_weights

class CostWeightedCELossWithLogits(nn.Module):
    def __init__(
            self, 
            cost_matrix, 
            eps=1e-8
            ):
        super(CostWeightedCELossWithLogits, self).__init__()
        self.cost_matrix = cost_matrix
        self.softmax = nn.Softmax(dim=1)
        self.eps = eps

        assert self.cost_matrix.shape[0] == self.cost_matrix.shape[1], \
            "Cost matrix must be square."

    def forward(self, inputs, targets):
        batch_indices, applied_weights = compute_applied_weights(
            self.cost_matrix, 
            inputs, 
            targets
            )

        inputs_probs = self.softmax(inputs)
        inv_probs = 1 - inputs_probs

        log_probs = torch.log(inputs_probs + self.eps)
        log_inv_probs = torch.log(inv_probs + self.eps)

        # Select the log probs for the correct classes,
        # And select the log inv probs for the incorrect classes,
        # Combine them into one matrix of batch_size x num_classes, log_args
        selected_log_probs = log_probs[batch_indices, targets]

        inv_mask = torch.ones_like(log_probs, dtype=torch.bool)
        inv_mask[batch_indices, targets] = False

        selected_log_inv_probs = log_inv_probs[inv_mask]

        # Insert the selected log probs and log inv probs into a new tensor
        log_args = torch.zeros_like(log_probs)

        log_args[batch_indices, targets] = selected_log_probs
        log_args[inv_mask] = selected_log_inv_probs

        loss = applied_weights * log_args
        loss = -torch.sum(loss, dim=1)
        loss = torch.mean(loss)

        return loss
        
class DistanceLossWithLogits(CostWeightedCELossWithLogits):
    def __init__(self, cost_matrix, eps=1e-8):
        super(DistanceLossWithLogits, self).__init__(cost_matrix, eps)

    def forward(self, inputs, targets):
        _, applied_weights = compute_applied_weights(
            self.cost_matrix, 
            inputs, 
            targets
            )
         
        inputs_probs = self.softmax(inputs)

        loss = applied_weights * inputs_probs
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)
        return loss

class CalcDistance():
    def __init__(self, cost_matrix):
        self.cost_matrix = cost_matrix

    def __call__(self, predictions, targets):
        batch_size = predictions.shape[0]

        cost_matrix_expanded = self.cost_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        batch_indices = torch.arange(batch_size, device=targets.device)

        selected_cost_rows = cost_matrix_expanded[batch_indices, targets]

        selected_costs = selected_cost_rows[batch_indices, predictions].float()

        selected_cost = selected_costs.mean()

        return selected_cost.cpu().item()

        
