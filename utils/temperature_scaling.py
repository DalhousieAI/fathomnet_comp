import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
"""
https://github.com/gpleiss/temperature_scaling/tree/master
"""


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, save_debug_plots=False):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        scaled_logits = self.temperature_scale(logits)
        after_temperature_nll = nll_criterion(scaled_logits, labels).item()
        after_temperature_ece = ece_criterion(scaled_logits, labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        if save_debug_plots:
            # Save calibration curves
            self.plot_calibration_curves_save(
                logits_before=logits,
                logits_after=scaled_logits,
                labels=labels,
            )

        return self

    def plot_calibration_curves_save(self, logits_before, logits_after, labels, n_bins=15,
                                    before_save_path="calibration_before.png",
                                    after_save_path="calibration_after.png"):
        """
        Save calibration curves (confidence vs. accuracy) as image files for logits
        before and after temperature scaling.

        Args:
            logits_before (Tensor): Model logits before temperature scaling, shape (N, num_classes).
            logits_after (Tensor): Model logits after temperature scaling, shape (N, num_classes).
            labels (Tensor): True labels, shape (N,).
            n_bins (int): Number of bins to compute average confidence and accuracy.
            before_save_path (str): File path to save the plot before temperature scaling.
            after_save_path (str): File path to save the plot after temperature scaling.
        """
        # Compute softmax probabilities and extract max confidences and predictions.
        softmax_before = F.softmax(logits_before, dim=1)
        softmax_after  = F.softmax(logits_after, dim=1)
        
        confidences_before, predictions_before = torch.max(softmax_before, dim=1)
        accuracies_before = predictions_before.eq(labels).float()
        
        confidences_after, predictions_after = torch.max(softmax_after, dim=1)
        accuracies_after = predictions_after.eq(labels).float()
        
        # Set up bins.
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1].numpy()
        bin_uppers = bin_boundaries[1:].numpy()
        
        avg_confidences_before = []
        accuracies_in_bin_before = []
        avg_confidences_after = []
        accuracies_in_bin_after = []
        
        # Average confidence and accuracy for each bin (for both before and after).
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # For logits before temperature scaling.
            in_bin_before = (confidences_before > bin_lower) & (confidences_before <= bin_upper)
            if in_bin_before.sum() > 0:
                avg_conf_before = confidences_before[in_bin_before].mean().item()
                acc_before = accuracies_before[in_bin_before].mean().item()
            else:
                avg_conf_before = (bin_lower + bin_upper) / 2
                acc_before = 0.0
            avg_confidences_before.append(avg_conf_before)
            accuracies_in_bin_before.append(acc_before)
            
            # For logits after temperature scaling.
            in_bin_after = (confidences_after > bin_lower) & (confidences_after <= bin_upper)
            if in_bin_after.sum() > 0:
                avg_conf_after = confidences_after[in_bin_after].mean().item()
                acc_after = accuracies_after[in_bin_after].mean().item()
            else:
                avg_conf_after = (bin_lower + bin_upper) / 2
                acc_after = 0.0
            avg_confidences_after.append(avg_conf_after)
            accuracies_in_bin_after.append(acc_after)
        
        # Plot and save the calibration curve BEFORE temperature scaling.
        plt.figure(figsize=(6, 5))
        plt.plot(avg_confidences_before, accuracies_in_bin_before, marker='o',
                label='Before Temperature Scaling', color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Calibration Curve (Before Temperature Scaling)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(before_save_path)
        plt.close()

        # Plot and save the calibration curve AFTER temperature scaling.
        plt.figure(figsize=(6, 5))
        plt.plot(avg_confidences_after, accuracies_in_bin_after, marker='o',
                label='After Temperature Scaling', color='green')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Calibration Curve (After Temperature Scaling)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(after_save_path)
        plt.close()


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
