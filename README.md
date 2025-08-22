# Introduction
The purpose of this repo is to provide code and a description of the methods employed by the Dal team in our efforts while participating in the FathomNet 2025 Kaggle competition. Although we experimented with a number of different approaches, there are two methods which we credit with the majority of our progress: distance-weighted cross-entropy loss and ensemble-based self-training.

# Methods
## Distance-Weighted Cross-Entropy
Let \( D \) be a distance matrix of dimension \( N \times N \), where each entry \( d_{ij} \) represents the distance between class \( i \) and class \( j \). In the case of FathomNet, these classes correspond to leaf nodes. Furthermore, since FathomNet is a full-depth hierarchical classification problem (meaning that appropriate annotations must be leaf nodes, and cannot terminate prior to reaching the full depth of the hierarchy), \( D \) is then related to the hierarchical distance, that also evaluates model predictions based on their prediction's "closeness" to ground truth". Therefore, we can also imagine employing \( D \) as a means of more heavily penalizing distance predictions. However there exists a number of issues. Firstly, we cannot minimize for distance directly or even scaling with the model probability outputs, as this tends to be quite unstable, and tends to result in poor performance (we verify this emperically as well). Therefore, we are motivated to mix distance into cross-entropy loss. Standard cross-entropy exists as:

$$
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)
$$

where \( y_i \) is the ground truth label (one-hot encoded), \( \hat{y}_i \) is the predicted probability for class \( i \), and \( N \) is the total number of classes. Since \( y_i \) being negative (zero) means the term is essentially ignored in the sum, cross-entropy does not "care" about which incorrect class gets which predicted probability --- it is "error agnostic". As an example with three classes, if the ground truth is class one, and class zero gets probability \( p \), while class two gets probability \( q \), cross-entropy does not distinguish this scenario from if class zero received a prediction of \( q \), and class two \( p \) instead. However, since errors in the classes for our hierarchy are not symmetric, we need to modify the loss such that it is no longer error agnostic. We introduce a distance-weighted cross-entropy loss:

$$
\mathcal{L}_{\text{DW-CE} = -\sum_{i=1}^{N} \sum_{j=1}^n \tilde{d_{ij}} y_i \log(\hat{\delta}_j)
$$





