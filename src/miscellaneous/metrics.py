import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def loss_function(output: torch.Tensor,  # (b, n_class)
                  target: torch.Tensor,  # (b)
                  ) -> torch.Tensor:
    softmax = F.softmax(output, dim=-1)  # (b, n_class)
    loss = F.binary_cross_entropy(softmax[:, 1], target.float(), reduction='sum')
    """
        # for document classification
        loss = F.crossentropy(output, target, reduction='none').sum() / b
    """
    return loss


def accuracy(output: torch.Tensor,  # (b, n_class)
             target: torch.Tensor   # (b)
             ) -> int:
    prediction = torch.argmax(output, dim=1)
    return (prediction == target).sum().item()


def f_measure(output: torch.Tensor,  # (b, n_class)
              target: torch.Tensor   # (b)
              ) -> int:
    prediction = torch.argmax(output, dim=1)
    f_score = f1_score(target.cpu(), prediction.cpu(), average='macro')
    return f_score
