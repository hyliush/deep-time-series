from torch.autograd import Variable
import torch
import torch.nn as nn
def _Normal_loss(mu: Variable, sigma: Variable, labels: Variable):
    '''
    
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    zero_index = (labels != 0)
    distribution = torch.distributions.normal.Normal(mu[zero_index], sigma[zero_index])
    likelihood = distribution.log_prob(labels[zero_index])
    return -torch.mean(likelihood)
    
def Normal_loss(mu_sigma: Variable, labels: Variable):
    '''
       mu_sigma: batch_size * seq_len * 2,
       labels: batch_size * seq_len * 1
    '''
    mu, sigma = mu_sigma[..., 0].transpose(1, 0), mu_sigma[..., 1].transpose(1, 0)
    labels = labels.squeeze(dim=-1).transpose(1, 0)
    if mu.ndim == 1:
        loss = _Normal_loss(mu, sigma, labels)
    if mu.ndim == 2:
        loss = 0
        for i in range(mu.shape[0]):
            loss += _Normal_loss(mu[i], sigma[i], labels[i])
    return loss



class OZELoss(nn.Module):
    """Custom loss for TRNSys metamodel.

    Compute, for temperature and consumptions, the intergral of the squared differences
    over time. Sum the log with a coeficient ``alpha``.

    .. math::
        \Delta_T = \sqrt{\int (y_{est}^T - y^T)^2}

        \Delta_Q = \sqrt{\int (y_{est}^Q - y^Q)^2}

        loss = log(1 + \Delta_T) + \\alpha \cdot log(1 + \Delta_Q)

    Parameters:
    -----------
    alpha:
        Coefficient for consumption. Default is ``0.3``.
    """

    def __init__(self, reduction: str = 'mean', alpha: float = 0.3):
        super().__init__()

        self.alpha = alpha
        self.reduction = reduction

        self.base_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_true: torch.Tensor,
                y_pred: torch.Tensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Parameters
        ----------
        y_true:
            Target value.
        y_pred:
            Estimated value.

        Returns
        -------
        Loss as a tensor with gradient attached.
        """
        delta_Q = self.base_loss(y_pred[..., :-1], y_true[..., :-1])
        delta_T = self.base_loss(y_pred[..., -1], y_true[..., -1])

        if self.reduction == 'none':
            delta_Q = delta_Q.mean(dim=(1, 2))
            delta_T = delta_T.mean(dim=(1))

        return torch.log(1 + delta_T) + self.alpha * torch.log(1 + delta_Q)

class QuantileLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.reshape(torch.tensor([0.1, 0.5, 0.9]), (1, 3)).to("cuda")
        self.zero = torch.tensor(0.0).to("cuda")
        self.one = torch.tensor(1.0).to("cuda")
    def forward(self, y_pred, y_true):  
        y_true = y_true.unsqueeze(dim=-1)
        e = torch.subtract(y_true, y_pred)
        L = torch.multiply(self.q, torch.maximum(self.zero, e)) + torch.multiply(self.one-self.q, torch.maximum(self.zero, -e))
        return torch.mean(L)

import math
class GaussianLoss(nn.Module):
    '''negative log-likelihood'''
    def __init__(self):
        super().__init__()
    def forward(self, y_pred, y_true, softZero=1e-4):
        mu, sigma = y_pred[..., 0], y_pred[..., 1]
        logGaussian =  -torch.log(2*math.pi*(sigma**2)+softZero)/2 -(y_true- mu)**2/(2*(sigma**2)+softZero)
        res = -torch.sum(logGaussian, axis= 0)
        return torch.mean(res)