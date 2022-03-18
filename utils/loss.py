from torch.autograd import Variable
import torch

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