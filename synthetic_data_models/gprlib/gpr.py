import torch
import gpytorch
from typing import Union


class ExactGP(gpytorch.models.ExactGP):
  def __init__(self,
               train_x: torch.Tensor,
               train_y: torch.Tensor,
               likelihood: gpytorch.likelihoods._GaussianLikelihoodBase,
              mean: Union[
                 gpytorch.means.Mean,
                 gpytorch.means.ConstantMean,
                 gpytorch.means.ZeroMean,
                 gpytorch.means.LinearMean,
                 gpytorch.means.MultitaskMean
               ],
               kernel: gpytorch.kernels.Kernel,
               distribution: gpytorch.distributions.Distribution
  ) -> None:
    super(ExactGP, self).__init__(train_x, train_y, likelihood)
    self.mean = mean
    self.kernel = kernel
    self.distribution = distribution

  def forward(self, x: torch.Tensor) -> gpytorch.distributions.Distribution:
    mean_x = self.mean(x)
    covar_x = self.kernel(x)
    return self.distribution(mean_x, covar_x)
