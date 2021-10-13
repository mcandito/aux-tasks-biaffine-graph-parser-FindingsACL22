#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# losses and submodules


"""

## Custom loss : binary cross-entropy over a matrix of scores, with padding cells
"""

from torch import Tensor
from typing import Callable, Optional

class BinaryHingeLoss_with_mask(nn.Module):
    """
    sum of binary Hinge loss on all over all the potential arcs

    default margin = 1

    see examples of losses in code https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
    """
    __constants__ = ['margin', 'reduction']
    margin: float

    def __init__(self, margin: float = 1.0) -> None:
        super(BinaryHingeLoss_with_mask, self).__init__()
        self.margin = margin

    def forward(self, arc_scores: Tensor, target_arc_adja: Tensor, mask: Tensor) -> Tensor:
        r"""
        Sums the binary hinge loss over all the potential arcs:
        
        sum over all gold arcs a, with score s_a : of max(0, margin - s_a)
        plus
        sum over all gold non arcs a, with score s_a : of max(0, s_a - margin)
        
        Enforce that each gold arc gets a score > margin, 
        and that each gold non arc gets a score < -margin
        
        """
        non_gov = (1 - target_arc_adja) * mask

        s = (arc_scores - self.margin)

        # gold arcs not reaching margin
        loss  = torch.sum(-s * ( (arc_scores <  self.margin).int() * target_arc_adja) )

        # gold non arcs with score above -margin
        loss += torch.sum( s * ( (arc_scores > -self.margin).int() * non_gov) )

        return loss


class BCEWithLogitsLoss_with_mask(nn.BCEWithLogitsLoss):
    r""" Customized BCEWithLogitsLoss
    (cf. https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss)
    allowing to apply a mask to mask padded items

    BCEWithLogitsLoss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    BCEWithLogitsLoss_with_mask:
      - If mask is provided in forxard, zero cells in mask are used to set loss to zero for these cells
      - loss of positive examples (1 examples) is multiplied by pos_weight_scalar 

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        pos_weight_scalar (float, optional): a weight for the loss of positive examples
                
    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as input.

    """
    def __init__(self, reduction: str = 'sum', pos_weight_scalar: Optional[float] = None) -> None:
        # NB: pos_weight_scalar different from the pos_weight existing in BCEWithLogitsLoss
        # pos_weight_scalar : weight applied to loss of positive examples

        # see code of BCEWithLogitsLoss (the super of BCEWithLogitsLoss_with_mask)
        super(BCEWithLogitsLoss_with_mask, self).__init__(reduction=reduction, pos_weight=None)
        
        self.pos_weight_scalar = pos_weight_scalar

    def forward(self, input: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """ 
        for each cell in input x and target y
        l_n = - ( pos_weight_scalar * y_n * log(sigma(x_n)) + (1 - y_n) * log (1 - sigma(x_n)) )
        """
        
        # non reduced version
        loss = F.binary_cross_entropy_with_logits(input, target,
                                                  reduction='none')
        # apply weighting to positive examples (cf. unbalanced)
        if self.pos_weight_scalar is not None:
            # target is a 0/1 matrix
            # Trick to get weight = 1                 for neg examples (0 in target)
            #          and weight = pos_weight_scalar for pos examples (1 in target)
            weights = ((self.pos_weight_scalar - 1) * target) + 1
            loss = loss * weights

        # apply masking
        if mask is not None:
            loss = loss * mask

        if self.reduction == 'sum':
            return torch.sum(loss)
        if self.reduction == 'mean':
            return torch.mean(loss)
        return loss


class MSELoss_with_mask(nn.MSELoss):
    r""" MSELoss with mask for padded tokens
    see code of MSELoss https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
      super(MSELoss_with_mask, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
      # apply without reduction first
      loss = F.mse_loss(input, target, reduction='none')
      # apply masking
      if mask is not None:
          loss = loss * mask
      if self.reduction == 'sum':
          return torch.sum(loss)
      if self.reduction == 'mean':
          return torch.mean(loss)
      return loss

class CosineLoss_with_mask(nn.Module):
    r""" Input and target are vectors which similarity we wish to maximize, 
        
        Hence we take as loss: - cosine(input, target)
    """
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(CosineLoss_with_mask, self).__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
      # input : b, d, l
      # target : b, d, l
      loss = -F.cosine_similarity(input, target, dim=2) # b, d

      # apply masking
      if mask is not None:
          loss = loss * mask
      if self.reduction == 'sum':
          return torch.sum(loss)
      if self.reduction == 'mean':
          return torch.mean(loss)
      return loss

class L2DistanceLoss_with_mask(nn.Module):
    r""" Input and target are vectors whose L2 dist we wish to maximize, 
        
        Hence we take as loss: || input - target||**2
    """
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(L2DistanceLoss_with_mask, self).__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
      # input : b*d, l+1
      # target : b*d, l+1
      loss = F.pairwise_distance(input, target, p=2, keepdim=False) # b*d

      # apply masking
      if mask is not None:
          loss = loss * mask
      if self.reduction == 'sum':
          return torch.sum(loss)
      if self.reduction == 'mean':
          return torch.mean(loss)
      return loss

    
"""## MLP and Biaffine modules"""

#a = torch.empty(2,3,4,device='cuda')
#print(a)
#print(a.dtype)
#print(type(a))

class MLP(nn.Module):
    """ MLP with single hidden layer, with dropout"""
    def __init__(self, input_size, hidden_size, output_size, activation='ReLU', dropout=0.25):
        super(MLP, self).__init__()
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, output_size)
        #self.g = nn.ReLU()
        self.g = getattr(nn, activation)()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self,input):
        #print("W2", self.W2.weight.shape)
        a = self.g(self.W1(input))
        #print("after W1 and activation", a.shape)
        b = self.dropout(a)
        #print("after dropout", b.shape)
        c = self.W2(b)
        #print("after W2", c.shape)
        return c
        #return self.W2(self.dropout(self.g(self.W1(input))))

class MLP_out_hidden(MLP):
    """ MLP with single hidden layer, with dropout, outputing both the output and the hidden layer"""
        
    def forward(self, input):
        a = self.g(self.W1(input))
        b = self.dropout(a)
        c = self.W2(b)
        return c, a

class BiAffine(nn.Module):
    """
Biaffine attention layer (Dozat and Manning, 2017):
applies to 2 input tensors of shape [batch_size, seq_len, biaffine_size].

Returns score matrices of shape 
- [batch_size, num_scores_per_arc=nb of labels, seq_len, seq_len] for label scores
  S(batch_k, l, i, j) = score of sample k in batch, label l, head word i, dep word j
  
- [batch_size, seq_len, seq_len] for arc scores (if num_scores_per_arc=1, axis=1 is squeezed)
  S(batch_k, i, j) = score of sample k in batch, head word i, dep word j

NB: actually bias not implemented, hence bilinear rather than biaffine
"""
    def __init__(self, device, head_size, dep_size, num_scores_per_arc=1, use_bias=False):
        super(BiAffine, self).__init__()
        
        self.device = device
        self.num_scores_per_arc = num_scores_per_arc
        self.use_bias = use_bias
        if use_bias:
            head_size += 1
            dep_size += 1
        self.U = nn.Parameter(torch.empty(num_scores_per_arc,
                                           head_size,
                                           dep_size,
                                           device=device))
        nn.init.xavier_uniform_(self.U)
        
    def forward(self, Hh, Hd):
        """
        Input: Hh = head tensor      shape  [batch_size, n,  head_size]
               Hd = dependent tensor shape  [batch_size, n,  dep_size]
                                with n = length of sequence,
                                     head_size = in size for head words
                                     dep_size = in size for dependent words
        
        Returns : a score matrix S of shape :
        - if self.num_scores_per_arc > 1 (for label scores):
          [batch_size, num_lab, n, n ] (num_lab = self.num_scores_per_arc) 
        - if self.num_scores_per_arc == 1 (for arc scores):
          [batch_size, n, n] 

        S(batch_k, l, i, j) = score of sample k in batch, label l, head word i, dep word j
        """
        
        if self.use_bias:
            bs = Hh.shape[0]
            n = Hh.shape[1]
            temp = torch.ones(bs,n,1, device=self.device)
            Hh = torch.cat((Hh, temp), 2)
            Hd = torch.cat((Hd, temp), 2)
            
        # add dimension in input tensors for num_label dimension
        # (cf. otherwise broadcast will prepend 1 dim of size 1)
        Hh = Hh.unsqueeze(1)
        Hd = Hd.unsqueeze(1)

        # @ (== torch.matmul) is matrix product of last 2 dimensions
        #                     (here: the head and dependent dimensions)
        # other dimensions are broadcasted
        S = Hh @ self.U @ Hd.transpose(-1, -2)

        # the squeeze will operate iff num_labels == 1:
        #     [batch_size, num_labels, d, d] => [batch_size, d, d]
        return S.squeeze(1)
