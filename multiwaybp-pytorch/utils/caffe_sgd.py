from torch.optim.optimizer import Optimizer, required

__all__ = ["caffeSGD"]


class caffeSGD(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum). 

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        caffe version:

        math::
             v = \rho * v - lr * g \\
             p = p + v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and momentum <= 0:
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(caffeSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(caffeSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                        buf_clone = buf.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf_clone = buf.clone()
                        buf.mul_(momentum).add_(
                            group['lr'], d_p)
                    if nesterov:
                        d_p = buf * (1 + momentum) - buf_clone * momentum
                        # d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-1, d_p)

        return loss
