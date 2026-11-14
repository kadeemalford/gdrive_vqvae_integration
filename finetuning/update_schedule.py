"""Explicit per-update learning rates with strict, budget-preserving resume."""
import copy
import math


def load_independent_optimizer_state(optimizer, state):
    """Do not share CPU step tensors retained by torch.optim.load_state_dict."""
    optimizer.load_state_dict(copy.deepcopy(state))


class UpdateSchedule:
    def __init__(self, kind, total_updates, high=1e-4, low=1e-5):
        if kind not in ('constant_high', 'cosine', 'constant_mean'):
            raise ValueError('Unknown schedule')
        if not isinstance(total_updates, int) or total_updates < 2:
            raise ValueError('At least two updates required')
        if not all(math.isfinite(v) and v > 0 for v in (high, low)) or low > high:
            raise ValueError('Invalid learning rate bounds')
        self.kind, self.total_updates, self.high, self.low = kind, total_updates, high, low
        self.completed_updates = 0

    def rate(self, update):
        if not isinstance(update, int) or not 0 <= update < self.total_updates:
            raise ValueError('Update outside fixed schedule budget')
        if self.kind == 'constant_high':
            return self.high
        if self.kind == 'constant_mean':
            return (self.high + self.low) / 2
        return self.low + .5 * (self.high - self.low) * (1 + math.cos(math.pi * update / (self.total_updates - 1)))

    def prepare_update(self, optimizer):
        rate = self.rate(self.completed_updates)
        for group in optimizer.param_groups:
            group['lr'] = rate
        return rate

    def advance(self):
        if self.completed_updates >= self.total_updates:
            raise ValueError('Schedule budget already exhausted')
        self.completed_updates += 1

    def state_dict(self):
        return {key: getattr(self, key) for key in ('kind', 'total_updates', 'high', 'low', 'completed_updates')}

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()):
            raise ValueError('Schedule state fields changed')
        if any(state[k] != getattr(self, k) for k in ('kind', 'total_updates', 'high', 'low')):
            raise ValueError('Schedule resume policy changed')
        n = state['completed_updates']
        if not isinstance(n, int) or not 0 <= n <= self.total_updates:
            raise ValueError('Invalid completed update count')
        self.completed_updates = n
