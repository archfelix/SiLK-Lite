import math


class LowpassFilter():
    last_x = None
    alpha = 0.01

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def update(self, x):
        if self.last_x is None:
            if math.isnan(x) or math.isinf(x):
                return 0
            else:
                self.last_x = x
        else:
            if math.isnan(x) or math.isinf(x):
                return self.last_x
            else:
                r = x * self.alpha + (1 - self.alpha) * self.last_x
                self.last_x = r
        return self.last_x
