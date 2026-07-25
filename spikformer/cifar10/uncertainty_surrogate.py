from spikingjelly.clock_driven import surrogate


class UncertaintyModulatedSigmoid(surrogate.Sigmoid):
    def __init__(self, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma


def make_surrogate_function(sigma=1.0):
    return UncertaintyModulatedSigmoid(sigma=sigma)
