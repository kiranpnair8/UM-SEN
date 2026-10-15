"""Non-trainable transformer alpha access, independent of SAGE."""

import math

from spikingjelly.clock_driven.neuron import MultiStepLIFNode


class BlockAlphas:
    def __init__(self, model):
        self.model = getattr(model, "module", model)
        if hasattr(self.model, "_sage_controller"):
            raise ValueError("Causal alpha settings must not compete with SAGE")
        self.nodes = [dict((name, node) for name, node in block.named_modules()
                           if isinstance(node, MultiStepLIFNode)) for block in self.model.block]
        if not self.nodes or any(not nodes for nodes in self.nodes):
            raise ValueError("Expected spiking transformer blocks")
        all_nodes = [node for node in self.model.modules() if isinstance(node, MultiStepLIFNode)]
        if len({id(node.surrogate_function) for node in all_nodes}) != len(all_nodes):
            raise ValueError("Shared surrogate instances prevent independent alpha settings")

    def get(self):
        values = []
        for nodes in self.nodes:
            alphas = {float(node.surrogate_function.alpha) for node in nodes.values()}
            if len(alphas) != 1:
                raise ValueError("Neurons within a block have inconsistent alphas")
            values.append(alphas.pop())
        return values

    def set(self, block_id, alpha):
        if not isinstance(block_id, int) or not 0 <= block_id < len(self.nodes):
            raise ValueError("Invalid zero-based block_id")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Alpha must be finite and positive")
        for node in self.nodes[block_id].values():
            if not isinstance(node.surrogate_function.alpha, (int, float)):
                raise ValueError("This stage supports non-trainable scalar alpha only")
            node.surrogate_function.alpha = float(alpha)

    def set_all(self, values):
        if len(values) != len(self.nodes):
            raise ValueError("Expected one alpha per transformer block")
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("All alphas must be finite and positive")
        for block_id, value in enumerate(values):
            self.set(block_id, value)

    def snapshot(self):
        return {name: float(node.surrogate_function.alpha) for name, node in self.model.named_modules()
                if isinstance(node, MultiStepLIFNode)}

    def restore(self, state):
        nodes = {name: node for name, node in self.model.named_modules() if isinstance(node, MultiStepLIFNode)}
        if nodes.keys() != state.keys():
            raise ValueError("Alpha snapshot does not match model")
        for name, node in nodes.items():
            node.surrogate_function.alpha = float(state[name])
