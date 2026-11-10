"""Base interfaces for activation hooks and representation erasers."""
from abc import ABC, abstractmethod


class BaseHook:
    def __init__(self):
        self._handle = None

    def register(self, block, forward_pos="forward"):
        self.remove()
        if forward_pos == "forward":
            self._handle = block.register_forward_hook(self._hook_fn)
        elif forward_pos == "forward_pre":
            self._handle = block.register_forward_pre_hook(self._hook_fn_pre)
        else:
            raise ValueError(f"Unknown hook position: {forward_pos}")
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __call__(self, module, inputs, output):
        return self._hook_fn(module, inputs, output)

    def _hook_fn_pre(self, module, inputs):
        raise NotImplementedError("This hook operates on module outputs")


class BaseEraser(ABC):
    @abstractmethod
    def erase(self, values, alpha=None):
        raise NotImplementedError
