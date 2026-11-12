from __future__ import annotations
class AbstractDatasetPrior:
    def new_dataset(self) -> None:
        raise NotImplementedError
