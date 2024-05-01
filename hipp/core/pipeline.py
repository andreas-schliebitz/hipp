import os
import logging
import numpy as np

from typing import List, Tuple
from threadpoolctl import threadpool_limits

from hipp.core.module import Module
from hipp.utils.utility import is_normalized, normalize


class Pipeline:
    def __init__(self, n_jobs=-1) -> None:
        self.modules: List[Module] = []

        if "HIPP_JOBS" not in os.environ:
            os.environ["HIPP_JOBS"] = str(n_jobs)

        logging.info(f"Using {os.environ['HIPP_JOBS']} threads.")

    def add(self, module: Module) -> None:
        self.modules.append(module)
        logging.info(f"Added {module} to pipeline.")

    def run(
        self, data: np.ndarray, mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        numpy_thread_limit = (
            None if os.environ["HIPP_JOBS"] == "-1" else int(os.environ["HIPP_JOBS"])
        )
        if numpy_thread_limit is not None:
            with threadpool_limits(limits=numpy_thread_limit, user_api="blas"):
                return self._run(data, mask)
        return self._run(data, mask)

    def _run(
        self, data: np.ndarray, mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.modules:
            if not is_normalized(data):
                data = normalize(data)

            for module in self.modules:
                logging.info(f"Current module: {module}")

                module.load(data, mask)
                module.run()
                data = module.get_data()

            return data.filled(fill_value=0), ~data.mask[..., 0]
        raise RuntimeError("Pipeline is empty.")

    def __str__(self) -> str:
        modules: str = "\n\t".join([str(m) for m in self.modules])
        return f"Pipeline(modules={modules}, logger={self.logger})"
