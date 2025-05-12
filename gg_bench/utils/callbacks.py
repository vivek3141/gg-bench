from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class ProgressBarCallback(BaseCallback):
    """
    A callback that updates a progress bar during training.
    """

    def __init__(self, pbar: tqdm):
        """
        Initialize the ProgressBarCallback.

        Args:
            pbar (tqdm): The progress bar to update.
        """
        super(ProgressBarCallback, self).__init__()
        self.pbar: tqdm = pbar

    def _on_step(self) -> bool:
        """
        Update the progress bar by one step.

        Returns:
            bool: Always returns True to continue training.
        """
        self.pbar.update(1)
        return True
