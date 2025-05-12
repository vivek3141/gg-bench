import logging
from logging import handlers
import multiprocessing
import sys


class MultiprocessLogger:
    def __init__(self):
        self.log_queue = multiprocessing.Queue()
        self.logger_process = multiprocessing.Process(
            target=self._logger_process, args=(self.log_queue,)
        )
        self.logger_process.start()

    def _logger_process(self, queue):
        """
        The process that prints logs to stdout. It continuously listens to the queue for log records.
        """
        # Set up logging to stdout
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        root_logger.addHandler(handler)

        while True:
            try:
                record = queue.get()
                if record is None:
                    break  # Sentinel value to terminate the logging process
                root_logger.handle(record)
            except Exception as e:
                print(f"Logger encountered an error: {e}", file=sys.stderr)

    def get_logger(self, name=None):
        """
        Returns a logger instance that puts log records into the shared queue.
        """
        logger = logging.getLogger(name or str(multiprocessing.current_process().name))
        logger.setLevel(logging.DEBUG)

        queue_handler = handlers.QueueHandler(self.log_queue)
        logger.addHandler(queue_handler)

        return logger

    def stop(self):
        """
        Sends a sentinel value to stop the logger process.
        """
        self.log_queue.put(None)
        self.logger_process.join()
