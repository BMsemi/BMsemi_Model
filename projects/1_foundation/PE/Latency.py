class latencycalculator:
    def __init__(self):
      self.init_latency = 9 
      self.iter_latency = 6

    def compute_total_latency(self, iteration_count: int) -> int:
        """
        Calculate total latency based on initial latency and per-iteration latency.

        Args:
          iteration_count: Number of temporal iterations.

        Returns:
          Total latency as an integer.

        """
        total_latency = 0
        total_latency = self.init_latency + (self.iter_latency * iteration_count)
        return total_latency
