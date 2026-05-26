class SimulatorApiClient:
    """Client for communicating with the simulator API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
