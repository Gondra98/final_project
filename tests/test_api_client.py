from src.simulator.api_client import SimulatorApiClient


def test_api_client_trims_trailing_slash():
    client = SimulatorApiClient("http://localhost:8000/")

    assert client.base_url == "http://localhost:8000"
