import pytest
from app import app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_homepage(client):
    """Test if the homepage loads correctly."""
    response = client.get("/")
    assert response.status_code == 200
