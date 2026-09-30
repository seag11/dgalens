from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from dgalens.api import MAX_BATCH, create_app
from dgalens.model import MarkovModel


@pytest.fixture
def client(trained: MarkovModel) -> Iterator[TestClient]:
    with TestClient(create_app(trained)) as c:
        yield c


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_score(client: TestClient) -> None:
    body = client.post("/score", json={"domain": "www.garden.com"}).json()
    assert body["label"] == "garden"
    assert body["suspicious"] is False


def test_score_flags_random_label(client: TestClient) -> None:
    body = client.post("/score", json={"domain": "xq7zk2v9wpt.biz"}).json()
    assert body["suspicious"] is True


def test_batch(client: TestClient) -> None:
    body = client.post("/score/batch", json={"domains": ["garden.com", "forest.org"]}).json()
    assert [r["label"] for r in body["results"]] == ["garden", "forest"]


@pytest.mark.parametrize(
    "payload",
    [{"domain": ""}, {"domain": "com"}, {}],
)
def test_score_rejects_bad_input(client: TestClient, payload: dict[str, str]) -> None:
    assert client.post("/score", json=payload).status_code == 422


def test_batch_size_limit(client: TestClient) -> None:
    response = client.post("/score/batch", json={"domains": ["a.com"] * (MAX_BATCH + 1)})
    assert response.status_code == 422
