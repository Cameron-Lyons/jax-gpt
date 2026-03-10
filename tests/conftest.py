import pytest
from jax import random


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked as requiring network access",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return

    skip_network = pytest.mark.skip(reason="need --run-network to run network-dependent tests")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture
def rng():
    return random.PRNGKey(42)


@pytest.fixture
def small_config():
    from model import GPTConfig

    return GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=256,
        block_size=32,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        bias=True,
        dtype="float32",
        tie_word_embeddings=True,
    )


@pytest.fixture
def dummy_batch(rng, small_config):
    batch_size = 2
    seq_len = small_config.block_size
    x = random.randint(rng, (batch_size, seq_len), 0, small_config.vocab_size)
    return x
