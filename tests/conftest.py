import sys
from pathlib import Path

# Allow `import stratval` from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from stratval import generate_rebalance_panel  # noqa: E402


@pytest.fixture(scope="session")
def panel():
    """A synthetic panel with a real within-event signal + a planted regime confound."""
    return generate_rebalance_panel(seed=7)


@pytest.fixture(scope="session")
def null_panel():
    """A panel with no within-event signal (beta_harvest=0) — gates must fail on it."""
    return generate_rebalance_panel(beta_harvest=0.0, regime_intercept=1.3, seed=7)
