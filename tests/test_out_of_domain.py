import pandas as pd
from src.trust.out_of_domain_check import domain_report


def test_out_of_domain_warning():
    train = pd.DataFrame({"laser_power_W": [100, 200], "alloy": ["316L", "Ti64"]})
    query = pd.DataFrame({"laser_power_W": [500], "alloy": ["UnknownAlloy"]})
    warnings = domain_report(train, query)
    assert len(warnings) >= 2
