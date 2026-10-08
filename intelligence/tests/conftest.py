from pathlib import Path
import pytest
from src.ml.data_loader import load_data
from src.ml.synthetic_data import generate_synthetic
from src.ml.train import train_model


@pytest.fixture(scope="session")
def observed():
    return load_data()


@pytest.fixture(scope="session")
def synthetic(observed):
    # Two hospitals x two medicines, 90 days, enough for all splits.
    catalog = observed[observed.hospital_id.isin(["HOSP_A", "HOSP_B"]) &
                       observed.medicine_id.isin(["PGMED001", "PGMED002"])]
    return generate_synthetic(days=90, catalog=catalog)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, synthetic):
    directory = tmp_path_factory.mktemp("training")
    data_path = directory / "demand.csv"
    synthetic.to_csv(data_path, index=False)
    model_path = directory / "lightgbm_outbreak_v2.joblib"
    artifact = train_model(data_path, model_path, directory / "reports")
    return artifact, model_path, directory
