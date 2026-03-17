from pathlib import Path

from modules.data_process.data_loader import load_simulated_reports


def test_load_simulated_reports_count_and_shape() -> None:
    data_file = Path(__file__).resolve().parents[1] / "data" / "simulate_data.json"
    reports = load_simulated_reports(data_file)

    assert len(reports) >= 10
    sample = reports[0]
    assert set(sample.keys()) == {"report_id", "report_type", "report_subtype", "content", "label"}
    assert set(sample["content"].keys()) == {"描述", "检查所见", "检查提示"}
