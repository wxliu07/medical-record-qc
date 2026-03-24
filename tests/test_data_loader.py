from pathlib import Path

from modules.data_process.data_loader import load_simulated_reports


def test_load_simulated_reports_count_and_shape() -> None:
    data_file = Path(__file__).resolve().parents[1] / "data" / "simulate_data.json"
    reports = load_simulated_reports(data_file)

    assert len(reports) >= 10
    sample = reports[0]
    assert set(sample.keys()) == {"report_id", "report_type", "report_subtype", "content", "label"}
    assert set(sample["content"].keys()) == {"描述", "检查所见", "检查提示"}

    report_types = {r["report_type"] for r in reports}
    assert "指标类" in report_types
    assert "影像类" in report_types

    # 模拟数据应尽量贴近体检/复查实际：每类至少有多条样本
    indicator_count = sum(1 for r in reports if r["report_type"] == "指标类")
    imaging_count = sum(1 for r in reports if r["report_type"] == "影像类")
    assert indicator_count >= 4
    assert imaging_count >= 4
