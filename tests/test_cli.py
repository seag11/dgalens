import json
from pathlib import Path

from typer.testing import CliRunner

from dgalens.cli import app

runner = CliRunner()


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_train_score_eval(tmp_path: Path, benign: list[str], random_labels: list[str]) -> None:
    benign_file = write_lines(tmp_path / "benign.txt", [f"{d}.com" for d in benign])
    dga_file = write_lines(tmp_path / "dga.txt", [f"{d}.net" for d in random_labels])
    model = tmp_path / "model.json"

    result = runner.invoke(app, ["train", str(benign_file), "--model", str(model), "--order", "2"])
    assert result.exit_code == 0, result.output
    assert model.exists()

    result = runner.invoke(app, ["score", "--model", str(model), "--json", "garden.com", "xq7zk2v9.biz"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.output.splitlines()]
    assert [r["label"] for r in rows] == ["garden", "xq7zk2v9"]
    assert rows[0]["score"] > rows[1]["score"]

    result = runner.invoke(app, ["eval", str(benign_file), str(dga_file), "--model", str(model)])
    assert result.exit_code == 0, result.output
    assert "ROC AUC" in result.output


def test_score_reads_stdin(tmp_path: Path, benign: list[str]) -> None:
    benign_file = write_lines(tmp_path / "benign.txt", benign)
    model = tmp_path / "model.json"
    runner.invoke(app, ["train", str(benign_file), "--model", str(model)])

    result = runner.invoke(
        app, ["score", "--model", str(model)], input="garden.com\n\n# note\n7,forest.org\n"
    )
    assert result.exit_code == 0, result.output
    assert [line.split()[-1] for line in result.output.splitlines()] == ["garden.com", "forest.org"]


def test_train_rejects_empty_input(tmp_path: Path) -> None:
    empty = write_lines(tmp_path / "empty.txt", ["# nothing"])
    result = runner.invoke(app, ["train", str(empty), "--model", str(tmp_path / "m.json")])
    assert result.exit_code != 0
