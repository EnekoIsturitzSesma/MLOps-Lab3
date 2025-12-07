import os
import pytest
from PIL import Image
from click.testing import CliRunner
from cli.cli import cli


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary RGB image for CLI tests."""
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100), color="white").save(img_path)
    return img_path


def test_cli_predict(temp_image):
    runner = CliRunner()
    result = runner.invoke(cli, ["predict", str(temp_image)])
    assert result.exit_code == 0
    assert result.output.strip() in ["dog", "cat", "horse", "bear", "pig"]


def test_cli_resize(temp_image):
    runner = CliRunner()
    result = runner.invoke(cli, ["resize", str(temp_image), "50", "60"])
    assert result.exit_code == 0
    assert result.output.strip() == "50x60"


def test_cli_grayscale(temp_image):
    runner = CliRunner()
    result = runner.invoke(cli, ["grayscale", str(temp_image)])
    assert result.exit_code == 0
    assert result.output.strip() == "L"


def test_cli_normalize(temp_image):
    runner = CliRunner()
    result = runner.invoke(cli, ["normalize", str(temp_image)])
    assert result.exit_code == 0
    assert "min=" in result.output and "max=" in result.output
