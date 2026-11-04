"""Run from the project root: python -m Blood_Test.plot_model_architecture."""

from pathlib import Path

from Blood_Test.Functions.model_diagram import plot_model_architecture


if __name__ == "__main__":
    _, output_paths = plot_model_architecture(
        output_dir=Path(__file__).resolve().parent / "figures",
        in_channels=1,
        frequency_points=200,
        channel_label="Channel 10",
    )
    for format_name, output_path in output_paths.items():
        print(f"{format_name.upper()}: {output_path}")
