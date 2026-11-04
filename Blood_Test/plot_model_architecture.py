"""Run from the project root: python -m Blood_Test.plot_model_architecture."""

from Blood_Test.Functions.model_diagram import plot_model_architecture


if __name__ == "__main__":
    plot_model_architecture(
        in_channels=1,
        frequency_points=200,
        channel_label="Channel 10",
        show=True,
    )
