from Functions.Data_Loading_Old import get_subject_info, load_endpoint_dataset


METADATA_COLUMNS = {"label", "source_file", "night", "sensor_number", "subject_number", "sample_time"}


##
dataset = load_endpoint_dataset()
print(dataset.shape)
print(dataset["label"].value_counts())

subject_metadata, repeated_subject_metadata = get_subject_info(dataset)
print("Repeated subject numbers:")
print(repeated_subject_metadata["subject_number"].unique())



##
def plot_subject_night_samples(
    dataset,
    subject_number: int,
    night: int,
    sample_times: list[int] | None = None,
    max_samples: int = 10,
    ax=None,
    title: str | None = None,
    show_legend: bool = True,
    y_lim: tuple[float, float] | None = (1000, 3500),
):
    import matplotlib.pyplot as plt
    import numpy as np

    subject_night_data = dataset[
        (dataset["subject_number"] == subject_number) & (dataset["night"] == night)
    ].sort_values(["source_file", "sample_time"])

    if subject_night_data.empty:
        raise ValueError(f"No samples found for subject {subject_number} at night {night}")

    if sample_times is not None:
        subject_night_data = subject_night_data[
            subject_night_data["sample_time"].isin(sample_times)
        ]
    else:
        subject_night_data = subject_night_data.head(max_samples)

    if subject_night_data.empty:
        raise ValueError(f"No requested sample times found for subject {subject_number} at night {night}")

    frequency_columns = [column for column in dataset.columns if column not in METADATA_COLUMNS]
    frequencies = np.array(frequency_columns, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    for _, sample in subject_night_data.iterrows():
        amplitudes = sample[frequency_columns].to_numpy(dtype=float)
        ax.plot(
            frequencies,
            amplitudes,
            linewidth=1.0,
            alpha=0.35,
            label=f"time_point {int(sample['sample_time'])}",
        )

    ax.set_title(title or f"Subject {subject_number}, Night {night}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if show_legend:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_subject_range_summary(dataset):
    import matplotlib.pyplot as plt

    plot_config = [
        {
            "subject_number": 20,
            "night": 1,
            "ranges": [(0, 18), (18, 76), (0, 76)],
        },
        {
            "subject_number": 270,
            "night": 8,
            "ranges": [(0, 28), (28, 76), (0, 76)],
        },
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 8),
        sharex=True,
        sharey=True,
    )

    for row_idx, config in enumerate(plot_config):
        subject_night_data = dataset[
            (dataset["subject_number"] == config["subject_number"])
            & (dataset["night"] == config["night"])
        ]
        label = subject_night_data["label"].iloc[0]
        for col_idx, sample_range in enumerate(config["ranges"]):
            start, stop = sample_range
            ax = axes[row_idx, col_idx]
            plot_subject_night_samples(
                dataset,
                subject_number=config["subject_number"],
                night=config["night"],
                sample_times=list(range(start, stop)),
                ax=ax,
                title=(
                    f"Subject {config['subject_number']}, Night {config['night']}\n"
                    f"{label}, sample_time {start}:{stop}"
                ),
                show_legend=False,
            )

    fig.suptitle("Sample-Time Ranges by Subject and Night", fontsize=14)
    fig.tight_layout()
    fig.show()
    return fig, axes



##
fig, axes = plot_subject_range_summary(dataset)
