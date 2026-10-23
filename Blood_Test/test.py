from pathlib import Path

from Functions import (
    append_pbs_complex_difference,
    append_pbs_reference_difference,
    apply_savgol_filter_nested,
    apply_savgol_to_re_im_and_recompute,
    downsample_nested_data,
    load_blood_folder,
    print_blood_data_summary,
)


## Load the raw data in PyCharm/Python console mode
DATA_DIR = Path(r"/home/shibojing/data/blood")
raw_data = load_blood_folder(DATA_DIR)
print_blood_data_summary(raw_data)


## Filter real/imaginary signals and recompute amplitude/phase
filtered_data = apply_savgol_to_re_im_and_recompute(
    raw_data,
    window_length=31,
    polyorder=3,
)
print("Savitzky-Golay filtering completed: window_length=31, polyorder=3")


## Calculate set-specific PBS-referenced features
# For each set, average its five PBS repetitions and append the pointwise
# [real, imaginary, amplitude, phase] differences from that set's mean PBS trace.
difference_data = append_pbs_reference_difference(filtered_data)

# Append two complex-valued comparison features using the same set's mean PBS:
# the phase of (SAU / PBS) and the magnitude of (SAU - PBS).
difference_data = append_pbs_complex_difference(difference_data)

# Smooth all ten existing and PBS-derived channels along the frequency axis.
filtered_difference_data = apply_savgol_filter_nested(
    difference_data,
    window_length=31,
    polyorder=3,
)
print("PBS difference calculation and Savitzky-Golay filtering completed")


## Downsample the processed signals
# Keep every 100th frequency point, reducing 20,000 points to 200 while
# preserving every set, sample, repetition, and all 10 signal channels.
downsampled_difference_data = downsample_nested_data(
    filtered_difference_data,
    ratio=100,
)
print(
    "Downsampling completed: "
    f"SAU 10-4 shape={downsampled_difference_data['SAU']['10-4'].shape}"
)
