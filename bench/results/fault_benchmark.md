# Fault-injection benchmark

Window 400 requests; baseline threshold calibrated on reference data at alpha 0.01 (min-p threshold 6.69e-04).

* Detection: **322/330 = 97.6%** (95% CI 95.3% to 98.8%)
* False alarms on clean traffic: **0/300 = 0.0%** (95% CI 0.0% to 1.3%)

| Fault | What it simulates | Detected | Signals |
|---|---|---|---|
| exposure_shift | camera exposure or white balance regression darkens 10 to 40% | 30/30 | input_drift 30 |
| sensor_noise | codec or sensor degradation adds noise | 29/30 | input_drift 29 |
| resize_blur | resize bug downsamples by 2x or 4x and upsamples | 30/30 | input_drift 30 |
| channel_swap | client release sends BGR instead of RGB | 26/30 | input_drift 23, output_drift 3 |
| frozen_frames | stuck cameras send black, gray or white frames | 30/30 | input_drift 30 |
| locale_launch | new locale sends Spanish captions | 30/30 | input_drift 30 |
| caption_template | upstream template change wraps captions in boilerplate | 30/30 | input_drift 30 |
| client_mix | app release shifts 15 to 40% of traffic to android | 27/30 | input_drift 27 |
| schema_break | SDK sends 48x48 images, FP32 tensors or drops parameters | 30/30 | contract 30, output_drift 1 |
| output_corruption | post-processing drops normalization or emits NaN | 30/30 | contract 30, output_drift 1 |
| bad_replica | some replicas serve the BGR-regressed build | 30/30 | output_drift 30 |
