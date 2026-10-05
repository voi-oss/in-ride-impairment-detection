from typing import List, Optional

import numpy as np
import pandas as pd
import scrapbook as sb
from scipy.signal import firwin, lfilter


def reshape_to_numpy(
    data: pd.DataFrame, features: List[str], max_timestamps: Optional[int] = None
) -> np.ndarray:
    """Reshape grouped time series data into a 3D numpy array.

    Returns array of shape (n_instances, n_timestamps, n_features).
    """
    grouped_data = (
        data.groupby("ride_id")[features].apply(lambda x: x.values).reset_index()
    )

    time_series = grouped_data.iloc[:, 1].values

    if max_timestamps is None:
        max_timestamps = max(len(ts) for ts in time_series)

    n_features = len(features)

    reshaped_data = np.full((len(time_series), max_timestamps, n_features), np.nan)

    for i, ts in enumerate(time_series):
        reshaped_data[i, : len(ts), :] = ts

    return reshaped_data


def interpolate_missing_values(
    data: np.ndarray,
    method: str,
    limit: Optional[int] = None,
) -> np.ndarray:
    """Interpolate missing values in time series data.

    Only interpolates up to the last valid value to avoid trailing NaNs.
    """
    interpolated_data = data.copy()

    for ride_idx in range(data.shape[0]):
        for feat_idx in range(data.shape[2]):
            series = pd.Series(interpolated_data[ride_idx, :, feat_idx])

            last_valid_idx = series.last_valid_index()
            if last_valid_idx is not None:
                series_to_interpolate = series.iloc[: last_valid_idx + 1]
                interpolated_series = series_to_interpolate.interpolate(
                    method=method, limit=limit
                )
                interpolated_data[
                    ride_idx, : last_valid_idx + 1, feat_idx
                ] = interpolated_series.values

    return interpolated_data


def bootstrap_report(
    df_res: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
    glue: bool = False,
    per_ride_path: Optional[str] = None,
    per_ride_cols: tuple = (
        "ride_id",
        "participant_id",
        "run",
        "detected",
        "detection_time_s",
    ),
) -> pd.DataFrame:
    has_time = "detection_time_s" in df_res.columns
    rng = np.random.default_rng(seed)

    # optionally save the per-ride outcomes (scalar columns only) for paired
    # cross-method analysis; skipped when no path is given
    if per_ride_path is not None:
        keep = [c for c in per_ride_cols if c in df_res.columns]
        df_res[keep].to_csv(per_ride_path, index=False)

    # detection rate and support per run
    summary = (
        df_res.groupby("run")
        .agg(support=("detected", "size"), detection_rate=("detected", "mean"))
        .reindex(["High", "Low", "Sober"])
    )

    # set FAR to detection_rate for Sober
    summary["false_alarm_rate"] = np.where(
        summary.index == "Sober", summary["detection_rate"], np.nan
    )

    # set NaN as detection rate for Sober
    summary.loc["Sober", "detection_rate"] = np.nan

    # clustered bootstrap CIs over participants
    pids = df_res["participant_id"].values  # participant IDs
    unique_pids = np.unique(pids)  # unique participants
    (
        boot_rate,
        boot_tmin,
        boot_t25,
        boot_t50,
        boot_t75,
        boot_tmax,
        boot_acc_ns,
        boot_acc_h,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )  # per-resample stats

    for _ in range(n_boot):  # n_boot resamples
        # resample participants (with replacement)
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        rows = np.concatenate([np.flatnonzero(pids == p) for p in sampled])
        sub = df_res.iloc[rows]  # take all participants rides
        # detection rate for this sample
        boot_rate.append(
            sub.groupby("run")["detected"].mean().reindex(["High", "Low", "Sober"])
        )
        if has_time:
            # q25, median, q75 detection time for this sample (TPs only)
            sub_tp = sub[sub["detected"] & (sub["run"] != "Sober")]
            gt = sub_tp.groupby("run")["detection_time_s"]
            boot_tmin.append(gt.min().reindex(["High", "Low"]))
            boot_t25.append(gt.quantile(0.25).reindex(["High", "Low"]))
            boot_t50.append(gt.quantile(0.50).reindex(["High", "Low"]))
            boot_t75.append(gt.quantile(0.75).reindex(["High", "Low"]))
            boot_tmax.append(gt.max().reindex(["High", "Low"]))
        # accuracy for this sample
        sub_correct = np.where(
            sub["run"] == "Sober", ~sub["detected"], sub["detected"]
        )  # per-ride correct
        boot_acc_ns.append(sub_correct.mean())  # sober vs non-sober
        boot_acc_h.append(sub_correct[sub["run"] != "Low"].mean())  # sober vs high

    # compute 95% CI for detection rate
    boot_rate = pd.DataFrame(boot_rate)
    summary["dr_ci_lo"] = boot_rate.quantile(0.025)
    summary["dr_ci_hi"] = boot_rate.quantile(0.975)

    if has_time:
        # compute 95% CI for q25, median, q75 detection time (TPs only)
        boot_tmin, boot_t25, boot_t50, boot_t75, boot_tmax = (
            pd.DataFrame(boot_tmin),
            pd.DataFrame(boot_t25),
            pd.DataFrame(boot_t50),
            pd.DataFrame(boot_t75),
            pd.DataFrame(boot_tmax),
        )
        tp_grp = df_res[df_res["detected"] & (df_res["run"] != "Sober")].groupby("run")[
            "detection_time_s"
        ]  # true positives
        summary["min_det_time"] = tp_grp.min().reindex(["High", "Low", "Sober"])
        summary["tmin_ci_lo"] = boot_tmin.quantile(0.025).reindex(
            ["High", "Low", "Sober"]
        )
        summary["tmin_ci_hi"] = boot_tmin.quantile(0.975).reindex(
            ["High", "Low", "Sober"]
        )
        summary["q25_det_time"] = tp_grp.quantile(0.25).reindex(
            ["High", "Low", "Sober"]
        )
        summary["median_det_time"] = tp_grp.median().reindex(["High", "Low", "Sober"])
        summary["q75_det_time"] = tp_grp.quantile(0.75).reindex(
            ["High", "Low", "Sober"]
        )
        summary["t25_ci_lo"] = boot_t25.quantile(0.025).reindex(
            ["High", "Low", "Sober"]
        )
        summary["t25_ci_hi"] = boot_t25.quantile(0.975).reindex(
            ["High", "Low", "Sober"]
        )
        summary["t_ci_lo"] = boot_t50.quantile(0.025).reindex(["High", "Low", "Sober"])
        summary["t_ci_hi"] = boot_t50.quantile(0.975).reindex(["High", "Low", "Sober"])
        summary["t75_ci_lo"] = boot_t75.quantile(0.025).reindex(
            ["High", "Low", "Sober"]
        )
        summary["t75_ci_hi"] = boot_t75.quantile(0.975).reindex(
            ["High", "Low", "Sober"]
        )
        summary["max_det_time"] = tp_grp.max().reindex(["High", "Low", "Sober"])
        summary["tmax_ci_lo"] = boot_tmax.quantile(0.025).reindex(
            ["High", "Low", "Sober"]
        )
        summary["tmax_ci_hi"] = boot_tmax.quantile(0.975).reindex(
            ["High", "Low", "Sober"]
        )

    # accuracy: Sober correct if not detected, High/Low correct if detected
    correct = np.where(
        df_res["run"] == "Sober", ~df_res["detected"], df_res["detected"]
    )  # get correct predictions
    acc_non_sober = correct.mean()  # accuracy sober vs non-sober
    acc_high = correct[df_res["run"] != "Low"].mean()  # accuracy sober vs high
    ci_ns = np.quantile(boot_acc_ns, [0.025, 0.975])  # CI sober vs non-sober (95%)
    ci_h = np.quantile(boot_acc_h, [0.025, 0.975])  # CI sober vs high (95%)

    # print summary as table (rate columns: 2 decimals; time columns: integer seconds)
    def _fmt2(p, lo, hi):
        return f"{p:.2f} [{lo:.2f}, {hi:.2f}]" if pd.notna(p) else "NaN"

    def _fmt0(p, lo, hi):
        return f"{p:.0f} [{lo:.0f}, {hi:.0f}]" if pd.notna(p) else "NaN"

    tbl = pd.DataFrame(index=summary.index)
    tbl["support"] = summary["support"]
    tbl["detection_rate"] = [
        _fmt2(
            summary.loc[c, "detection_rate"],
            summary.loc[c, "dr_ci_lo"],
            summary.loc[c, "dr_ci_hi"],
        )
        for c in tbl.index
    ]
    if has_time:
        tbl["det_time_min"] = [
            _fmt0(
                summary.loc[c, "min_det_time"],
                summary.loc[c, "tmin_ci_lo"],
                summary.loc[c, "tmin_ci_hi"],
            )
            for c in tbl.index
        ]
        tbl["det_time_q25"] = [
            _fmt0(
                summary.loc[c, "q25_det_time"],
                summary.loc[c, "t25_ci_lo"],
                summary.loc[c, "t25_ci_hi"],
            )
            for c in tbl.index
        ]
        tbl["det_time_med"] = [
            _fmt0(
                summary.loc[c, "median_det_time"],
                summary.loc[c, "t_ci_lo"],
                summary.loc[c, "t_ci_hi"],
            )
            for c in tbl.index
        ]
        tbl["det_time_q75"] = [
            _fmt0(
                summary.loc[c, "q75_det_time"],
                summary.loc[c, "t75_ci_lo"],
                summary.loc[c, "t75_ci_hi"],
            )
            for c in tbl.index
        ]
        tbl["det_time_max"] = [
            _fmt0(
                summary.loc[c, "max_det_time"],
                summary.loc[c, "tmax_ci_lo"],
                summary.loc[c, "tmax_ci_hi"],
            )
            for c in tbl.index
        ]
    tbl["FAR"] = [
        _fmt2(
            summary.loc[c, "false_alarm_rate"],
            summary.loc[c, "dr_ci_lo"],
            summary.loc[c, "dr_ci_hi"],
        )
        for c in tbl.index
    ]
    print(tbl.to_string())

    # accuracy on separate lines
    print(
        f"\naccuracy sober vs non-sober: {acc_non_sober:.2f} [{ci_ns[0]:.2f}, {ci_ns[1]:.2f}]"
    )
    print(f"accuracy sober vs high:      {acc_high:.2f} [{ci_h[0]:.2f}, {ci_h[1]:.2f}]")

    # glue metrics (point estimates + CI bounds)
    if glue:
        # per-run summary columns
        for run in summary.index:
            for col in summary.columns:
                sb.glue(f"{run}_{col}", float(summary.loc[run, col]))

        # accuracy point estimates and 95% CI bounds
        sb.glue("accuracy_non_sober", float(acc_non_sober))
        sb.glue("accuracy_non_sober_ci_lo", float(ci_ns[0]))
        sb.glue("accuracy_non_sober_ci_hi", float(ci_ns[1]))
        sb.glue("accuracy_high", float(acc_high))
        sb.glue("accuracy_high_ci_lo", float(ci_h[0]))
        sb.glue("accuracy_high_ci_hi", float(ci_h[1]))

    return summary


def lowpass_filter_causal(
    data: np.ndarray,
    accel_indices: List[int],
    gyro_indices: List[int],
    accel_cutoff: float,
    gyro_cutoff: float,
    order: int,
    fs: float,
) -> np.ndarray:
    """Apply causal FIR lowpass filters and shift non-filtered dims to match group delay.

    The causal FIR filter introduces a group delay of order // 2 samples.
    Non-filtered dimensions are delayed by the same amount to stay aligned.
    The first `delay` samples are blanked (NaN) across all channels to
    discard the filter transient.
    """
    delay = order // 2
    other_indices = [
        i
        for i in range(data.shape[2])
        if i not in set(accel_indices) | set(gyro_indices)
    ]

    out = np.full_like(data, np.nan)

    for indices, cutoff in [(accel_indices, accel_cutoff), (gyro_indices, gyro_cutoff)]:
        if indices:
            b = firwin(order + 1, cutoff, fs=fs, window="hamming")
            out[:, delay:, indices] = lfilter(b, 1.0, data[:, :, indices], axis=1)[
                :, delay:
            ]

    if other_indices:
        out[:, delay:, other_indices] = (
            data[:, :-delay, other_indices] if delay else data[:, :, other_indices]
        )

    return out
