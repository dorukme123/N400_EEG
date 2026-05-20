"""N400 Group Analysis — Grand Averages and Planned Comparisons.

Loads individual *-ave.fif files from processed patients, computes grand
averages per group (healthy vs speech disorder), and runs within-block
and between-block comparisons from the study protocol.

Generates an HTML report with group ERP plots and a summary CSV.

Usage:
    python group_analysis_n400.py --input <dir_with_fif_files> --output <output_dir>
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import io
import logging
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == 'win32':
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                      errors='replace')

import matplotlib.pyplot as plt
import mne
import numpy as np

from process_n400 import (compute_uniform_ylim, find_best_n400_channels,
                          plot_best_n400_summary, plot_per_electrode_erps,
                          plot_peak_topomaps, plot_roi_channel_overlay)

# ── Constants ────────────────────────────────────────────────────────────────

N400_TMIN = 0.2
N400_TMAX = 0.6

# Explicit x-axis ticks so 400 ms is always visible
ERP_XTICKS = [-200, 0, 200, 400, 600, 800]

# Time points for topographic maps (seconds)
TOPOMAP_TIMES = [0.2, 0.3, 0.4, 0.5, 0.6]

N400_ROI_CHS = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4',
                'CP1', 'CP2']

REPORT_ELECTRODES = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4',
                     'CP1', 'CP2']

# Group assignments (from reviewer feedback, Комментарии N400.docx, 52 healthy)
HEALTHY = {
    'INP0008', 'INP0019', 'INP0036', 'INP0037', 'INP0055', 'INP0064',
    'INP0086', 'INP0089', 'INP0092', 'INP0094', 'INP0096', 'INP0101',
    'INP0102', 'INP0103', 'INP0104', 'INP0106', 'INP0107',
    'INP0110', 'INP0117', 'INP0125', 'INP0126', 'INP0129', 'INP0131',
    'INP0136', 'INP0138', 'INP0140', 'INP0144', 'INP0145', 'INP0146',
    'INP0149', 'INP0150', 'INP0151', 'INP0152', 'INP0154',
    'INP0155', 'INP0156', 'INP0161', 'INP0163', 'INP0164', 'INP0165',
    'INP0172', 'INP0173', 'INP0174', 'INP0175', 'INP0180', 'INP0185',
    'INP0188', 'INP0189', 'INP0190', 'INP0196', 'INP0198', 'INP0200',
}
# 17 effective (reviewer excluded INP0116, INP0123)
SPEECH_DISORDER = {
    'INP0014', 'INP0057', 'INP0076', 'INP0093', 'INP0100', 'INP0109',
    'INP0112', 'INP0113', 'INP0118', 'INP0127',
    'INP0128', 'INP0148', 'INP0160', 'INP0166', 'INP0168', 'INP0177',
    'INP0186',
}
# Excluded per reviewer: INP0116 (no epochs), INP0123 (broken report)
EXCLUDED = {'INP0116', 'INP0123'}

# All conditions we want in the report
ALL_CONDITIONS = (
    'BTR', 'BTP', 'BBTR', 'BBTP', 'BBBTR', 'BBBTP',
    'BLR', 'BLRR', 'BLP', 'BLPP', 'BLPPP', 'BLPPPP',
    'BBLR', 'BBLRR', 'BBLP', 'BBLPP',
    'BBBLR', 'BBBLP',
)

# Planned comparisons: list of (name, cond_A, cond_B)
# cond_A / cond_B are either a single string or a tuple to average first
WITHIN_BLOCK = [
    ('Block1: BLPPPP vs BLRR', 'BLPPPP', 'BLRR'),
    ('Block1: BLPPPP vs BLP', 'BLPPPP', 'BLP'),
    ('Block1: BTR vs BLRR', 'BTR', 'BLRR'),
    ('Block1: BTR vs BTP', 'BTR', 'BTP'),
    ('Block2: BBLPP vs BBLRR', 'BBLPP', 'BBLRR'),
    ('Block2: BBLPP vs BBLP', 'BBLPP', 'BBLP'),
    ('Block2: BBTP vs BBTR', 'BBTP', 'BBTR'),
    ('Block3: BBBTP vs BBBLP', 'BBBTP', 'BBBLP'),
    ('Block3: BBBTR vs BBBLR', 'BBBTR', 'BBBLR'),
    ('Block3: BBBTP vs BBBTR', 'BBBTP', 'BBBTR'),
]

BETWEEN_BLOCK = [
    ('(BTR+BBTR) vs BBBTR', ('BTR', 'BBTR'), 'BBBTR'),
    ('(BTR+BBTR) vs BTP', ('BTR', 'BBTR'), 'BTP'),
    ('(BTR+BBTR) vs BBTP', ('BTR', 'BBTR'), 'BBTP'),
    ('BBBTR vs BBBTP', 'BBBTR', 'BBBTP'),
]


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'group_analysis_{timestamp}.log'

    logger = logging.getLogger('group_n400')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%H:%M:%S')

    fh = logging.FileHandler(str(log_file), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f'Log file: {log_file}')
    return logger


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_evokeds(input_dir, logger):
    """Load all *-ave.fif files and group by patient/group.

    Returns dict: {patient_id: {condition: Evoked, ...}}
    and group assignments: {patient_id: 'healthy'|'disorder'|'unknown'}
    """
    fif_files = sorted(Path(input_dir).glob('*-ave.fif'))
    if not fif_files:
        raise FileNotFoundError(f'No *-ave.fif files in {input_dir}')

    logger.info(f'Found {len(fif_files)} .fif files')

    patients = {}
    groups = {}

    for fif_path in fif_files:
        # Extract patient ID from filename
        name = fif_path.stem  # e.g. INP0089_посещение_5-ave
        pid = name.split('_')[0]

        # Skip patients not in analysis groups
        if pid in EXCLUDED:
            logger.info(f'  {pid}: SKIPPED (excluded)')
            continue
        if pid not in HEALTHY and pid not in SPEECH_DISORDER:
            logger.info(f'  {pid}: SKIPPED (not in healthy or disorder list)')
            continue

        try:
            evokeds = mne.read_evokeds(str(fif_path), verbose='WARNING')
        except Exception as e:
            logger.error(f'Failed to load {fif_path}: {e}')
            continue

        patient_data = {}
        for evk in evokeds:
            if evk.comment:
                patient_data[evk.comment] = evk

        patients[pid] = patient_data
        groups[pid] = 'healthy' if pid in HEALTHY else 'disorder'
        logger.info(f'  {pid}: {len(patient_data)} conditions, '
                    f'group={groups[pid]}')

    return patients, groups


# ── Grand Averaging ──────────────────────────────────────────────────────────

def compute_grand_averages(patients, groups, conditions, logger):
    """Compute grand averages per group for each condition.

    Returns dict: {group: {condition: Evoked}}
    """
    group_names = sorted(set(groups.values()))
    result = {g: {} for g in group_names}

    for cond in conditions:
        for group in group_names:
            group_pids = [pid for pid, g in groups.items() if g == group]
            evokeds_for_avg = []
            for pid in group_pids:
                if cond in patients[pid]:
                    evokeds_for_avg.append(patients[pid][cond])

            if not evokeds_for_avg:
                continue

            grand = mne.grand_average(evokeds_for_avg, interpolate_bads=True,
                                       drop_bads=False)
            grand.comment = f'{cond}_{group}'
            result[group][cond] = grand
            logger.debug(f'  {cond} [{group}]: {len(evokeds_for_avg)} patients')

    return result


# ── N400 Measurement ─────────────────────────────────────────────────────────

def measure_n400(evoked, electrodes):
    """Measure N400 metrics in the 200-600ms window for given electrodes.

    Returns dict: {electrode: {mean_amp, peak_amp, peak_lat, area_lat_50}}
    """
    available = [ch for ch in electrodes if ch in evoked.ch_names]
    t_mask = (evoked.times >= N400_TMIN) & (evoked.times <= N400_TMAX)
    times_ms = evoked.times[t_mask] * 1000

    if not t_mask.any():
        return {}

    results = {}
    for ch in available:
        ch_idx = evoked.ch_names.index(ch)
        data_uv = evoked.data[ch_idx, t_mask] * 1e6

        mean_amp = data_uv.mean()
        peak_amp = data_uv.min()
        peak_lat = times_ms[np.argmin(data_uv)]

        # 50% area latency
        cumulative = np.cumsum(np.abs(data_uv))
        half_area = cumulative[-1] / 2.0
        area_idx = np.searchsorted(cumulative, half_area)
        area_lat = times_ms[min(area_idx, len(times_ms) - 1)]

        results[ch] = {
            'mean_amp': mean_amp,
            'peak_amp': peak_amp,
            'peak_lat': peak_lat,
            'area_lat_50': area_lat,
        }

    # Cluster average
    if available:
        all_data = np.array([evoked.data[evoked.ch_names.index(ch), t_mask]
                             for ch in available]) * 1e6
        cluster_mean = all_data.mean(axis=0)
        results['cluster'] = {
            'mean_amp': cluster_mean.mean(),
            'peak_amp': cluster_mean.min(),
            'peak_lat': times_ms[np.argmin(cluster_mean)],
            'area_lat_50': times_ms[min(
                np.searchsorted(np.cumsum(np.abs(cluster_mean)),
                                np.sum(np.abs(cluster_mean)) / 2),
                len(times_ms) - 1)],
        }

    return results


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_group_comparison(grand_avgs, cond, roi_chs, title=None, ylim=None):
    """Plot group overlay for a single condition — ROI average."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {'healthy': 'steelblue', 'disorder': 'indianred', 'unknown': 'grey'}
    labels = {'healthy': 'Healthy', 'disorder': 'Speech disorder',
              'unknown': 'Other'}

    for group in ('healthy', 'disorder'):
        if group not in grand_avgs or cond not in grand_avgs[group]:
            continue
        evk = grand_avgs[group][cond]
        avail = [ch for ch in roi_chs if ch in evk.ch_names]
        if not avail:
            continue
        idx = [evk.ch_names.index(ch) for ch in avail]
        roi_mean = evk.data[idx].mean(axis=0) * 1e6
        times = evk.times * 1000
        ax.plot(times, roi_mean, color=colors[group], linewidth=1.8,
                label=f'{labels[group]} (n={evk.nave})')

    ax.axvspan(N400_TMIN * 1000, N400_TMAX * 1000,
               alpha=0.12, color='blue', label='N400 window')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axhline(0, color='grey', linewidth=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title(title or f'{cond} — Group comparison (ROI average)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xticks(ERP_XTICKS)
    fig.tight_layout()
    return fig


def plot_comparison_pair(evk_a, evk_b, label_a, label_b, roi_chs, title,
                         ylim=None):
    """Plot two conditions overlaid + their difference, ROI-averaged."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [1.2, 1]})

    def get_roi_mean(evk):
        avail = [ch for ch in roi_chs if ch in evk.ch_names]
        if not avail:
            return evk.times * 1000, evk.data.mean(axis=0) * 1e6
        idx = [evk.ch_names.index(ch) for ch in avail]
        return evk.times * 1000, evk.data[idx].mean(axis=0) * 1e6

    times_a, mean_a = get_roi_mean(evk_a)
    times_b, mean_b = get_roi_mean(evk_b)

    # Top: overlay
    ax1.plot(times_a, mean_a, color='steelblue', linewidth=1.5, label=label_a)
    ax1.plot(times_b, mean_b, color='indianred', linewidth=1.5, label=label_b)
    ax1.axvspan(N400_TMIN * 1000, N400_TMAX * 1000, alpha=0.12, color='blue')
    ax1.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.axhline(0, color='grey', linewidth=0.5)
    ax1.set_ylabel('Amplitude (uV)')
    ax1.set_title(title)
    ax1.legend(loc='upper right', fontsize=8)

    # Bottom: difference
    if len(mean_a) == len(mean_b):
        diff = mean_a - mean_b
        ax2.plot(times_a, diff, color='darkorange', linewidth=1.5,
                 label=f'{label_a} - {label_b}')
        ax2.axvspan(N400_TMIN * 1000, N400_TMAX * 1000, alpha=0.12,
                    color='blue')
        n400_mask = (times_a >= N400_TMIN * 1000) & (times_a <= N400_TMAX * 1000)
        if n400_mask.any():
            n400_diff = diff[n400_mask]
            ax2.fill_between(times_a[n400_mask], 0, n400_diff,
                             where=(n400_diff < 0), alpha=0.3, color='red')
            ax2.text(0.98, 0.02, f'N400 mean diff: {n400_diff.mean():.2f} uV',
                     transform=ax2.transAxes, fontsize=9, ha='right',
                     va='bottom',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.axhline(0, color='grey', linewidth=0.8, linestyle='-')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude (uV)')
    ax2.set_title(f'Difference: {label_a} - {label_b}')
    ax2.legend(loc='upper right', fontsize=8)

    if ylim is not None:
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
    ax1.set_xticks(ERP_XTICKS)
    ax2.set_xticks(ERP_XTICKS)

    fig.tight_layout()
    return fig


def plot_three_way(evokeds_list, labels, roi_chs, title, ylim=None):
    """Plot 3 conditions overlaid (for BTP vs BBTP vs BBBTP)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ['steelblue', 'indianred', 'seagreen']

    for evk, label, color in zip(evokeds_list, labels, colors):
        avail = [ch for ch in roi_chs if ch in evk.ch_names]
        if not avail:
            continue
        idx = [evk.ch_names.index(ch) for ch in avail]
        roi_mean = evk.data[idx].mean(axis=0) * 1e6
        times = evk.times * 1000
        ax.plot(times, roi_mean, color=color, linewidth=1.5, label=label)

    ax.axvspan(N400_TMIN * 1000, N400_TMAX * 1000, alpha=0.12, color='blue',
               label='N400 window')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axhline(0, color='grey', linewidth=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xticks(ERP_XTICKS)
    fig.tight_layout()
    return fig


# ── CSV Export ───────────────────────────────────────────────────────────────

def export_group_table(grand_avgs, output_path, logger):
    """Export N400 metrics per group/condition/electrode to CSV."""
    import csv

    electrodes = REPORT_ELECTRODES + ['cluster']
    conditions = [c for c in ALL_CONDITIONS]

    with open(str(output_path), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Group', 'Condition', 'Electrode',
                         'Mean_Amplitude_uV', 'Peak_Amplitude_uV',
                         'Peak_Latency_ms', 'Area_Latency_50pct_ms'])

        for group in ('healthy', 'disorder'):
            if group not in grand_avgs:
                continue
            for cond in conditions:
                if cond not in grand_avgs[group]:
                    continue
                evk = grand_avgs[group][cond]
                metrics = measure_n400(evk, REPORT_ELECTRODES)
                for elec in electrodes:
                    if elec not in metrics:
                        continue
                    m = metrics[elec]
                    writer.writerow([
                        group, cond, elec,
                        f'{m["mean_amp"]:.3f}',
                        f'{m["peak_amp"]:.3f}',
                        f'{m["peak_lat"]:.1f}',
                        f'{m["area_lat_50"]:.1f}',
                    ])

    logger.info(f'Group table saved: {output_path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='N400 Group Analysis — Grand Averages and Comparisons')
    parser.add_argument('--input', required=True,
                        help='Directory containing *-ave.fif files')
    parser.add_argument('--output', required=True,
                        help='Output directory for group report and CSV')
    parser.add_argument('--uniform-scale', action='store_true',
                        help='Use the same y-axis scale across all ERP plots')
    args = parser.parse_args()

    mne.set_log_level('WARNING')

    script_dir = Path(__file__).resolve().parent
    logger = setup_logging(script_dir / 'logs')
    logger.info('=' * 60)
    logger.info('N400 Group Analysis')
    logger.info('=' * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────
    logger.info('Loading individual evokeds...')
    patients, groups = load_all_evokeds(args.input, logger)

    group_counts = {}
    for g in groups.values():
        group_counts[g] = group_counts.get(g, 0) + 1
    logger.info(f'Groups: {group_counts}')

    if len(patients) < 2:
        logger.warning('Need at least 2 patients for group analysis')

    # ── 2. Grand averages ───────────────────────────────────────────────
    logger.info('Computing grand averages...')
    grand_avgs = compute_grand_averages(patients, groups, ALL_CONDITIONS,
                                         logger)

    # ── 3. Export CSV table ─────────────────────────────────────────────
    csv_path = output_dir / 'n400_group_table.csv'
    export_group_table(grand_avgs, csv_path, logger)

    # ── Compute uniform y-axis limits if requested ──────────────────────
    ylim = None
    if args.uniform_scale:
        all_evokeds = {}
        for group in ('healthy', 'disorder'):
            if group in grand_avgs:
                for cond, evk in grand_avgs[group].items():
                    all_evokeds[f'{cond}_{group}'] = evk
        ylim = compute_uniform_ylim(all_evokeds, N400_ROI_CHS)
        if ylim:
            logger.info(f'Uniform y-axis: {ylim[0]:.1f} to {ylim[1]:.1f} uV')

    # ── 4. Generate report ──────────────────────────────────────────────
    logger.info('Generating group report...')
    report = mne.Report(title='N400 Group Analysis', verbose='WARNING')

    # Section 1: Group comparison per condition
    for cond in ALL_CONDITIONS:
        has_data = any(cond in grand_avgs.get(g, {})
                       for g in ('healthy', 'disorder'))
        if not has_data:
            continue
        fig = plot_group_comparison(grand_avgs, cond, N400_ROI_CHS, ylim=ylim)
        report.add_figure(fig, title=f'{cond} — group comparison',
                          tags=('group', 'evoked'))
        plt.close(fig)

    # Section 1b: Topographic maps per test condition per group
    test_conds = ('BTR', 'BTP', 'BBTR', 'BBTP', 'BBBTR', 'BBBTP')
    for cond in test_conds:
        for group in ('healthy', 'disorder'):
            if group not in grand_avgs or cond not in grand_avgs[group]:
                continue
            evk = grand_avgs[group][cond]
            try:
                fig = evk.plot_topomap(
                    times=TOPOMAP_TIMES, ch_type='eeg', average=None,
                    colorbar=True, show=False, time_unit='s')
            except Exception:
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.text(0.5, 0.5, f'{cond} [{group}]: topography unavailable',
                        ha='center', va='center', fontsize=12)
                ax.axis('off')
            report.add_figure(fig, title=f'{cond} [{group}] — Topography',
                              tags=('topomap', 'group'))
            plt.close(fig)

    # Section 1c: Averaged topography 400-500ms per test condition per group
    for cond in test_conds:
        for group in ('healthy', 'disorder'):
            if group not in grand_avgs or cond not in grand_avgs[group]:
                continue
            evk = grand_avgs[group][cond]
            try:
                fig = evk.plot_topomap(
                    times=[0.45], ch_type='eeg', average=0.1,
                    colorbar=True, show=False, time_unit='s')
                w, h = fig.get_size_inches()
                fig.set_size_inches(w + 1.5, h + 0.8)
                for ax in fig.get_axes():
                    if ax.get_label() == '<colorbar>':
                        pos = ax.get_position()
                        ax.set_position([pos.x0 + 0.08, pos.y0,
                                         pos.width, pos.height])
                try:
                    fig.subplots_adjust(top=0.82)
                except Exception:
                    pass
            except Exception:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.text(0.5, 0.5, f'{cond} [{group}]: avg topo unavailable',
                        ha='center', va='center', fontsize=10)
                ax.axis('off')
            report.add_figure(fig,
                              title=f'{cond} [{group}] — Avg topo 400-500ms',
                              tags=('topomap', 'group', 'averaged'))
            plt.close(fig)

    # Section 1d: Per-electrode ERPs per test condition per group
    for cond in test_conds:
        for group in ('healthy', 'disorder'):
            if group not in grand_avgs or cond not in grand_avgs[group]:
                continue
            evk = grand_avgs[group][cond]
            fig = plot_per_electrode_erps(evk, N400_ROI_CHS,
                                          f'{cond} [{group}]', ylim=ylim)
            report.add_figure(fig,
                              title=f'{cond} [{group}] — Per-electrode',
                              tags=('evoked', 'group', 'electrodes'))
            plt.close(fig)

    # Section 1e: Peak topomaps (Fz, Cz, Pz) per test condition per group
    for cond in test_conds:
        for group in ('healthy', 'disorder'):
            if group not in grand_avgs or cond not in grand_avgs[group]:
                continue
            evk = grand_avgs[group][cond]
            fig = plot_peak_topomaps(evk, f'{cond} [{group}]')
            report.add_figure(fig,
                              title=f'{cond} [{group}] — Peak topography',
                              tags=('topomap', 'group', 'electrodes'))
            plt.close(fig)

    # Section 1f: ROI channel overlay per test condition per group
    for cond in test_conds:
        for group in ('healthy', 'disorder'):
            if group not in grand_avgs or cond not in grand_avgs[group]:
                continue
            evk = grand_avgs[group][cond]
            fig = plot_roi_channel_overlay(evk, N400_ROI_CHS,
                                           f'{cond} [{group}]', ylim=ylim)
            report.add_figure(fig,
                              title=f'{cond} [{group}] — ROI channels',
                              tags=('evoked', 'group', 'roi'))
            plt.close(fig)

    # Section 1g: Best N400 channel summary per group
    test_conds_tuple = ('BTR', 'BTP', 'BBTR', 'BBTP', 'BBBTR', 'BBBTP')
    for group in ('healthy', 'disorder'):
        if group not in grand_avgs:
            continue
        ga = grand_avgs[group]
        fig, best_chs = plot_best_n400_summary(ga, test_conds_tuple,
                                                group.capitalize())
        report.add_figure(fig,
                          title=f'Best N400 channel [{group}]',
                          tags=('n400', 'best-channel', 'group'))
        plt.close(fig)
        if best_chs:
            logger.info(f'Best N400 channels [{group}]:')
            for cond, (ch, amp, lat) in best_chs.items():
                logger.info(f'  {cond}: {ch} ({amp:.1f} uV @ {lat:.0f} ms)')

    # Section 2: Within-block planned comparisons (per group)
    logger.info('Plotting within-block comparisons...')
    for comp_name, cond_a, cond_b in WITHIN_BLOCK:
        for group in ('healthy', 'disorder'):
            if group not in grand_avgs:
                continue
            ga = grand_avgs[group]
            if cond_a not in ga or cond_b not in ga:
                continue
            fig = plot_comparison_pair(
                ga[cond_a], ga[cond_b], cond_a, cond_b,
                N400_ROI_CHS,
                f'{comp_name} [{group}]', ylim=ylim)
            report.add_figure(fig, title=f'{comp_name} [{group}]',
                              tags=('comparison', 'within-block'))
            plt.close(fig)

    # Section 3: Between-block planned comparisons (per group)
    logger.info('Plotting between-block comparisons...')
    for comp_name, cond_a, cond_b in BETWEEN_BLOCK:
        for group in ('healthy', 'disorder'):
            if group not in grand_avgs:
                continue
            ga = grand_avgs[group]

            # Handle tuple (average of conditions)
            if isinstance(cond_a, tuple):
                avail_a = [ga[c] for c in cond_a if c in ga]
                if not avail_a:
                    continue
                evk_a = mne.grand_average(avail_a, interpolate_bads=True,
                                           drop_bads=False)
                label_a = '+'.join(cond_a)
            else:
                if cond_a not in ga:
                    continue
                evk_a = ga[cond_a]
                label_a = cond_a

            if isinstance(cond_b, tuple):
                avail_b = [ga[c] for c in cond_b if c in ga]
                if not avail_b:
                    continue
                evk_b = mne.grand_average(avail_b, interpolate_bads=True,
                                           drop_bads=False)
                label_b = '+'.join(cond_b)
            else:
                if cond_b not in ga:
                    continue
                evk_b = ga[cond_b]
                label_b = cond_b

            fig = plot_comparison_pair(
                evk_a, evk_b, label_a, label_b,
                N400_ROI_CHS,
                f'{comp_name} [{group}]', ylim=ylim)
            report.add_figure(fig, title=f'{comp_name} [{group}]',
                              tags=('comparison', 'between-block'))
            plt.close(fig)

    # Section 4: BTP vs BBTP vs BBBTP (per group)
    logger.info('Plotting BTP vs BBTP vs BBBTP...')
    for group in ('healthy', 'disorder'):
        if group not in grand_avgs:
            continue
        ga = grand_avgs[group]
        evks = [ga.get('BTP'), ga.get('BBTP'), ga.get('BBBTP')]
        labels = ['BTP', 'BBTP', 'BBBTP']
        valid = [(e, l) for e, l in zip(evks, labels) if e is not None]
        if len(valid) >= 2:
            fig = plot_three_way(
                [v[0] for v in valid], [v[1] for v in valid],
                N400_ROI_CHS,
                f'BTP vs BBTP vs BBBTP [{group}]', ylim=ylim)
            report.add_figure(fig,
                              title=f'BTP vs BBTP vs BBBTP [{group}]',
                              tags=('comparison', 'between-block'))
            plt.close(fig)

    # Save report
    report_path = output_dir / 'n400_group_report.html'
    report.save(str(report_path), overwrite=True, open_browser=False,
                verbose='WARNING')
    logger.info(f'Report saved: {report_path}')
    logger.info('Done.')


if __name__ == '__main__':
    main()
