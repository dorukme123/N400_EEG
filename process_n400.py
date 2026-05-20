"""N400 (Picture-match Paradigm) EEG Processing Pipeline.

Processes BrainVision EEG files through ICA artifact removal, epoching,
ERP averaging, and N400 analysis. Generates a self-contained HTML report
per patient using MNE-Python.

Usage:
    python process_n400.py --input <patient_folder_or_vhdr> --output <output_dir>
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import io
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for Russian/Unicode text
# Only when running as main script (not when imported by group_analysis)
if sys.platform == 'win32' and __name__ == '__main__':
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                      errors='replace')

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing import ICA

# ── Constants ────────────────────────────────────────────────────────────────

FILTER_L_FREQ = 0.1
PREPROCESS_H_FREQ = 45.0  # preprocessing filter (spec: 0.1-45 Hz)
N400_H_FREQ = 40.0         # N400 analysis filter (Table 8.1: 0.1-40 Hz)
RESAMPLE_FREQ = 256
ICA_N_COMPONENTS = 20
ICA_METHOD = 'fastica'
ICA_MAX_ITER = 1000
EPOCH_TMIN = -0.2
EPOCH_TMAX = 0.8
BASELINE = (-0.2, 0)
EPOCH_REJECT_DEFAULT = 150  # default uV peak-to-peak amplitude threshold
MONTAGE_NAME = 'standard_1005'  # 10-10 system per spec

# N400 analysis window (ms)
N400_TMIN = 0.2
N400_TMAX = 0.6

# Explicit x-axis ticks so 400 ms is always visible on ERP plots
ERP_XTICKS = [-200, 0, 200, 400, 600, 800]

# Time points for topographic maps (seconds, spanning N400 window)
TOPOMAP_TIMES = [0.2, 0.3, 0.4, 0.5, 0.6]

# Electrodes for peak-latency topographic maps
PEAK_TOPO_CHS = ['Fz', 'Cz', 'Pz']

# Event code substrings to match in BrainVision annotations
EVENT_MAP = {
    '134': 'BTR',
    '135': 'BTP',
    '136': 'BBTR',
    '137': 'BBTP',
    '138': 'BBBTR',
    '139': 'BBBTP',
}

# Electrodes of interest for N400 analysis
N400_ROI_CHS = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4',
                'CP1', 'CP2']

# Minimum time gap (seconds) between events to consider them non-duplicates.
# 500 Hz recordings have duplicate markers offset by ~7-8 samples (~14-16 ms).
DEDUP_MIN_GAP = 0.025  # 25 ms

# Bad channel detection threshold (MAD-based z-score)
BAD_CH_Z_THRESHOLD = 3.0

# Training condition constants
TRAINING_AUDIO_DELAY = 1.0  # seconds from picture to audio onset in training
PSEUDO_PAIRS = 3  # pseudoword pairs per block
REAL_PAIRS = 2    # real word pairs per block

# Repetition-level training conditions and their event IDs.
# Naming: BLP=1st rep, BLPP=2nd rep, BLPPP=3rd rep, BLPPPP=4th rep.
TRAINING_REP_MAP = {
    # (block, type, rep) -> (condition_name, event_id)
    (1, 'pseudo', 1): ('BLP', 9010),
    (1, 'pseudo', 2): ('BLPP', 9011),
    (1, 'pseudo', 3): ('BLPPP', 9012),
    (1, 'pseudo', 4): ('BLPPPP', 9013),
    (1, 'real', 1): ('BLR', 9014),
    (1, 'real', 2): ('BLRR', 9015),
    (2, 'pseudo', 1): ('BBLP', 9020),
    (2, 'pseudo', 2): ('BBLPP', 9021),
    (2, 'real', 1): ('BBLR', 9022),
    (2, 'real', 2): ('BBLRR', 9023),
    (3, 'pseudo', 1): ('BBBLP', 9030),
    (3, 'real', 1): ('BBBLR', 9031),
}

# Build lookup dicts from the map
TRAINING_EVENT_IDS = {name: eid for (name, eid) in TRAINING_REP_MAP.values()}

# All conditions in presentation order (test + training)
TEST_CONDITIONS = ('BTR', 'BTP', 'BBTR', 'BBTP', 'BBBTR', 'BBBTP')
TRAINING_CONDITIONS = (
    'BLR', 'BLRR', 'BLP', 'BLPP', 'BLPPP', 'BLPPPP',
    'BBLR', 'BBLRR', 'BBLP', 'BBLPP',
    'BBBLR', 'BBBLP',
)
ALL_CONDITIONS = TEST_CONDITIONS + TRAINING_CONDITIONS

# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_dir, patient_id):
    """Configure logging to console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{patient_id}_{timestamp}.log'

    logger = logging.getLogger(f'n400_{patient_id}')
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

# ── File Discovery ───────────────────────────────────────────────────────────

def discover_vhdr_files(input_path):
    """Find .vhdr files in the input path."""
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == '.vhdr':
        return [p]
    if p.is_dir():
        files = sorted(p.rglob('*.vhdr'))
        if not files:
            raise FileNotFoundError(f'No .vhdr files found in {p}')
        return files
    raise FileNotFoundError(f'Input path does not exist: {p}')


def extract_patient_id(vhdr_path):
    """Extract patient ID (e.g., INP0089) from filename."""
    match = re.search(r'(INP\d{4}(?:_RNS\d+)?)', vhdr_path.name)
    if match:
        return match.group(1)
    return vhdr_path.stem.split('_')[0]


def extract_visit_info(vhdr_path):
    """Extract visit info from directory structure for output naming."""
    for parent in vhdr_path.parents:
        name = parent.name
        if 'посещение' in name.lower() or 'визит' in name.lower():
            return name.replace(' ', '_')
    return 'visit_unknown'

# ── Channel Preparation ─────────────────────────────────────────────────────

def prepare_channels(raw, logger):
    """Set channel types and drop non-EEG channels."""
    ch_names = raw.ch_names
    logger.info(f'Channels ({len(ch_names)}): {ch_names}')
    logger.info(f'Sampling rate: {raw.info["sfreq"]} Hz')
    logger.info(f'Duration: {raw.times[-1]:.1f} s')

    # Set EOG channel type
    if 'EOG' in ch_names:
        raw.set_channel_types({'EOG': 'eog'})
        logger.info('Set EOG channel type to eog')
    else:
        logger.warning('No EOG channel found — will use Fp1 as proxy')

    # Drop BIP1 if present (not an EEG channel)
    if 'BIP1' in ch_names:
        raw.drop_channels(['BIP1'])
        logger.info('Dropped BIP1 channel')

    # Set montage (M1/M2 not in standard_1020, so ignore missing)
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    raw.set_montage(montage, on_missing='ignore')
    logger.info(f'Set {MONTAGE_NAME} montage')

    return raw

# ── Bad Channel Detection ───────────────────────────────────────────────────

def detect_bad_channels(raw, logger, z_threshold=BAD_CH_Z_THRESHOLD):
    """Detect noisy EEG channels using MAD-based z-scores of channel std.

    Only tests channels that have a montage position (excludes M1, M2).
    Returns list of bad channel names.
    """
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = [raw.ch_names[i] for i in eeg_picks]

    # Only test channels with montage positions (can be interpolated)
    montage = raw.get_montage()
    has_pos = set(montage.ch_names) if montage else set()
    testable = [(i, name) for i, name in zip(eeg_picks, ch_names)
                if name in has_pos]
    if not testable:
        return []

    picks_idx = [i for i, _ in testable]
    names = [name for _, name in testable]

    data = raw.get_data(picks=picks_idx)
    stds = np.std(data, axis=1)
    median_std = np.median(stds)
    mad = np.median(np.abs(stds - median_std))
    if mad == 0:
        return []

    z_scores = 0.6745 * (stds - median_std) / mad
    bad_chs = [names[i] for i in range(len(names)) if abs(z_scores[i]) > z_threshold]

    if bad_chs:
        for ch in bad_chs:
            idx = names.index(ch)
            logger.info(f'  Bad channel: {ch} (z={z_scores[idx]:.2f}, '
                        f'std={stds[idx]*1e6:.1f} uV)')

    return bad_chs

# ── Event Mapping ────────────────────────────────────────────────────────────

def build_event_id(annotations_event_id):
    """Map BrainVision annotations to our condition names.

    BrainVision annotations appear as 'Stimulus/s134' or similar.
    We match the numeric code substring to our EVENT_MAP.
    """
    event_id = {}
    for annot_key, annot_val in annotations_event_id.items():
        for code, condition in EVENT_MAP.items():
            if code in annot_key:
                event_id[condition] = annot_val
                break
    return event_id


def deduplicate_events(events, sfreq, logger):
    """Remove duplicate events from 500 Hz recordings.

    Some recordings have each marker duplicated with positions offset by
    ~7-8 samples. We keep the first occurrence when two identical event codes
    appear within DEDUP_MIN_GAP seconds of each other.
    """
    if len(events) == 0:
        return events

    min_gap_samples = int(DEDUP_MIN_GAP * sfreq)
    n_before = len(events)

    keep = [True] * n_before
    for i in range(1, n_before):
        if events[i, 2] == events[i - 1, 2]:
            gap = events[i, 0] - events[i - 1, 0]
            if gap < min_gap_samples:
                keep[i] = False

    events_clean = events[keep]
    n_removed = n_before - len(events_clean)
    if n_removed > 0:
        logger.info(f'Deduplicated events: removed {n_removed} duplicates '
                    f'({n_before} -> {len(events_clean)})')
    else:
        logger.info('No duplicate events detected')

    return events_clean

# ── Training Condition Assignment ────────────────────────────────────────────

def assign_training_conditions(events, all_event_id, sfreq, logger):
    """Create synthetic audio-onset events for training trials.

    Training trials only have picture onset markers (s203/s204). Audio onset
    occurs ~1000 ms later. This assigns each training trial to a repetition-
    level condition (BLP=1st rep, BLPP=2nd, etc.) by counting sequential
    occurrences within each training block phase.

    Within each training block, every PSEUDO_PAIRS (3) sequential s203 events
    form one repetition, and every REAL_PAIRS (2) sequential s204 events form
    one repetition.

    Returns (augmented_events, training_event_id).
    """
    def get_code(marker_str):
        for key, val in all_event_id.items():
            if marker_str in key:
                return val
        return None

    code_s167 = get_code('s167')
    code_s168 = get_code('s168')
    code_s169 = get_code('s169')
    code_s170 = get_code('s170')
    code_s171 = get_code('s171')
    code_s172 = get_code('s172')
    code_s203 = get_code('s203')
    code_s204 = get_code('s204')

    if code_s203 is None or code_s204 is None:
        logger.warning('Picture markers s203/s204 not found — skipping training')
        return events, {}

    boundary_codes = {code_s167, code_s168, code_s169, code_s170,
                      code_s171, code_s172}
    boundary_codes.discard(None)
    if len(boundary_codes) < 6:
        logger.warning('Some block boundary markers missing — skipping training')
        return events, {}

    audio_offset = int(round(TRAINING_AUDIO_DELAY * sfreq))
    state = 'training_1'
    pseudo_count = 0  # sequential s203 counter within current training phase
    real_count = 0    # sequential s204 counter within current training phase
    new_events = []

    for i in range(len(events)):
        code = events[i, 2]

        if code == code_s167:
            state = 'test_1'
        elif code == code_s168:
            state = 'training_2'
            pseudo_count = 0
            real_count = 0
        elif code == code_s169:
            state = 'test_2'
        elif code == code_s170:
            state = 'training_3'
            pseudo_count = 0
            real_count = 0
        elif code == code_s171:
            state = 'test_3'
        elif code == code_s172:
            state = 'training_1'
            pseudo_count = 0
            real_count = 0
        elif state.startswith('training_'):
            block = int(state.split('_')[1])
            if code == code_s203:
                pseudo_count += 1
                rep = (pseudo_count - 1) // PSEUDO_PAIRS + 1
                key = (block, 'pseudo', rep)
                if key in TRAINING_REP_MAP:
                    cond, eid = TRAINING_REP_MAP[key]
                    pos = events[i, 0] + audio_offset
                    new_events.append([pos, 0, eid])
            elif code == code_s204:
                real_count += 1
                rep = (real_count - 1) // REAL_PAIRS + 1
                key = (block, 'real', rep)
                if key in TRAINING_REP_MAP:
                    cond, eid = TRAINING_REP_MAP[key]
                    pos = events[i, 0] + audio_offset
                    new_events.append([pos, 0, eid])

    if not new_events:
        logger.warning('No training events identified')
        return events, {}

    new_events = np.array(new_events, dtype=events.dtype)

    all_events = np.vstack([events, new_events])
    order = np.argsort(all_events[:, 0])
    all_events = all_events[order]

    training_event_id = {}
    for cond, eid in TRAINING_EVENT_IDS.items():
        count = np.sum(new_events[:, 2] == eid)
        if count > 0:
            training_event_id[cond] = eid
            logger.info(f'  {cond}: {count} training events')

    logger.info(f'Created {len(new_events)} synthetic training events '
                f'(audio offset: +{TRAINING_AUDIO_DELAY*1000:.0f} ms)')

    return all_events, training_event_id

# ── Patient Info Lookup ───────────────────────────────────────────────────────

def lookup_patient_info(patient_id, script_dir):
    """Look up patient age, sex, diagnosis from Filename_to_patientdata.xlsx.

    Returns dict with keys: age (float or None), sex (str or None),
    diagnosis (str or None), age_group ('child'|'adult'|'unknown').
    """
    info = {'age': None, 'sex': None, 'diagnosis': None, 'age_group': 'unknown'}
    # Look in script dir first, then parent (legacy layout)
    xlsx_path = script_dir / 'Filename_to_patientdata.xlsx'
    if not xlsx_path.exists():
        xlsx_path = script_dir.parent / 'Filename_to_patientdata.xlsx'
    if not xlsx_path.exists():
        return info
    try:
        import openpyxl
        wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row[0] and str(row[0]).strip() == patient_id:
                # Column order: ID, Name, Sex, Age, Diagnosis
                info['sex'] = str(row[2]).strip() if row[2] else None
                if row[3] is not None:
                    age_str = str(row[3]).replace(',', '.')
                    try:
                        info['age'] = float(age_str)
                    except ValueError:
                        pass
                info['diagnosis'] = str(row[4]).strip() if row[4] else None
                break
        wb.close()
    except Exception:
        pass

    if info['age'] is not None:
        info['age_group'] = 'child' if info['age'] < 18 else 'adult'

    return info


# ── Post-Processing Validation ───────────────────────────────────────────────

# Normative thresholds by age group, with literature references.
#
# Each dict maps: parameter -> (normal_range, warning_range)
# where ranges are (low, high) inclusive.

NORMS = {
    'child': {
        'n400_amp_normal':   (-10.0, -0.5),
        'n400_amp_warn':     (-15.0, -0.2),
        'n400_lat_normal':   (200, 600),
        'n400_lat_warn':     (150, 700),
        'erp_amp_max':       25.0,
        'min_epochs':        5,
        'baseline_max':      0.5,
        'reject_warn':       15,
        'reject_fail':       50,
    },
    'adult': {
        'n400_amp_normal':   (-8.0, -0.5),
        'n400_amp_warn':     (-12.0, -0.2),
        'n400_lat_normal':   (250, 500),
        'n400_lat_warn':     (200, 600),
        'erp_amp_max':       20.0,
        'min_epochs':        10,
        'baseline_max':      0.5,
        'reject_warn':       15,
        'reject_fail':       50,
    },
    'unknown': {
        # Conservative: union of child + adult ranges
        'n400_amp_normal':   (-10.0, -0.5),
        'n400_amp_warn':     (-15.0, -0.2),
        'n400_lat_normal':   (200, 600),
        'n400_lat_warn':     (150, 700),
        'erp_amp_max':       25.0,
        'min_epochs':        5,
        'baseline_max':      0.5,
        'reject_warn':       15,
        'reject_fail':       50,
    },
}

# Full citation block printed at the end of each validation report.
REFERENCES = """
REFERENCES — normative values and thresholds used in this validation:

[1] Kutas M, Federmeier KD (2011).
    "Thirty years and counting: Finding meaning in the N400 component
    of the event-related brain potential (ERP)."
    Annual Review of Psychology, 62:621-647.
    doi:10.1146/annurev.psych.093008.131123
    >> N400 peaks between 200-600 ms post-stimulus onset at
       centroparietal sites; amplitude modulated by semantic
       expectancy and context.
    >> Used for: N400 time window, amplitude interpretation.

[2] Duncan CC, Barry RJ, Connolly JF, Fischer C, Michie PT, Naatanen R,
    Polich J, Reinvang I, Van Petten C (2009).
    "Event-related potentials in clinical research: Guidelines for eliciting,
    recording, and quantifying mismatch negativity, P300, and N400."
    Clinical Neurophysiology, 120(11):1883-1908.
    doi:10.1016/j.clinph.2009.07.045
    >> N400 amplitude typically 2-10 uV (negative) at centro-parietal sites.
    >> Amplitude rejection threshold: +/-75 to +/-150 uV typical.
    >> Used for: amplitude ranges, epoch rejection thresholds, QC criteria.

[3] Friedrich M, Friederici AD (2010).
    "Maturing brain mechanisms and developing behavioral language skills."
    Brain and Language, 114(2):66-71.
    doi:10.1016/j.bandl.2009.07.004
    >> Infants with high language production show N400 effect.
    >> N400 is functional in children with high language skills.
    >> Used for: child N400 presence as a marker of language ability.

[4] Borgstrom K, von Koss Torkildsen J, Lindgren M (2015).
    "Substantial gains in word learning ability between 20 and 24 months."
    Brain and Language, 149:33-45.
    doi:10.1016/j.bandl.2015.07.002
    >> Children at 24 months show N400 for incongruent pseudoword pairs
       after only 5 training exposures (more negative for incongruent).
    >> Used for: child N400 amplitude and learning paradigm validation.

[5] Delogu F, Brouwer H, Crocker MW (2019).
    "Event-related potentials index lexical retrieval (N400) and
    integration (P600) during language comprehension."
    Brain and Cognition, 2019.
    >> N400 amplitude is less negative for more predictable words.
    >> Used for: congruency effect interpretation.

[6] Cacioppo JT, Tassinary LG, Berntson GG (2007).
    "Handbook of Psychophysiology."
    >> N400 sensitive to semantic relationships; degree of semantic
       proximity is determining factor.
    >> Used for: semantic priming context.

[7] Picton TW, Bentin S, Berg P, Donchin E, Hillyard SA, Johnson R Jr,
    Miller GA, Ritter W, Ruchkin DS, Rugg MD, Taylor MJ (2000).
    "Guidelines for using human event-related potentials to study
    cognition: Recording standards and publication criteria."
    Psychophysiology, 37(2):127-152.
    doi:10.1111/1469-8986.3720127
    >> Baseline correction: pre-stimulus interval should yield mean ~ 0 uV.
    >> ERP amplitudes exceeding 20-25 uV suggest residual artifact.
    >> Used for: baseline and ERP amplitude sanity checks.
"""


def validate_results(patient_id, epochs, evokeds, n400_diffs, ica,
                     eog_indices, n_dropped, bad_chs, patient_info, logger):
    """Run validation checks on processed data and log a structured report.

    Uses age-appropriate normative thresholds from published literature.
    Logs PASS/WARNING/FAIL for each check, with a full reference list.
    """
    age_group = patient_info.get('age_group', 'unknown')
    norms = NORMS[age_group]

    logger.info('')
    logger.info('=' * 70)
    logger.info('  VALIDATION REPORT: %s', patient_id)
    logger.info('=' * 70)

    # Log patient info
    logger.info('--- Patient Info ---')
    age = patient_info.get('age')
    sex = patient_info.get('sex')
    diag = patient_info.get('diagnosis')
    logger.info('  Age:       %s', f'{age} years' if age else 'unknown')
    logger.info('  Sex:       %s', sex if sex else 'unknown')
    logger.info('  Diagnosis: %s', diag if diag else 'unknown')
    logger.info('  Norms:     %s', age_group.upper())
    if age_group == 'unknown':
        logger.info('             (using conservative union of child + adult ranges)')

    warnings_count = 0
    fails_count = 0

    def log_check(name, status, detail, refs=''):
        nonlocal warnings_count, fails_count
        ref_str = f'  [{refs}]' if refs else ''
        if status == 'FAIL':
            fails_count += 1
            logger.error('  [FAIL]    %-30s  %s%s', name, detail, ref_str)
        elif status == 'WARNING':
            warnings_count += 1
            logger.warning('  [WARNING] %-30s  %s%s', name, detail, ref_str)
        else:
            logger.info('  [PASS]    %-30s  %s%s', name, detail, ref_str)

    # ── 0. Bad channels ─────────────────────────────────────────────────
    logger.info('--- Bad Channels ---')
    if bad_chs:
        log_check('Bad channels', 'WARNING',
                  f'{len(bad_chs)} detected and interpolated: {bad_chs}')
    else:
        log_check('Bad channels', 'PASS', 'none detected')

    # ── 1. Epoch counts ──────────────────────────────────────────────────
    logger.info('--- Epoch Counts [ref 2] ---')
    total_epochs = len(epochs)
    event_id = epochs.event_id
    min_ep = norms['min_epochs']

    for cond in ALL_CONDITIONS:
        if cond not in event_id:
            log_check(f'{cond} epochs', 'FAIL',
                      'condition missing from data', 'ref 2')
            continue
        n = len(epochs[cond])
        if n == 0:
            log_check(f'{cond} epochs', 'FAIL',
                      f'{n} epochs (none survived)', 'ref 2')
        elif n < min_ep:
            log_check(f'{cond} epochs', 'WARNING',
                      f'{n} epochs (< {min_ep} recommended minimum)',
                      'ref 2')
        else:
            log_check(f'{cond} epochs', 'PASS', f'{n} epochs', 'ref 2')

    drop_pct = (n_dropped / (total_epochs + n_dropped) * 100
                if (total_epochs + n_dropped) > 0 else 0)
    if drop_pct > norms['reject_fail']:
        log_check('Epoch rejection rate', 'FAIL',
                  f'{n_dropped} dropped ({drop_pct:.1f}% > '
                  f'{norms["reject_fail"]}% limit)', 'ref 2')
    elif drop_pct > norms['reject_warn']:
        log_check('Epoch rejection rate', 'WARNING',
                  f'{n_dropped} dropped ({drop_pct:.1f}% > '
                  f'{norms["reject_warn"]}% quality threshold)', 'ref 2')
    else:
        log_check('Epoch rejection rate', 'PASS',
                  f'{n_dropped} dropped ({drop_pct:.1f}%)', 'ref 2')

    # ── 2. ICA quality ───────────────────────────────────────────────────
    logger.info('--- ICA [ref 7] ---')
    n_excluded = len(eog_indices)
    if n_excluded == 0:
        log_check('ICA EOG exclusion', 'WARNING',
                  'no components excluded (eye artifacts may remain)', 'ref 7')
    elif n_excluded > 5:
        log_check('ICA EOG exclusion', 'WARNING',
                  f'{n_excluded} excluded (unusually many — possible '
                  f'over-rejection)', 'ref 7')
    else:
        log_check('ICA EOG exclusion', 'PASS',
                  f'{n_excluded} component(s) excluded: '
                  f'{list(eog_indices)}', 'ref 7')

    # ── 3. Baseline ──────────────────────────────────────────────────────
    logger.info('--- Baseline [ref 7] ---')
    # Use the first available condition for baseline check
    for cond in ('BTR', 'BTP', 'BBTR', 'BBTP', 'BBBTR', 'BBBTP'):
        if cond in evokeds:
            evk = evokeds[cond]
            bl_mask = evk.times <= 0
            bl_data = evk.data[:, bl_mask] * 1e6
            bl_mean_abs = np.abs(bl_data.mean())
            bl_mean_val = bl_data.mean()
            if bl_mean_abs > norms['baseline_max']:
                log_check('Baseline correction', 'WARNING',
                          f'mean = {bl_mean_val:.4f} uV '
                          f'(|mean| > {norms["baseline_max"]} uV, '
                          f'checked on {cond})', 'ref 7')
            else:
                log_check('Baseline correction', 'PASS',
                          f'mean = {bl_mean_val:.4f} uV '
                          f'(checked on {cond})', 'ref 7')
            break

    # ── 4. ERP amplitudes ────────────────────────────────────────────────
    erp_max = norms['erp_amp_max']
    logger.info('--- ERP Amplitudes [ref 7] ---')
    for name, evk in evokeds.items():
        amp_range = evk.data * 1e6
        amp_min, amp_max = amp_range.min(), amp_range.max()
        peak = max(abs(amp_min), abs(amp_max))
        if peak > erp_max:
            log_check(f'{name} amplitude', 'WARNING',
                      f'range [{amp_min:.1f}, {amp_max:.1f}] uV '
                      f'(peak {peak:.1f} > {erp_max} uV limit)',
                      'ref 7')
        else:
            log_check(f'{name} amplitude', 'PASS',
                      f'range [{amp_min:.1f}, {amp_max:.1f}] uV', 'ref 7')

    # ── 5. N400 checks ───────────────────────────────────────────────────
    logger.info('--- N400 Component [ref 1,2,3,5] ---')
    # Use first available evoked to determine ROI channels
    first_evk = next(iter(evokeds.values()), None)
    if first_evk is not None:
        available_roi = [ch for ch in N400_ROI_CHS
                         if ch in first_evk.ch_names]
    else:
        available_roi = []

    amp_normal = norms['n400_amp_normal']
    amp_warn = norms['n400_amp_warn']
    lat_normal = norms['n400_lat_normal']
    lat_warn = norms['n400_lat_warn']

    for name, evk in evokeds.items():
        if not available_roi:
            log_check(f'{name} N400', 'FAIL',
                      'no ROI channels found', 'ref 1')
            continue

        # N400 window: 200-600 ms
        t_mask = (evk.times >= N400_TMIN) & (evk.times <= N400_TMAX)
        if not t_mask.any():
            log_check(f'{name} N400 window', 'FAIL',
                      'no data in 200-600ms window', 'ref 1')
            continue

        # Find peak negative amplitude across ROI channels
        best_ch = None
        best_peak = 0
        best_lat = 0
        for ch in available_roi:
            ch_idx = evk.ch_names.index(ch)
            data_uv = evk.data[ch_idx, t_mask] * 1e6
            times_ms = evk.times[t_mask] * 1000
            peak_val = data_uv.min()
            peak_lat = times_ms[np.argmin(data_uv)]
            if peak_val < best_peak:
                best_peak = peak_val
                best_lat = peak_lat
                best_ch = ch

        if best_ch is None:
            log_check(f'{name} N400 peak', 'WARNING',
                      'no negative peak found in ROI', 'ref 1')
            continue

        # Amplitude check
        if amp_normal[0] <= best_peak <= amp_normal[1]:
            log_check(f'{name} N400 amplitude', 'PASS',
                      f'{best_peak:.2f} uV at {best_ch} '
                      f'(normal: {amp_normal[0]} to {amp_normal[1]} uV)',
                      'ref 1,2')
        elif amp_warn[0] <= best_peak <= amp_warn[1]:
            log_check(f'{name} N400 amplitude', 'WARNING',
                      f'{best_peak:.2f} uV at {best_ch} '
                      f'(outside normal {amp_normal[0]} to {amp_normal[1]} uV)',
                      'ref 1,2')
        else:
            log_check(f'{name} N400 amplitude', 'FAIL',
                      f'{best_peak:.2f} uV at {best_ch} '
                      f'(outside range {amp_warn[0]} to {amp_warn[1]} uV)',
                      'ref 1,2')

        # Latency check
        if lat_normal[0] <= best_lat <= lat_normal[1]:
            log_check(f'{name} N400 latency', 'PASS',
                      f'{best_lat:.0f} ms at {best_ch} '
                      f'(normal: {lat_normal[0]}-{lat_normal[1]} ms)',
                      'ref 1')
        elif lat_warn[0] <= best_lat <= lat_warn[1]:
            log_check(f'{name} N400 latency', 'WARNING',
                      f'{best_lat:.0f} ms at {best_ch} '
                      f'(outside normal {lat_normal[0]}-{lat_normal[1]} ms)',
                      'ref 1')
        else:
            log_check(f'{name} N400 latency', 'FAIL',
                      f'{best_lat:.0f} ms at {best_ch} '
                      f'(expected {lat_warn[0]}-{lat_warn[1]} ms)',
                      'ref 1')

    # ── 6. N400 effect (difference waves) ────────────────────────────────
    logger.info('--- N400 Effect (difference waves) [ref 1,5] ---')
    for name, diff in n400_diffs.items():
        if not available_roi:
            log_check(f'{name} N400 effect', 'FAIL',
                      'no ROI channels found', 'ref 1')
            continue

        t_mask = (diff.times >= N400_TMIN) & (diff.times <= N400_TMAX)
        if not t_mask.any():
            log_check(f'{name} N400 effect', 'FAIL',
                      'no data in 200-600ms window', 'ref 1')
            continue

        # Check for negative deflection in the difference wave (N400 effect)
        neg_count = 0
        mean_vals = []
        for ch in available_roi:
            ch_idx = diff.ch_names.index(ch)
            mean_val = (diff.data[ch_idx, t_mask] * 1e6).mean()
            mean_vals.append(mean_val)
            if mean_val < 0:
                neg_count += 1

        roi_mean = np.mean(mean_vals)
        if neg_count >= len(available_roi) * 0.5:
            log_check(f'{name} N400 effect', 'PASS',
                      f'negative at {neg_count}/{len(available_roi)} ROI '
                      f'channels (mean = {roi_mean:.2f} uV)', 'ref 1,5')
        else:
            log_check(f'{name} N400 effect', 'WARNING',
                      f'negative at only {neg_count}/{len(available_roi)} ROI '
                      f'channels (mean = {roi_mean:.2f} uV)', 'ref 1,5')

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info('--- Summary ---')
    if fails_count == 0 and warnings_count == 0:
        logger.info('  RESULT: ALL CHECKS PASSED')
    elif fails_count == 0:
        logger.warning('  RESULT: PASSED with %d warning(s)', warnings_count)
    else:
        logger.error('  RESULT: %d FAIL(s), %d warning(s)',
                     fails_count, warnings_count)

    # ── References ───────────────────────────────────────────────────────
    for line in REFERENCES.strip().splitlines():
        logger.info('  %s', line)
    logger.info('=' * 70)

    return fails_count, warnings_count


# ── Per-Trial N400 Table ─────────────────────────────────────────────────────

def export_per_trial_n400(epochs, patient_id, output_dir, logger):
    """Export per-trial N400 peak amplitude for each electrode and condition.

    Spec: table per patient, 200-600 ms window, 62 electrodes (no M1/M2),
    for each test condition (BTR, BTP, BBTR, BBTP, BBBTR, BBBTP), per trial.

    Saves CSV: rows = (condition, trial#), columns = electrodes, values = peak
    amplitude (uV) in the N400 window.
    """
    import csv

    # Exclude M1, M2, EOG from electrode list
    exclude = {'M1', 'M2', 'EOG'}
    eeg_chs = [ch for ch in epochs.ch_names
                if ch not in exclude
                and epochs.info['chs'][epochs.ch_names.index(ch)]['kind']
                == mne.io.constants.FIFF.FIFFV_EEG_CH]

    n400_tmin_idx = np.argmin(np.abs(epochs.times - N400_TMIN))
    n400_tmax_idx = np.argmin(np.abs(epochs.times - N400_TMAX))
    t_slice = slice(n400_tmin_idx, n400_tmax_idx + 1)

    output_path = Path(output_dir) / f'{patient_id}_per_trial_n400.csv'

    with open(str(output_path), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Condition', 'Trial'] + eeg_chs)

        for cond in TEST_CONDITIONS:
            if cond not in epochs.event_id:
                continue
            cond_epochs = epochs[cond]
            data = cond_epochs.get_data()  # (n_trials, n_channels, n_times)

            for trial_idx in range(data.shape[0]):
                row = [cond, trial_idx + 1]
                for ch in eeg_chs:
                    ch_idx = epochs.ch_names.index(ch)
                    segment_uv = data[trial_idx, ch_idx, t_slice] * 1e6
                    peak = segment_uv.min()  # N400 = most negative peak
                    row.append(f'{peak:.3f}')
                writer.writerow(row)

    logger.info(f'Per-trial N400 table saved: {output_path}')


# ── Custom ERP Figures ───────────────────────────────────────────────────────

def compute_uniform_ylim(evokeds_dict, roi_chs, margin=1.1):
    """Compute global (ymin, ymax) across all evokeds for ROI channels.

    Returns (ymin, ymax) in uV with a small margin, or None if empty.
    """
    global_min = np.inf
    global_max = -np.inf
    for evk in evokeds_dict.values():
        available = [ch for ch in roi_chs if ch in evk.ch_names]
        if not available:
            continue
        idx = [evk.ch_names.index(ch) for ch in available]
        data_uv = evk.data[idx] * 1e6
        global_min = min(global_min, data_uv.min())
        global_max = max(global_max, data_uv.max())
    if np.isinf(global_min):
        return None
    span = global_max - global_min
    pad = span * (margin - 1) / 2
    return (global_min - pad, global_max + pad)


def plot_erp_with_n400_window(evoked, title, roi_chs=None, ylim=None):
    """Plot ERP butterfly + ROI average with N400 window shading.

    Returns a matplotlib Figure with two subplots:
      - Top: butterfly plot of all channels
      - Bottom: ROI channel average with N400 window highlighted
    If ylim is given as (ymin, ymax), both axes use that range.
    """
    times = evoked.times * 1000  # to ms
    data_uv = evoked.data * 1e6  # to uV

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [1.2, 1]})

    # ── Top: butterfly plot (ROI channels only) ────────────────────────
    if roi_chs:
        roi_available = [ch for ch in roi_chs if ch in evoked.ch_names]
        roi_idx = [evoked.ch_names.index(ch) for ch in roi_available]
        butterfly_data = data_uv[roi_idx]
    else:
        roi_available = evoked.ch_names
        butterfly_data = data_uv

    for i in range(butterfly_data.shape[0]):
        ax1.plot(times, butterfly_data[i], linewidth=0.6, alpha=0.6)
    ax1.axvspan(N400_TMIN * 1000, N400_TMAX * 1000,
                alpha=0.15, color='blue', label='N400 window')
    ax1.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.axhline(0, color='grey', linewidth=0.5)
    ax1.set_ylabel('Amplitude (uV)')
    ax1.set_title(f'{title} — ROI channels ({len(roi_available)} ch, '
                  f'N_ave={evoked.nave})')
    ax1.legend(loc='upper right', fontsize=8)

    # ── Bottom: ROI average ──────────────────────────────────────────────
    if roi_chs:
        available = [ch for ch in roi_chs if ch in evoked.ch_names]
    else:
        available = []

    if available:
        roi_idx = [evoked.ch_names.index(ch) for ch in available]
        roi_mean = data_uv[roi_idx].mean(axis=0)
        roi_sem = data_uv[roi_idx].std(axis=0) / np.sqrt(len(roi_idx))

        ax2.fill_between(times, roi_mean - roi_sem, roi_mean + roi_sem,
                         alpha=0.25, color='steelblue')
        ax2.plot(times, roi_mean, color='steelblue', linewidth=1.5,
                 label=f'ROI mean ({len(available)} ch)')

        # Mark peak in N400 window
        n400_mask = (times >= N400_TMIN * 1000) & (times <= N400_TMAX * 1000)
        if n400_mask.any():
            peak_idx = np.argmin(roi_mean[n400_mask])
            peak_time = times[n400_mask][peak_idx]
            peak_amp = roi_mean[n400_mask][peak_idx]
            ax2.plot(peak_time, peak_amp, 'rv', markersize=8)
            ax2.annotate(f'{peak_amp:.1f} uV @ {peak_time:.0f} ms',
                         xy=(peak_time, peak_amp),
                         xytext=(peak_time + 30, peak_amp - 2),
                         fontsize=8, color='red',
                         arrowprops=dict(arrowstyle='->', color='red',
                                         lw=0.8))
    else:
        ax2.plot(times, data_uv.mean(axis=0), color='steelblue',
                 linewidth=1.5, label='Global mean')

    ax2.axvspan(N400_TMIN * 1000, N400_TMAX * 1000,
                alpha=0.15, color='blue')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.axhline(0, color='grey', linewidth=0.5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude (uV)')
    ax2.set_title(f'{title} — ROI average')
    ax2.legend(loc='upper right', fontsize=8)

    if ylim is not None:
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
    ax1.set_xticks(ERP_XTICKS)
    ax2.set_xticks(ERP_XTICKS)

    fig.tight_layout()
    return fig


def plot_diff_wave(diff_evoked, title, roi_chs=None, ylim=None):
    """Plot difference wave (real - pseudo) with N400 window.

    Returns a matplotlib Figure showing the ROI-averaged difference wave
    with the N400 window highlighted and zero line.
    If ylim is given as (ymin, ymax), the axis uses that range.
    """
    times = diff_evoked.times * 1000
    data_uv = diff_evoked.data * 1e6

    fig, ax = plt.subplots(figsize=(10, 4))

    if roi_chs:
        available = [ch for ch in roi_chs if ch in diff_evoked.ch_names]
    else:
        available = []

    if available:
        roi_idx = [diff_evoked.ch_names.index(ch) for ch in available]
        roi_mean = data_uv[roi_idx].mean(axis=0)
        roi_sem = data_uv[roi_idx].std(axis=0) / np.sqrt(len(roi_idx))

        ax.fill_between(times, roi_mean - roi_sem, roi_mean + roi_sem,
                        alpha=0.25, color='darkorange')
        ax.plot(times, roi_mean, color='darkorange', linewidth=1.5,
                label=f'Difference (ROI mean, {len(available)} ch)')

        # Shade where difference is negative in N400 window
        n400_mask = (times >= N400_TMIN * 1000) & (times <= N400_TMAX * 1000)
        if n400_mask.any():
            n400_times = times[n400_mask]
            n400_diff = roi_mean[n400_mask]
            ax.fill_between(n400_times, 0, n400_diff,
                            where=(n400_diff < 0), alpha=0.3, color='red',
                            label='N400 effect (negative)')
            mean_amp = n400_diff.mean()
            ax.text(0.98, 0.02,
                    f'N400 window mean: {mean_amp:.2f} uV',
                    transform=ax.transAxes, fontsize=9,
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax.plot(times, data_uv.mean(axis=0), color='darkorange',
                linewidth=1.5, label='Difference (global mean)')

    ax.axvspan(N400_TMIN * 1000, N400_TMAX * 1000,
               alpha=0.12, color='blue', label='N400 window')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='-')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xticks(ERP_XTICKS)

    fig.tight_layout()
    return fig


def plot_roi_channel_overlay(evoked, roi_chs, title, ylim=None):
    """Plot each ROI channel as a separate line on one graph.

    Shows individual channel contributions to the ROI average,
    with N400 window shading and a legend identifying each channel.
    If ylim is given as (ymin, ymax), the axis uses that range.
    """
    times = evoked.times * 1000
    data_uv = evoked.data * 1e6

    available = [ch for ch in roi_chs if ch in evoked.ch_names]
    if not available:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f'{title}: no ROI channels available',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    # Use a distinguishable color cycle
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(available))]

    fig, ax = plt.subplots(figsize=(12, 5))

    for ch, color in zip(available, colors):
        idx = evoked.ch_names.index(ch)
        ax.plot(times, data_uv[idx], color=color, linewidth=1.2,
                alpha=0.8, label=ch)

    # ROI average as thick dashed line
    roi_idx = [evoked.ch_names.index(ch) for ch in available]
    roi_mean = data_uv[roi_idx].mean(axis=0)
    ax.plot(times, roi_mean, color='black', linewidth=2.0, linestyle='--',
            alpha=0.9, label='ROI mean')

    ax.axvspan(N400_TMIN * 1000, N400_TMAX * 1000,
               alpha=0.12, color='blue')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.axhline(0, color='grey', linewidth=0.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title(f'{title} — Individual ROI channels ({len(available)} ch)')
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    ax.set_xticks(ERP_XTICKS)

    fig.tight_layout()
    return fig


def plot_condition_topomaps(evoked, title):
    """Plot topographic maps at several time points across the N400 window.

    Returns a matplotlib Figure with one topomap per time point.
    """
    try:
        fig = evoked.plot_topomap(
            times=TOPOMAP_TIMES, ch_type='eeg', average=None,
            colorbar=True, show=False, time_unit='s')
        return fig
    except Exception:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, f'{title}: topographic map unavailable',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig


def plot_per_electrode_erps(evoked, roi_chs, title, ylim=None):
    """Plot ERP waveform for each ROI electrode in an anatomical grid.

    Returns a matplotlib Figure with 4x3 subplots:
      Row 0: F3, Fz, F4
      Row 1: C3, Cz, C4
      Row 2: CP1, (empty), CP2
      Row 3: P3, Pz, P4
    If ylim is given as (ymin, ymax), all subplots use that range.
    """
    grid = [
        ['F3', 'Fz', 'F4'],
        ['C3', 'Cz', 'C4'],
        ['CP1', None, 'CP2'],
        ['P3', 'Pz', 'P4'],
    ]
    times = evoked.times * 1000
    data_uv = evoked.data * 1e6

    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle(f'{title} — Per-electrode ERPs', fontsize=13, y=0.98)

    for r, row in enumerate(grid):
        for c, ch_name in enumerate(row):
            ax = axes[r][c]
            if ch_name is None:
                ax.axis('off')
                continue
            if ch_name not in evoked.ch_names:
                ax.text(0.5, 0.5, f'{ch_name}\n(missing)', ha='center',
                        va='center', fontsize=9, color='grey',
                        transform=ax.transAxes)
                ax.axis('off')
                continue

            idx = evoked.ch_names.index(ch_name)
            trace = data_uv[idx]

            ax.plot(times, trace, color='steelblue', linewidth=1.2)
            ax.axvspan(N400_TMIN * 1000, N400_TMAX * 1000,
                       alpha=0.12, color='blue')
            ax.axvline(0, color='black', linewidth=0.6, linestyle='--')
            ax.axhline(0, color='grey', linewidth=0.4)

            # Peak in N400 window
            n400_mask = (times >= N400_TMIN * 1000) & (times <= N400_TMAX * 1000)
            if n400_mask.any():
                peak_idx = np.argmin(trace[n400_mask])
                peak_time = times[n400_mask][peak_idx]
                peak_amp = trace[n400_mask][peak_idx]
                ax.plot(peak_time, peak_amp, 'rv', markersize=5)
                ax.text(0.02, 0.02, f'{peak_amp:.1f}uV\n{peak_time:.0f}ms',
                        fontsize=6, transform=ax.transAxes, va='bottom',
                        color='red')

            ax.set_title(ch_name, fontsize=10, fontweight='bold')
            ax.set_xticks(ERP_XTICKS)
            ax.tick_params(labelsize=7)

    # Axis labels on edge subplots only
    for ax in axes[-1]:
        if ax.axison:
            ax.set_xlabel('Time (ms)', fontsize=8)
    for ax in axes[:, 0]:
        if ax.axison:
            ax.set_ylabel('uV', fontsize=8)

    if ylim is not None:
        axes[0][0].set_ylim(ylim)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def find_best_n400_channels(evoked):
    """Find the channel with the strongest N400 peak (most negative) in 200-600ms.

    Searches all EEG channels (excluding M1, M2, EOG).
    Returns (best_ch, peak_amp_uv, peak_lat_ms) or (None, None, None).
    """
    exclude = {'M1', 'M2', 'EOG'}
    t_mask = (evoked.times >= N400_TMIN) & (evoked.times <= N400_TMAX)
    if not t_mask.any():
        return None, None, None
    times_ms = evoked.times[t_mask] * 1000

    best_ch = None
    best_amp = 0  # looking for most negative
    best_lat = 0
    for i, ch in enumerate(evoked.ch_names):
        if ch in exclude:
            continue
        info_ch = evoked.info['chs'][i]
        if info_ch['kind'] != mne.io.constants.FIFF.FIFFV_EEG_CH:
            continue
        trace_uv = evoked.data[i, t_mask] * 1e6
        peak_val = trace_uv.min()
        if peak_val < best_amp:
            best_amp = peak_val
            best_lat = times_ms[np.argmin(trace_uv)]
            best_ch = ch
    return best_ch, best_amp, best_lat


def plot_best_n400_summary(evokeds, test_conditions, title):
    """Plot waveforms from the best N400 channel for each test condition.

    One subplot per condition showing the single channel where N400
    is most pronounced, plus a summary table.
    """
    results = {}
    for cond in test_conditions:
        if cond not in evokeds:
            continue
        ch, amp, lat = find_best_n400_channels(evokeds[cond])
        if ch is not None:
            results[cond] = (ch, amp, lat)

    if not results:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, f'{title}: no N400 peaks found',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, {}

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True,
                              squeeze=False)
    fig.suptitle(f'{title} — Best N400 channel per condition', fontsize=13,
                 y=1.0)

    for idx, (cond, (ch, amp, lat)) in enumerate(results.items()):
        ax = axes[idx][0]
        evk = evokeds[cond]
        ch_idx = evk.ch_names.index(ch)
        times = evk.times * 1000
        trace = evk.data[ch_idx] * 1e6

        ax.plot(times, trace, color='steelblue', linewidth=1.5)
        ax.axvspan(N400_TMIN * 1000, N400_TMAX * 1000,
                   alpha=0.12, color='blue')
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.plot(lat, amp, 'rv', markersize=10)
        ax.annotate(f'{amp:.1f} uV @ {lat:.0f} ms',
                     xy=(lat, amp), xytext=(lat + 40, amp - 2),
                     fontsize=9, color='red', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1))
        ax.set_ylabel('uV')
        ax.set_title(f'{cond} — best channel: {ch}  '
                     f'(peak: {amp:.1f} uV @ {lat:.0f} ms)',
                     fontsize=11, fontweight='bold')
        ax.set_xticks(ERP_XTICKS)

    axes[-1][0].set_xlabel('Time (ms)')
    fig.tight_layout()
    return fig, results


def plot_peak_topomaps(evoked, title):
    """Plot topographic maps at peak N400 latency of Fz, Cz, Pz.

    Returns a matplotlib Figure with 3 topographic head plots.
    """
    times_sec = evoked.times
    data_uv = evoked.data * 1e6
    n400_mask = (times_sec >= N400_TMIN) & (times_sec <= N400_TMAX)

    peak_times = []
    labels = []
    for ch in PEAK_TOPO_CHS:
        if ch not in evoked.ch_names:
            continue
        idx = evoked.ch_names.index(ch)
        if n400_mask.any():
            trace = data_uv[idx][n400_mask]
            peak_idx = np.argmin(trace)
            peak_t = times_sec[n400_mask][peak_idx]
            peak_times.append(peak_t)
            labels.append(f'{ch} peak: {peak_t*1000:.0f} ms')

    if not peak_times:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, f'{title}: peak topomaps unavailable',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    try:
        fig = evoked.plot_topomap(
            times=peak_times, ch_type='eeg', average=None,
            colorbar=True, show=False, time_unit='s')
        return fig
    except Exception:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, f'{title}: peak topomaps unavailable',
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig


# ── Main Pipeline ────────────────────────────────────────────────────────────

def process_single_file(vhdr_path, output_dir, log_dir, logger,
                        epoch_reject_uv=EPOCH_REJECT_DEFAULT,
                        detrend=False, uniform_scale=False,
                        l_freq=FILTER_L_FREQ, h_freq=PREPROCESS_H_FREQ):
    """Run the full N400 processing pipeline on a single recording."""
    patient_id = extract_patient_id(vhdr_path)
    visit_info = extract_visit_info(vhdr_path)
    script_dir = Path(__file__).resolve().parent
    patient_info = lookup_patient_info(patient_id, script_dir)
    logger.info(f'Processing {patient_id} / {visit_info}')
    logger.info(f'Input: {vhdr_path}')

    # ── 1. Load raw data ─────────────────────────────────────────────────
    logger.info('Loading raw data...')
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True,
                                       verbose='WARNING')
    logger.info(f'Loaded: {len(raw.ch_names)} channels, '
                f'{raw.info["sfreq"]} Hz, {raw.times[-1]:.1f} s')

    # ── 2. Prepare channels & montage ────────────────────────────────────
    raw = prepare_channels(raw, logger)

    # ── 3. Detect and interpolate bad channels ─────────────────────────
    logger.info('Detecting bad channels...')
    bad_chs = detect_bad_channels(raw, logger)
    if bad_chs:
        raw.info['bads'] = bad_chs
        logger.info(f'Interpolating {len(bad_chs)} bad channel(s): {bad_chs}')
        raw.interpolate_bads(verbose='WARNING')
    else:
        logger.info('No bad channels detected')

    # ── 4. Preprocessing bandpass filter ────────────────────────────────
    logger.info(f'Preprocessing filter: {l_freq}–{h_freq} Hz')
    raw.filter(l_freq, h_freq, verbose='WARNING')

    # ── 5. Resample ──────────────────────────────────────────────────────
    orig_sfreq = raw.info['sfreq']
    logger.info(f'Resampling: {orig_sfreq} Hz -> {RESAMPLE_FREQ} Hz')
    raw.resample(RESAMPLE_FREQ, verbose='WARNING')

    # ── 6. ICA artifact removal ──────────────────────────────────────────
    logger.info('Fitting ICA...')
    raw_before_ica = raw.copy()

    ica = ICA(n_components=ICA_N_COMPONENTS, method=ICA_METHOD,
              max_iter=ICA_MAX_ITER, random_state=42)
    ica.fit(raw, verbose='WARNING')
    logger.info(f'ICA fitted: {ica.n_components_} components')

    # Auto-detect EOG artifacts using multiple proxy channels
    eog_proxies = []
    if 'EOG' in raw.ch_names:
        eog_proxies.append('EOG')
    if 'Fp1' in raw.ch_names:
        eog_proxies.append('Fp1')
    if 'Fp2' in raw.ch_names:
        eog_proxies.append('Fp2')

    all_eog_indices = set()
    eog_scores_list = []  # one score array per proxy channel for report
    if eog_proxies:
        logger.info(f'EOG proxy channels: {eog_proxies}')
        for ch in eog_proxies:
            # Clear labels from previous call so find_bads_eog adds fresh ones
            ica.labels_ = {k: v for k, v in ica.labels_.items()
                           if not k.startswith('eog/')}
            indices, scores = ica.find_bads_eog(raw, ch_name=ch,
                                                threshold=1.5,
                                                verbose='WARNING')
            if not indices:
                # Clear again before retry with lower threshold
                ica.labels_ = {k: v for k, v in ica.labels_.items()
                               if not k.startswith('eog/')}
                indices, scores = ica.find_bads_eog(raw, ch_name=ch,
                                                    threshold=1.0,
                                                    verbose='WARNING')
            if indices:
                logger.info(f'  EOG via {ch}: components {indices}')
            eog_scores_list.append(scores)
            all_eog_indices.update(indices)

        # Use the score array from the primary proxy for the report
        eog_scores = eog_scores_list[0] if eog_scores_list else None

        eog_indices = sorted(all_eog_indices)
        # Cap at 4 to prevent over-rejection of EOG
        if len(eog_indices) > 4:
            logger.warning(f'Too many EOG components ({len(eog_indices)}), '
                           f'keeping top 4 by score')
            if eog_scores is not None:
                scored = [(abs(eog_scores[i]), i) for i in eog_indices
                          if i < len(eog_scores)]
                scored.sort(reverse=True)
                eog_indices = sorted([i for _, i in scored[:4]])
            else:
                eog_indices = eog_indices[:4]

        if eog_indices:
            logger.info(f'EOG components to exclude: {eog_indices}')
        else:
            logger.warning('No EOG components found')
    else:
        eog_indices = []
        eog_scores = None
        logger.warning('No EOG proxy available — skipping EOG detection')

    # Auto-detect muscle artifacts
    muscle_indices = []
    try:
        muscle_idx, muscle_scores = ica.find_bads_muscle(
            raw, verbose='WARNING')
        if muscle_idx:
            # Cap at 2 muscle components to avoid over-rejection
            if len(muscle_idx) > 2:
                scored_m = [(abs(muscle_scores[i]), i) for i in muscle_idx
                            if i < len(muscle_scores)]
                scored_m.sort(reverse=True)
                muscle_idx = sorted([i for _, i in scored_m[:2]])
            muscle_indices = [i for i in muscle_idx if i not in eog_indices]
            if muscle_indices:
                logger.info(f'Muscle components to exclude: {muscle_indices}')
    except Exception as e:
        logger.debug(f'Muscle artifact detection skipped: {e}')

    # Combine all artifact components
    all_artifact_indices = sorted(set(eog_indices) | set(muscle_indices))
    ica.exclude = all_artifact_indices
    ica.labels_ = {k: v for k, v in ica.labels_.items()
                   if not k.startswith('eog/') and not k.startswith('muscle/')}
    if eog_indices:
        ica.labels_['eog'] = list(eog_indices)
    if muscle_indices:
        ica.labels_['muscle'] = list(muscle_indices)
    if all_artifact_indices:
        logger.info(f'Total ICA components excluded: {all_artifact_indices} '
                     f'(EOG: {eog_indices}, muscle: {muscle_indices})')
    else:
        logger.warning('No artifact components found — ICA applied without '
                       'exclusions')

    ica.apply(raw, verbose='WARNING')
    logger.info('ICA applied')

    # ── 7a. Average re-reference (per spec: "перереферирование среднее") ─
    logger.info('Re-referencing to average...')
    raw.set_eeg_reference('average', verbose='WARNING')

    # ── 7b. N400 analysis lowpass filter (40 Hz per Table 8.1) ──────────
    logger.info(f'N400 analysis filter: lowpass {N400_H_FREQ} Hz')
    raw.filter(None, N400_H_FREQ, verbose='WARNING')

    # ── 8. Extract events & deduplicate ──────────────────────────────────
    logger.info('Extracting events...')
    events, all_event_id = mne.events_from_annotations(raw, verbose='WARNING')
    logger.info(f'Raw events extracted: {len(events)}')

    # Log ALL annotation keys found so we can debug marker issues
    logger.info(f'Annotation keys found ({len(all_event_id)}):')
    for annot_key, annot_val in all_event_id.items():
        count = np.sum(events[:, 2] == annot_val)
        if count > 0:
            logger.info(f'  {annot_key} (id={annot_val}): {count} events')

    # Deduplicate events (handles 500 Hz duplicate marker issue)
    events = deduplicate_events(events, raw.info['sfreq'], logger)

    # Build event ID mapping for test conditions
    event_id = build_event_id(all_event_id)
    logger.info(f'Test event mapping: {event_id}')

    if not event_id:
        logger.warning('No target stimulus events (s134-s139) found!')
        logger.warning('This file may not be an N400 (Picture-match) recording '
                       '— possibly MMNs or another paradigm.')
        logger.warning('Available annotations: %s', list(all_event_id.keys()))
        logger.warning('Total events: %d (N400 typically has ~340)',
                       len(events))
        logger.warning('SKIPPING this file.')
        return None

    # Log test event counts after dedup
    for cond, eid in event_id.items():
        count = np.sum(events[:, 2] == eid)
        logger.info(f'  {cond}: {count} events')

    # ── 9. Assign training conditions ────────────────────────────────────
    logger.info('Assigning training conditions...')
    events, training_event_id = assign_training_conditions(
        events, all_event_id, raw.info['sfreq'], logger)
    event_id.update(training_event_id)

    # ── 10. Create epochs ────────────────────────────────────────────────
    logger.info('Creating epochs...')
    if epoch_reject_uv is not None:
        epoch_reject = dict(eeg=epoch_reject_uv * 1e-6)
        logger.info(f'  Epoch rejection threshold: {epoch_reject_uv} uV')
    else:
        epoch_reject = None
        logger.info('  Epoch rejection: DISABLED')

    detrend_val = 1 if detrend else None
    if detrend:
        logger.info('  Linear detrending enabled')
    epochs = mne.Epochs(raw, events, event_id,
                        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                        baseline=BASELINE, preload=True,
                        reject=epoch_reject, detrend=detrend_val,
                        verbose='WARNING')

    n_total = sum(1 for d in epochs.drop_log if not any('IGNORED' in s for s in d))
    n_dropped = n_total - len(epochs)
    if n_dropped:
        logger.info(f'  Dropped {n_dropped}/{n_total} epochs by amplitude '
                     f'rejection (threshold: {epoch_reject_uv} uV)')
    else:
        logger.info(f'  No epochs dropped ({n_total} total)')

    for cond in event_id:
        n = len(epochs[cond])
        logger.info(f'  {cond}: {n} epochs')
    logger.info(f'  Total: {len(epochs)} epochs')

    # ── 11. Compute ERPs ────────────────────────────────────────────────
    logger.info('Computing evoked responses...')
    evokeds = {}

    for cond in ALL_CONDITIONS:
        if cond in event_id and len(epochs[cond]) > 0:
            evokeds[cond] = epochs[cond].average()
            evokeds[cond].comment = cond

    # ── 12. Compute N400 difference waves ──────────────────────────────
    logger.info('Computing N400 difference waves...')
    n400_diffs = {}

    # Within-block: real vs pseudo test (incongruent real - incongruent pseudo)
    diff_pairs = [
        ('BTR', 'BTP', 'n400_test_block1'),
        ('BBTR', 'BBTP', 'n400_test_block2'),
        ('BBBTR', 'BBBTP', 'n400_test_block3'),
    ]
    for real_cond, pseudo_cond, diff_name in diff_pairs:
        if real_cond in evokeds and pseudo_cond in evokeds:
            n400_diffs[diff_name] = mne.combine_evoked(
                [evokeds[real_cond], evokeds[pseudo_cond]], weights=[1, -1])
            n400_diffs[diff_name].comment = diff_name

    # ── Compute uniform y-axis limits if requested ────────────────────
    ylim = None
    if uniform_scale:
        ylim = compute_uniform_ylim(evokeds, N400_ROI_CHS)
        if ylim:
            logger.info(f'Uniform y-axis: {ylim[0]:.1f} to {ylim[1]:.1f} uV')

    # ── 13. Generate report ────────────────────────────────────────────
    logger.info('Generating report...')
    report = mne.Report(title=f'N400 report: {patient_id}', verbose='WARNING')

    # Section 1: ICA
    report.add_ica(
        ica=ica,
        title='ICA and automatic artifact components',
        inst=raw_before_ica,
        eog_evoked=None,
        eog_scores=eog_scores,
        tags=('ica', 'qc'),
        n_jobs=1,
    )

    # Section 2: Epochs after rejection
    if len(epochs) > 0:
        report.add_epochs(
            epochs=epochs,
            title='Epochs after rejection',
            tags=('epochs', 'qc'),
            psd=True,
        )
    else:
        logger.warning('All epochs rejected — skipping epochs report section')

    # Section 3: Test evoked responses (custom figures with N400 window)
    for cond in TEST_CONDITIONS:
        if cond in evokeds:
            fig = plot_erp_with_n400_window(evokeds[cond], cond,
                                            roi_chs=N400_ROI_CHS, ylim=ylim)
            report.add_figure(fig, title=cond,
                              tags=('evoked', 'n400', 'test'))
            plt.close(fig)

    # Section 3a2: ROI channel overlay per test condition
    for cond in TEST_CONDITIONS:
        if cond in evokeds:
            fig = plot_roi_channel_overlay(evokeds[cond], N400_ROI_CHS, cond,
                                          ylim=ylim)
            report.add_figure(fig, title=f'{cond} — ROI channels',
                              tags=('evoked', 'n400', 'roi'))
            plt.close(fig)

    # Section 3b: Topographic maps per test condition
    for cond in TEST_CONDITIONS:
        if cond in evokeds:
            fig = plot_condition_topomaps(evokeds[cond], cond)
            report.add_figure(fig, title=f'{cond} — Topography',
                              tags=('topomap', 'n400', 'test'))
            plt.close(fig)

    # Section 3b2: Averaged topography around 400-500ms per test condition
    for cond in TEST_CONDITIONS:
        if cond in evokeds:
            try:
                fig = evokeds[cond].plot_topomap(
                    times=[0.45], ch_type='eeg', average=0.1,
                    colorbar=True, show=False, time_unit='s')
                w, h = fig.get_size_inches()
                fig.set_size_inches(w + 1.5, h + 0.8)
                # Move colorbar away from topomap head
                for ax in fig.get_axes():
                    if ax.get_label() == '<colorbar>':
                        pos = ax.get_position()
                        ax.set_position([pos.x0 + 0.08, pos.y0,
                                         pos.width, pos.height])
                fig.subplots_adjust(top=0.82)
            except Exception:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.text(0.5, 0.5, f'{cond}: avg topography unavailable',
                        ha='center', va='center', fontsize=10)
                ax.axis('off')
            report.add_figure(fig, title=f'{cond} — Avg topo 400-500ms',
                              tags=('topomap', 'n400', 'averaged'))
            plt.close(fig)

    # Section 3c: Per-electrode ERP grids per test condition
    for cond in TEST_CONDITIONS:
        if cond in evokeds:
            fig = plot_per_electrode_erps(evokeds[cond], N400_ROI_CHS, cond,
                                         ylim=ylim)
            report.add_figure(fig, title=f'{cond} — Per-electrode',
                              tags=('evoked', 'n400', 'electrodes'))
            plt.close(fig)

    # Section 3d: Peak topomaps (Fz, Cz, Pz) per test condition
    for cond in TEST_CONDITIONS:
        if cond in evokeds:
            fig = plot_peak_topomaps(evokeds[cond], cond)
            report.add_figure(fig, title=f'{cond} — Peak topography',
                              tags=('topomap', 'n400', 'electrodes'))
            plt.close(fig)

    # Section 4: N400 difference waves (custom figures)
    for key in ('n400_test_block1', 'n400_test_block2', 'n400_test_block3'):
        if key in n400_diffs:
            fig = plot_diff_wave(n400_diffs[key], key,
                                 roi_chs=N400_ROI_CHS, ylim=ylim)
            report.add_figure(fig, title=key,
                              tags=('evoked', 'n400', 'difference'))
            plt.close(fig)

    # Section 4b: Best N400 channel summary
    fig, best_chs = plot_best_n400_summary(evokeds, TEST_CONDITIONS, patient_id)
    report.add_figure(fig, title='Best N400 channel per condition',
                      tags=('n400', 'best-channel', 'summary'))
    plt.close(fig)
    if best_chs:
        logger.info('Best N400 channels:')
        for cond, (ch, amp, lat) in best_chs.items():
            logger.info(f'  {cond}: {ch} ({amp:.1f} uV @ {lat:.0f} ms)')

    # Section 5: Training evoked responses (custom figures with N400 window)
    for cond in TRAINING_CONDITIONS:
        if cond in evokeds:
            fig = plot_erp_with_n400_window(evokeds[cond], cond,
                                            roi_chs=N400_ROI_CHS, ylim=ylim)
            report.add_figure(fig, title=cond,
                              tags=('evoked', 'training'))
            plt.close(fig)

    # Save report
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f'{patient_id}_{visit_info}_individual_report.html'
    output_path = output_dir / output_filename
    report.save(str(output_path), overwrite=True, open_browser=False,
                verbose='WARNING')
    logger.info(f'Report saved: {output_path}')

    # ── 14. Export per-trial N400 table and evokeds ────────────────────
    export_per_trial_n400(epochs, patient_id, output_dir, logger)

    # Save evokeds as .fif for group analysis
    all_evokeds = list(evokeds.values()) + list(n400_diffs.values())
    if all_evokeds:
        fif_path = output_dir / f'{patient_id}_{visit_info}-ave.fif'
        mne.write_evokeds(str(fif_path), all_evokeds, overwrite=True,
                          verbose='WARNING')
        logger.info(f'Evokeds saved: {fif_path}')

    # ── 15. Validate results ───────────────────────────────────────────
    fails, warnings = validate_results(patient_id, epochs, evokeds,
                                        n400_diffs, ica, eog_indices,
                                        n_dropped, bad_chs, patient_info,
                                        logger)

    # QC classification
    n_total = len(epochs) + n_dropped
    drop_pct = (n_dropped / n_total * 100) if n_total > 0 else 100
    if fails > 0 or drop_pct > 15:
        qc_status = 'noisy'
    else:
        qc_status = 'clean'
    logger.info(f'QC status: {qc_status} (fails={fails}, warnings={warnings}, '
                f'drop={drop_pct:.1f}%)')

    return {
        'report_path': output_path,
        'qc_status': qc_status,
        'drop_pct': drop_pct,
        'n_epochs': len(epochs),
        'n_dropped': n_dropped,
        'fails': fails,
        'warnings': warnings,
    }

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='N400 (Picture-match) EEG Processing Pipeline')
    parser.add_argument('--input', required=True,
                        help='Path to patient folder or .vhdr file')
    parser.add_argument('--output', required=True,
                        help='Output directory for reports')
    parser.add_argument('--reject', default=str(EPOCH_REJECT_DEFAULT),
                        help='Epoch rejection threshold in uV, or "none" to '
                             'disable (default: %(default)s)')
    parser.add_argument('--l-freq', type=float, default=FILTER_L_FREQ,
                        help='High-pass filter cutoff in Hz '
                             f'(default: {FILTER_L_FREQ})')
    parser.add_argument('--h-freq', type=float, default=PREPROCESS_H_FREQ,
                        help='Low-pass filter cutoff in Hz '
                             f'(default: {PREPROCESS_H_FREQ})')
    parser.add_argument('--detrend', action='store_true',
                        help='Apply linear detrending to epochs')
    parser.add_argument('--uniform-scale', action='store_true',
                        help='Use the same y-axis scale across all ERP plots')
    parser.add_argument('--parallel', type=int, default=1, metavar='N',
                        help='Process N patients in parallel (default: 1 = '
                             'sequential)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose MNE output')
    return parser.parse_args()


def _collect_result(status, pid, payload,
                    clean_patients, noisy_patients, skipped, failures):
    """Collect a single processing result into the appropriate list."""
    if status == 'skip':
        print(f'  SKIP {pid}: no N400 markers (possibly MMNs data)')
        skipped.append((pid, payload))
    elif status == 'clean':
        print(f'  CLEAN {pid}: drop={payload["drop_pct"]:.1f}%, '
              f'epochs={payload["n_epochs"]}')
        clean_patients.append((pid, payload))
    elif status == 'noisy':
        print(f'  NOISY {pid}: drop={payload["drop_pct"]:.1f}%, '
              f'epochs={payload["n_epochs"]}, fails={payload["fails"]}')
        noisy_patients.append((pid, payload))
    else:
        print(f'  FAIL {pid}: {payload}')
        failures.append((pid, payload))


def _process_one(vhdr_path, output_dir, log_dir, epoch_reject_uv,
                  detrend, uniform_scale, l_freq, h_freq, verbose):
    """Worker function for processing a single patient.

    Designed to run in a separate process via ProcessPoolExecutor.
    Returns (status, patient_id, payload) where status is one of
    'clean', 'noisy', 'skip', 'fail'.
    """
    # Each spawned process re-imports the module, so matplotlib backend
    # and MNE log level are already configured.  Re-set log level in case
    # this is called inside a new process where the main-guard hasn't run.
    if not verbose:
        mne.set_log_level('WARNING')

    patient_id = extract_patient_id(vhdr_path)
    logger = setup_logging(log_dir, patient_id)

    try:
        result = process_single_file(vhdr_path, output_dir, log_dir, logger,
                                     epoch_reject_uv=epoch_reject_uv,
                                     detrend=detrend,
                                     uniform_scale=uniform_scale,
                                     l_freq=l_freq, h_freq=h_freq)
        if result is None:
            return ('skip', patient_id, str(vhdr_path))
        elif result['qc_status'] == 'clean':
            return ('clean', patient_id, result)
        else:
            return ('noisy', patient_id, result)
    except Exception as e:
        logger.exception(f'Failed to process {vhdr_path}')
        return ('fail', patient_id, str(e))


def main():
    args = parse_args()

    if not args.verbose:
        mne.set_log_level('WARNING')

    # Parse rejection threshold
    if args.reject.lower() in ('none', 'off', 'false', '0'):
        epoch_reject_uv = None
    else:
        try:
            epoch_reject_uv = float(args.reject)
        except ValueError:
            print(f'Error: --reject must be a number (uV) or "none", '
                  f'got "{args.reject}"')
            sys.exit(1)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    # Discover files
    vhdr_files = discover_vhdr_files(input_path)
    print(f'Found {len(vhdr_files)} .vhdr file(s)')

    # Resolve log directory relative to script location
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / 'logs'

    clean_patients = []
    noisy_patients = []
    skipped = []
    failures = []

    n_workers = max(1, args.parallel)

    if n_workers == 1:
        # Sequential mode (original behaviour)
        for vhdr_path in vhdr_files:
            status, pid, payload = _process_one(
                vhdr_path, output_dir, log_dir, epoch_reject_uv,
                args.detrend, args.uniform_scale, args.l_freq, args.h_freq,
                args.verbose)
            _collect_result(status, pid, payload,
                            clean_patients, noisy_patients, skipped, failures)
    else:
        # Parallel mode
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f'Running with {n_workers} parallel workers')
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for vhdr_path in vhdr_files:
                fut = pool.submit(
                    _process_one, vhdr_path, output_dir, log_dir,
                    epoch_reject_uv, args.detrend, args.uniform_scale,
                    args.l_freq, args.h_freq, args.verbose)
                futures[fut] = vhdr_path
            for fut in as_completed(futures):
                vhdr_path = futures[fut]
                try:
                    status, pid, payload = fut.result()
                except Exception as e:
                    pid = extract_patient_id(vhdr_path)
                    status, payload = 'fail', str(e)
                _collect_result(status, pid, payload,
                                clean_patients, noisy_patients, skipped,
                                failures)

    # Summary
    n_ok = len(clean_patients) + len(noisy_patients)
    print(f'\nDone: {n_ok} processed ({len(clean_patients)} clean, '
          f'{len(noisy_patients)} noisy), '
          f'{len(skipped)} skipped, {len(failures)} failed')

    print('\n--- CLEAN patients ---')
    for pid, res in clean_patients:
        print(f'  {pid}: drop={res["drop_pct"]:.1f}%, '
              f'epochs={res["n_epochs"]}, report={res["report_path"]}')
    print(f'\n--- NOISY patients (>15% dropped or validation fails) ---')
    for pid, res in noisy_patients:
        print(f'  {pid}: drop={res["drop_pct"]:.1f}%, '
              f'epochs={res["n_epochs"]}, fails={res["fails"]}, '
              f'warnings={res["warnings"]}')
    if skipped:
        print('\n--- SKIPPED ---')
        for pid, path in skipped:
            print(f'  {pid}: {path}')
    if failures:
        print('\n--- FAILED ---')
        for pid, path in failures:
            print(f'  {pid}: {path}')

    # Write QC summary file
    qc_path = output_dir / 'qc_summary.txt'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(str(qc_path), 'w', encoding='utf-8') as f:
        f.write(f'QC Summary — {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
        f.write(f'Filter: {args.l_freq}-{args.h_freq} Hz, '
                f'reject: {args.reject} uV, '
                f'detrend: {args.detrend}\n')
        f.write(f'Total: {n_ok} processed, {len(skipped)} skipped, '
                f'{len(failures)} failed\n\n')
        f.write(f'CLEAN ({len(clean_patients)}):\n')
        for pid, res in sorted(clean_patients):
            f.write(f'  {pid}  drop={res["drop_pct"]:.1f}%  '
                    f'epochs={res["n_epochs"]}  '
                    f'warnings={res["warnings"]}\n')
        f.write(f'\nNOISY ({len(noisy_patients)}):\n')
        for pid, res in sorted(noisy_patients):
            f.write(f'  {pid}  drop={res["drop_pct"]:.1f}%  '
                    f'epochs={res["n_epochs"]}  '
                    f'fails={res["fails"]}  warnings={res["warnings"]}\n')
        if skipped:
            f.write(f'\nSKIPPED ({len(skipped)}):\n')
            for pid, path in sorted(skipped):
                f.write(f'  {pid}  {path}\n')
        if failures:
            f.write(f'\nFAILED ({len(failures)}):\n')
            for pid, path in sorted(failures):
                f.write(f'  {pid}  {path}\n')
    print(f'\nQC summary saved: {qc_path}')


if __name__ == '__main__':
    main()
