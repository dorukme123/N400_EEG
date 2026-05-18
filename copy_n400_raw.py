"""Copy N400 raw EEG data from network drive to local inputs/ directory.

Reads the mapping file (z_drive_n400_raw_mapping.txt) to know exactly which
files exist for each patient, then copies them to the local inputs/ folder
preserving the expected directory structure.

The Z: drive (or --server path on Linux) is accessed READ-ONLY.

Usage:
    python copy_n400_raw.py --dry-run                     # preview only
    python copy_n400_raw.py                               # copy all mapped patients
    python copy_n400_raw.py --patients INP0019 INP0036    # copy specific patients
    python copy_n400_raw.py --server /mnt/z/Academic_perfomance/Data/EEG_behav_data
"""

import argparse
import io
import re
import shutil
import sys
from pathlib import Path

if sys.platform == 'win32':
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                      errors='replace')

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
MAPPING_FILE = SCRIPT_DIR / 'z_drive_n400_raw_mapping.txt'
DEFAULT_BASE = Path('Z:/Academic_perfomance/Data/EEG_behav_data')
DEFAULT_OUTPUT = SCRIPT_DIR / 'inputs'

# Target patient list (from reviewer, Комментарии.docx, 2026-05-18)
HEALTHY = {
    'INP0008', 'INP0019', 'INP0036', 'INP0037', 'INP0055', 'INP0064',
    'INP0086', 'INP0089', 'INP0092', 'INP0094', 'INP0096', 'INP0101',
    'INP0102', 'INP0103', 'INP0104', 'INP0105', 'INP0106', 'INP0107',
    'INP0110', 'INP0117', 'INP0125', 'INP0126', 'INP0129', 'INP0131',
    'INP0136', 'INP0138', 'INP0140', 'INP0144', 'INP0145', 'INP0146',
    'INP0149', 'INP0150', 'INP0151', 'INP0152', 'INP0153', 'INP0154',
    'INP0155', 'INP0156', 'INP0161', 'INP0163', 'INP0164', 'INP0165',
    'INP0172', 'INP0173', 'INP0174', 'INP0175', 'INP0180', 'INP0185',
    'INP0188', 'INP0189', 'INP0190', 'INP0196', 'INP0198', 'INP0200',
}
SPEECH_DISORDER = {
    'INP0014', 'INP0057', 'INP0076', 'INP0093', 'INP0100', 'INP0109',
    'INP0112', 'INP0113', 'INP0116', 'INP0118', 'INP0123', 'INP0127',
    'INP0128', 'INP0148', 'INP0160', 'INP0166', 'INP0168', 'INP0177',
    'INP0186',
}
ALL_PATIENTS = HEALTHY | SPEECH_DISORDER


# ── Mapping Parser ───────────────────────────────────────────────────────────

def parse_mapping(mapping_path):
    """Parse z_drive_n400_raw_mapping.txt into structured entries.

    Returns list of dicts:
      {'patient': 'INP0019', 'visit': 'посещение 5', 'subpath': 'N400/Raw/',
       'files': ['file1.eeg', 'file1.vhdr', 'file1.vmrk']}
    """
    entries = []
    current = None

    with open(mapping_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            # Entry header: [INP0019] посещение 5/N400/Raw/
            m = re.match(r'^\[(\S+)\]\s+(.+)$', line)
            if m:
                if current and current['files']:
                    entries.append(current)
                patient = m.group(1)
                rel_path = m.group(2).strip()
                # Extract visit folder (everything before /N400)
                parts = rel_path.split('/N400')
                visit = parts[0] if parts else rel_path
                subpath = 'N400' + parts[1] if len(parts) > 1 else 'N400/Raw/'
                # Clean up subpath annotations
                subpath = re.sub(r'\s*\(.*\)', '', subpath).rstrip('/')
                current = {
                    'patient': patient,
                    'visit': visit,
                    'subpath': subpath,
                    'files': [],
                }
                continue
            # File line:   filename.ext
            if current and line.startswith('  ') and line.strip():
                fname = line.strip()
                if not fname.startswith('('):  # skip annotations like (EMPTY)
                    current['files'].append(fname)

    if current and current['files']:
        entries.append(current)

    return entries


def extract_base_patient_id(patient_str):
    """Extract base INP#### ID from strings like 'INP0089_RNS004'."""
    m = re.match(r'(INP\d+)', patient_str)
    return m.group(1) if m else patient_str


# ── Copy Logic ───────────────────────────────────────────────────────────────

def copy_patient_files(entries, base_path, output_dir, target_patients,
                       dry_run=False):
    """Copy raw files for target patients from network drive to local dir."""
    copied = 0
    skipped = 0
    errors = 0
    not_found = set()

    # Build lookup: base_patient_id -> list of entries
    patient_entries = {}
    for entry in entries:
        base_id = extract_base_patient_id(entry['patient'])
        patient_entries.setdefault(base_id, []).append(entry)

    for patient_id in sorted(target_patients):
        if patient_id not in patient_entries:
            print(f'  MISS {patient_id}: not in mapping file')
            not_found.add(patient_id)
            continue

        p_entries = patient_entries[patient_id]
        # If multiple visits, use the first one (primary visit 5)
        entry = p_entries[0]

        # Source: base_path / patient_folder / visit / N400/Raw/
        src_dir = base_path / entry['patient'] / entry['visit'] / entry['subpath']
        # Destination: output_dir / patient_id / visit / N400/Raw/
        dst_dir = output_dir / patient_id / entry['visit'] / 'N400' / 'Raw'

        for fname in entry['files']:
            # Only copy EEG triplet files
            if not any(fname.endswith(ext) for ext in ('.eeg', '.vhdr', '.vmrk')):
                continue
            src_file = src_dir / fname
            dst_file = dst_dir / fname

            if dst_file.exists():
                print(f'  SKIP {patient_id}/{fname}: already exists')
                skipped += 1
                continue

            if dry_run:
                print(f'  COPY {patient_id}/{fname}')
                print(f'       {src_file} -> {dst_file}')
                copied += 1
                continue

            try:
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_file), str(dst_file))
                print(f'  OK   {patient_id}/{fname}')
                copied += 1
            except Exception as e:
                print(f'  FAIL {patient_id}/{fname}: {e}')
                errors += 1

    return copied, skipped, errors, not_found


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Copy N400 raw EEG data from network drive to local inputs/')
    parser.add_argument('--server', type=str, default=None,
                        help='Override base path (e.g., /mnt/z/Academic_perfomance'
                             '/Data/EEG_behav_data for Linux)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: inputs/)')
    parser.add_argument('--patients', nargs='+', default=None,
                        help='Specific patient IDs to copy (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview copies without writing')
    parser.add_argument('--mapping', type=str, default=None,
                        help='Path to mapping file (default: z_drive_n400_raw'
                             '_mapping.txt)')
    return parser.parse_args()


def main():
    args = parse_args()

    mapping_path = Path(args.mapping) if args.mapping else MAPPING_FILE
    if not mapping_path.exists():
        print(f'ERROR: Mapping file not found: {mapping_path}')
        print('Run the Z: drive scan first to create this file.')
        sys.exit(1)

    base_path = Path(args.server) if args.server else DEFAULT_BASE
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT

    if args.patients:
        target = set(args.patients)
    else:
        target = ALL_PATIENTS

    print(f'Mapping file: {mapping_path}')
    print(f'Source:        {base_path}')
    print(f'Destination:   {output_dir}')
    print(f'Patients:      {len(target)}')
    if args.dry_run:
        print('MODE:          DRY RUN (no files will be copied)')
    print()

    entries = parse_mapping(mapping_path)
    print(f'Parsed {len(entries)} entries from mapping file')
    print()

    copied, skipped, errors, not_found = copy_patient_files(
        entries, base_path, output_dir, target, dry_run=args.dry_run)

    print(f'\nDone: {copied} copied, {skipped} skipped, {errors} errors')
    if not_found:
        print(f'Missing from mapping: {sorted(not_found)}')


if __name__ == '__main__':
    main()
