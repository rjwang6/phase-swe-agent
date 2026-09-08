import json
from pathlib import Path
from collections import defaultdict

LOG_DIR = Path("token_stats")
PHASE_NAMES = ["exploration", "execution", "validation"]

# Overall stats
all_phase_lengths = []
phase_type_lengths = {name: [] for name in PHASE_NAMES}

# Per-occurrence stats:
# e.g. occurrence_lengths["exploration"][0] = list of all 1st exploration lengths
#      occurrence_lengths["exploration"][1] = list of all 2nd exploration lengths
occurrence_lengths = {
    name: defaultdict(list) for name in PHASE_NAMES
}

for path in LOG_DIR.iterdir():
    if not path.is_file():
        continue

    with open(path) as f:
        data = json.load(f)

    phase_lengths = data["phase_lengths"]

    # Track how many times we've seen each phase type in this file
    occurrence_index_within_file = {
        "exploration": 0,
        "execution": 0,
        "validation": 0,
    }

    for i, length in enumerate(phase_lengths):
        phase_name = PHASE_NAMES[i % 3]

        # Overall average over all phases
        all_phase_lengths.append(length)

        # Average by phase type
        phase_type_lengths[phase_name].append(length)

        # Average by occurrence number within phase type
        occ_idx = occurrence_index_within_file[phase_name]
        occurrence_lengths[phase_name][occ_idx].append(length)
        occurrence_index_within_file[phase_name] += 1

# 1. Overall average of all phase lengths
overall_avg = sum(all_phase_lengths) / len(all_phase_lengths)
print(f"Overall average phase length: {overall_avg:.4f}")
print()

# 2. Average length of each phase type
for phase in PHASE_NAMES:
    vals = phase_type_lengths[phase]
    avg = sum(vals) / len(vals)
    print(f"Average {phase} phase length: {avg:.4f} (n={len(vals)})")

print()

# 3. Average length of 1st exploration, 2nd exploration, etc.
#    and similarly for execution/validation
for phase in PHASE_NAMES:
    print(f"{phase.capitalize()} phase occurrence averages:")
    for occ_idx in sorted(occurrence_lengths[phase].keys()):
        vals = occurrence_lengths[phase][occ_idx]
        avg = sum(vals) / len(vals)
        print(f"  {occ_idx + 1} occurrence: {avg:.4f} (n={len(vals)})")
    print()