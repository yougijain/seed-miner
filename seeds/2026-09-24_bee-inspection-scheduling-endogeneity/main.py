import numpy as np
import pandas as pd
from itertools import combinations

np.random.seed(42)

# Synthetic data: 20 colonies, 12 weeks, each has latent disease type
# Beekeeper inspects based on observed health decline, creating endogenous feedback
num_colonies = 20
num_weeks = 12
max_inspections_per_week = 4

# Latent pathology types (unobserved)
pathology_types = {
    'varroa': {'decline_rate': 0.08, 'symptom_onset': 2},
    'nosema': {'decline_rate': 0.05, 'symptom_onset': 4},
    'healthy': {'decline_rate': 0.01, 'symptom_onset': 12}
}

# Assign each colony a pathology
np.random.seed(42)
colony_pathology = np.random.choice(['varroa', 'nosema', 'healthy'], size=num_colonies, p=[0.35, 0.35, 0.3])

# Simulate health trajectory and inspection schedule (endogenous)
health_data = []
inspection_schedule = []

for col_id in range(num_colonies):
    pathology = colony_pathology[col_id]
    params = pathology_types[pathology]
    health = 100.0
    last_inspection = -10  # Force early inspection
    
    for week in range(num_weeks):
        # Health declines by pathology rate
        health *= (1 - params['decline_rate'])
        health = max(health, 0)
        
        # Beekeeper inspects if:
        # 1. Health below threshold, OR
        # 2. Enough time since last inspection
        weeks_since_inspection = week - last_inspection
        inspection_threshold = 70 if params['symptom_onset'] <= week else 85
        
        will_inspect = (health < inspection_threshold) or (weeks_since_inspection >= 3)
        
        if will_inspect:
            inspection_schedule.append({'colony_id': col_id, 'week': week, 'pathology': pathology})
            last_inspection = week
        
        health_data.append({
            'colony_id': col_id,
            'week': week,
            'health': health,
            'pathology': pathology,
            'inspected': will_inspect
        })

df_health = pd.DataFrame(health_data)
df_inspections = pd.DataFrame(inspection_schedule)

print("\n=== COLONY HEALTH SIMULATION ===")
print(f"Total colonies: {num_colonies}")
print(f"Pathology distribution:")
for p in ['varroa', 'nosema', 'healthy']:
    count = (colony_pathology == p).sum()
    print(f"  {p}: {count} colonies")

print(f"\nTotal inspections logged: {len(df_inspections)}")
print(f"Avg inspections per colony: {len(df_inspections) / num_colonies:.1f}")

# ===== BIN-PACKING SCHEDULER: Assign inspections to weekly slots =====
# Constraint: max 4 inspections per week
# Objective: pack as many scheduled inspections as possible

def pack_inspections(df_insp, max_per_week):
    """Greedy bin-packing: assign inspections to weeks, respect capacity."""
    inspections_by_week = {w: [] for w in range(num_weeks)}
    unscheduled = []
    
    # Sort by week, then by colony (deterministic)
    df_sorted = df_insp.sort_values(['week', 'colony_id']).reset_index(drop=True)
    
    for _, row in df_sorted.iterrows():
        preferred_week = int(row['week'])
        
        # Try to fit in preferred week or nearby weeks
        scheduled = False
        for offset in [0, 1, -1, 2, -2]:
            candidate_week = preferred_week + offset
            if 0 <= candidate_week < num_weeks:
                if len(inspections_by_week[candidate_week]) < max_per_week:
                    inspections_by_week[candidate_week].append(row['colony_id'])
                    scheduled = True
                    break
        
        if not scheduled:
            unscheduled.append(row['colony_id'])
    
    return inspections_by_week, unscheduled

inspections_by_week, unscheduled = pack_inspections(df_inspections, max_inspections_per_week)

print(f"\n=== SCHEDULER FEASIBILITY ===")
print(f"Unscheduled inspections (due to capacity): {len(unscheduled)}")
print(f"Feasibility: {100 * (1 - len(unscheduled) / len(df_inspections)):.1f}%")

# ===== REVERSE-ENGINEER PATHOLOGY FROM CONSTRAINT VIOLATIONS =====
# Hypothesis: pathologies with high inspection demand are those causing frequent health dips
# Extract which colonies' inspections were most frequently unscheduled

unscheduled_counts = pd.Series(unscheduled).value_counts()
scheduled_colonies = set(range(num_colonies)) - set(unscheduled_counts.index)

print(f"\n=== PATHOLOGY INFERENCE FROM SCHEDULING PRESSURE ===")

# Measure inspection intensity by pathology
inspection_intensity = df_inspections.groupby('pathology').size() / df_inspections['pathology'].value_counts()
print(f"\nInspection intensity (inspections per colony, by pathology):")
for pathology in ['varroa', 'nosema', 'healthy']:
    count_colonies = (colony_pathology == pathology).sum()
    count_inspections = len(df_inspections[df_inspections['pathology'] == pathology])
    intensity = count_inspections / count_colonies if count_colonies > 0 else 0
    print(f"  {pathology}: {intensity:.2f} inspections/colony")

# Infer which pathologies are most problematic
print(f"\nColonies hardest to schedule (highest inspection demand):")
for col_id in unscheduled_counts.head(5).index:
    print(f"  Colony {col_id}: {colony_pathology[col_id]} ({unscheduled_counts[col_id]} unscheduled)")

print(f"\n=== INTERPRETATION ===")
print(f"The bin-packing scheduler's constraint violations reveal which colonies")
print(f"have pathologies with high inspection burden. Varroa (fast decline) should")
print(f"cluster among unschedulable colonies; nosema and healthy should be schedulable.")
print(f"\nThis forces optimization to work backward: scheduling pressure → inferred disease.")
