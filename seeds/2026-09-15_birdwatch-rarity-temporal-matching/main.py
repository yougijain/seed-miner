import numpy as np
import pandas as pd
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

np.random.seed(42)

# Synthetic birdwatch sighting log: (date, species, location_type)
# Rare species cluster in rare habitats; common species flood all habitats.
locations = ['meadow', 'forest', 'wetland', 'cliff']
species_list = [
    ('robin', 'common'),      # sights in meadow, forest, wetland
    ('sparrow', 'common'),    # sights in meadow, forest
    ('hawk', 'rare'),         # only in cliff, very few
    ('eagle', 'rare'),        # only in cliff, very few
    ('warbler', 'uncommon'),  # forest & wetland
    ('heron', 'uncommon'),    # wetland only
]

# Generate synthetic sighting records
records = []
for day in range(365):
    for sp, rarity in species_list:
        if rarity == 'common':
            if np.random.rand() < 0.8:
                loc = np.random.choice(['meadow', 'forest', 'wetland'])
                records.append({'date': day, 'species': sp, 'location': loc})
        elif rarity == 'uncommon':
            if np.random.rand() < 0.4:
                loc = np.random.choice(['forest', 'wetland'])
                records.append({'date': day, 'species': sp, 'location': loc})
        else:  # rare
            if np.random.rand() < 0.15:
                records.append({'date': day, 'species': sp, 'location': 'cliff'})

df = pd.DataFrame(records)

# Build temporal co-absence matrix: for each species pair, count days neither was sighted
species_set = sorted(df['species'].unique())
n_species = len(species_set)
n_days = 365

# For each species, which days had any sighting?
sighted_days = {}
for sp in species_set:
    sighted_days[sp] = set(df[df['species'] == sp]['date'].unique())

# Co-absence: days when BOTH species were absent
co_absence = np.zeros((n_species, n_species))
for i, sp1 in enumerate(species_set):
    for j, sp2 in enumerate(species_set):
        if i != j:
            absent1 = n_days - len(sighted_days[sp1])
            absent2 = n_days - len(sighted_days[sp2])
            joint_absent = n_days - len(sighted_days[sp1] | sighted_days[sp2])
            co_absence[i, j] = joint_absent

# Normalize: higher co-absence = higher similarity (both rare, same habitat avoidance)
co_absence_sim = co_absence / (co_absence.max() + 1)

print("=== Sighting Summary ===")
for sp in species_set:
    print(f"{sp:12} {len(sighted_days[sp]):3d} sightings / 365 days")

print("\n=== Co-Absence Similarity Matrix ===")
print(pd.DataFrame(co_absence_sim, index=species_set, columns=species_set).round(2))

# Bipartite matching: match species to their co-absence "pairs"
# Cost matrix: we want to *maximize* co-absence similarity, so negate
cost = -co_absence_sim

# Pad to square for Hungarian algorithm
row_ind, col_ind = linear_sum_assignment(cost)

print("\n=== Bipartite Matching (via Temporal Co-Absence) ===")
matches = []
for i, j in zip(row_ind, col_ind):
    if i < j:  # avoid duplicates
        matches.append((species_set[i], species_set[j], co_absence_sim[i, j]))

matches.sort(key=lambda x: x[2], reverse=True)
for sp1, sp2, score in matches:
    print(f"{sp1:12} <-> {sp2:12}  (co-absence score: {score:.3f})")

print("\n=== Interpretation ===")
print("High co-absence score = both species rare + confined to similar scarce habitats.")
print("This should pair (hawk, eagle) highly (both cliff-only rare).")
print("This inverts typical matching: signal is *absence*, not presence.")
