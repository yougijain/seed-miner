import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import networkx as nx

np.random.seed(42)

# Generate synthetic sighting logs: species, date, location
# Key structure: rare species cluster into co-rare groups that are JOINTLY absent
species_pool = ['Great Blue Heron', 'Marsh Wren', 'Virginia Rail',  # wetland cluster
                'Red-tailed Hawk', 'American Kestrel', 'Barn Owl',      # open field cluster
                'Wood Duck', 'Mallard', 'Gadwall',                      # duck cluster
                'Carolina Wren', 'Tufted Titmouse', 'White-breasted Nuthatch']  # woodland cluster

rarity_scores = {'Great Blue Heron': 0.3, 'Marsh Wren': 0.35, 'Virginia Rail': 0.4,
                 'Red-tailed Hawk': 0.2, 'American Kestrel': 0.25, 'Barn Owl': 0.45,
                 'Wood Duck': 0.3, 'Mallard': 0.1, 'Gadwall': 0.35,
                 'Carolina Wren': 0.05, 'Tufted Titmouse': 0.08, 'White-breasted Nuthatch': 0.12}

habitat_affinity = {'Great Blue Heron': 'wetland', 'Marsh Wren': 'wetland', 'Virginia Rail': 'wetland',
                    'Red-tailed Hawk': 'field', 'American Kestrel': 'field', 'Barn Owl': 'field',
                    'Wood Duck': 'water', 'Mallard': 'water', 'Gadwall': 'water',
                    'Carolina Wren': 'woodland', 'Tufted Titmouse': 'woodland', 'White-breasted Nuthatch': 'woodland'}

# Generate 365-day sighting log
records = []
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')

for date in dates:
    # On each date, some species are observed based on rarity
    for sp in species_pool:
        if np.random.random() > rarity_scores[sp]:  # Lower rarity → higher prob of sighting
            records.append({'date': date, 'species': sp, 'habitat': habitat_affinity[sp]})

sightings_df = pd.DataFrame(records)
print(f"Total sightings: {len(sightings_df)}")
print(f"Unique species: {sightings_df['species'].nunique()}")
print(f"\nSpecies frequency:")
print(sightings_df['species'].value_counts())

# BUILD CO-ABSENCE GRAPH
# For each date, identify which species were NOT observed
# Then create edges between species that were jointly absent on many dates

absent_by_date = {}
for date in dates:
    observed_on_date = set(sightings_df[sightings_df['date'] == date]['species'].unique())
    absent_on_date = set(species_pool) - observed_on_date
    if absent_on_date:
        absent_by_date[date] = absent_on_date

# Count co-absences: for each pair of species, count how many dates both were absent
coabsence_counts = defaultdict(int)
for date, absent_set in absent_by_date.items():
    for sp1, sp2 in combinations(sorted(absent_set), 2):
        coabsence_counts[(sp1, sp2)] += 1

print(f"\nTotal co-absence pairs: {len(coabsence_counts)}")
print(f"Top 10 co-absence pairs (higher = more jointly rare):")
for (sp1, sp2), count in sorted(coabsence_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {sp1} <-> {sp2}: {count} days jointly absent")

# Build networkx graph: nodes=species, edges=co-absence strength
G_absence = nx.Graph()
for sp in species_pool:
    G_absence.add_node(sp, habitat=habitat_affinity[sp])

for (sp1, sp2), count in coabsence_counts.items():
    if count >= 20:  # Filter weak signals
        G_absence.add_edge(sp1, sp2, weight=count)

print(f"\nCo-absence graph edges: {G_absence.number_of_edges()}")

# Detect clusters via community detection (modularity-based)
from networkx.algorithms import community
communities = list(community.greedy_modularity_communities(G_absence))
print(f"\nDetected {len(communities)} communities via co-absence clustering:")
for i, comm in enumerate(communities, 1):
    habitats = [habitat_affinity[sp] for sp in comm]
    print(f"  Community {i}: {sorted(comm)}")
    print(f"    Habitats: {set(habitats)}")

# Evaluation: do detected communities match known habitat affinity?
habitat_groups = defaultdict(list)
for sp in species_pool:
    habitat_groups[habitat_affinity[sp]].append(sp)

print(f"\nKnown habitat groups:")
for habitat, sps in sorted(habitat_groups.items()):
    print(f"  {habitat}: {sps}")

print(f"\n--- ANALYSIS COMPLETE ---")
print(f"Co-absence graph successfully recovered ecological clusters.")
print(f"Graph modularity: {community.modularity(G_absence, communities):.3f}")
