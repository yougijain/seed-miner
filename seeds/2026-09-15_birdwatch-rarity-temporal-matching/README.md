# Rare-Bird Sighting Matching via Temporal Scarcity Co-Occurrence

## Research Question

Can bipartite matching on temporal **co-absence** (days when two species are both unsighted) infer shared rarity class and habitat preference when direct co-sighting data is too sparse?

## Non-Obvious Angle

Traditional recommendation matching (collaborative filtering, bipartite graph matching) works on **co-occurrence**: items/entities that appear together are similar. Competitive birdwatching flips this: **rare species by definition are almost never co-sighted.** So the algorithm must reverse-engineer affinity from *joint absence* instead—two rare species that share habitat constraints will *both* be absent from common habitats on the same days, encoding their co-rarity in the temporal gap pattern.

This forces the matching algorithm to:
- Ignore presence (the sighted day) as a signal
- Treat absence (days with no sighting) as the primary feature
- Infer habitat affinity from *when species cluster in absence*, not when they cluster in presence

## Data

**Synthetic.** Generated a 365-day sighting log with:
- 6 species: 2 common (robin, sparrow), 2 uncommon (warbler, heron), 2 rare (hawk, eagle)
- Common species sighted ~80% of days, in multiple habitats (meadow, forest, wetland)
- Uncommon species sighted ~40% of days, in 2 habitats (forest, wetland)
- Rare species sighted ~15% of days, in one habitat only (cliff)

The structure is intentional: rare species will show high co-absence because both avoid common habitats.

## Limitation

With only 6 species, this is a toy proof-of-concept. Real birdwatch data (eBird, local clubs) would have hundreds of species and thousands of observers with location/date/count metadata, allowing validation that co-absence matching recovers actual biogeographic co-rarity. Here we're just showing that the *signal* exists in the right data structure.

## Expected Result

The bipartite matcher should pair (hawk, eagle) with high co-absence score, because both are rare and confined to cliff habitat—they're "together" in absence. Common species should score low with each other, because they're sighted frequently everywhere.

## Usage

```bash
python main.py
```

Outputs:
- Sighting counts per species
- Co-absence similarity matrix
- Matched species pairs ranked by co-absence score
- Interpretation note

---

*Auto-generated seed. Not a real birdwatching study.*
