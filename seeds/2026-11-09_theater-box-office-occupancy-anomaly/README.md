# Community Theater Box Office Anomaly Detection via Occupancy Pattern Inversion

## Research Question
Can anomaly detection identify hidden show failures (cast illness, technical collapse, word-of-mouth damage) by inverting the technique to detect *spatial clustering of occupied seats* rather than rare ticket transactions?

## The Non-Obvious Angle
Standard anomaly detection finds rare events or extreme values (e.g., unusually low ticket sales, outlier prices). This project inverts the technique: the anomaly is not *how many* seats are filled, but *where* they cluster.

**The insight**: healthy demand fills the theater randomly (seats scattered across sections). When a show fails silently (cast issue, technical collapse, word-of-mouth damage during previews), audiences still buy tickets but *self-organize spatially*—clustering in front rows, avoiding back sections, or concentrating in familiar sections. This clustering violates the spatial randomness expected from healthy demand.

Anomalies detection must work backward from occupancy *pattern structure* (spatial clustering measured by DBSCAN) rather than ticket volume or price.

## Data
**Synthetic.** Generates 50 shows with 8 rows × 12 seats (96 total seats):
- Shows 0-39 ("healthy"): random occupancy ~65–75% fill, seats scattered uniformly.
- Shows 40-49 ("failed"): clustered occupancy (front rows packed, back rows sparse; or lateral concentration).

Each show yields a flattened occupancy vector (1 = occupied, 0 = empty).

## Method
1. Convert flat seat occupancy into 2D spatial coordinates.
2. Apply DBSCAN clustering to occupied seats with `eps=2.5` (euclidean distance in seat-position space).
3. Compute anomaly score as `cluster_density = n_clusters / n_occupied_seats`.
4. Flag shows where cluster density or n_clusters exceeds 75th percentile.

## Limitation
Real theater box-office data would be proprietary and seat-occupancy per-show-performance is rarely logged (only revenue aggregates are tracked). The synthetic data structure is plausible but detection requires actual per-show seating records, which are not typically public. The technique itself is sound, but field validation is blocked by data availability.

## Output
Produces a table showing:
- `show_id`, `fill_rate`, `cluster_density`, `n_clusters`, `predicted_status`, `true_status`.
- Detection accuracy comparing predicted failure status to true labels.

---
*This is an auto-generated exploratory seed. Most seeds are discarded.*
