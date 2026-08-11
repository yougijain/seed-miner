# Little-League Schedule Equity via Isolation Forest

## Question
Can isolation forest detect unfair game scheduling (inequitable slot allocation and opponent assignment) by treating each team's schedule as a multivariate point in feature space, without requiring explicit fairness constraints or slot-counting rules?

## Why This Is Non-Obvious
Traditional fairness audits for league schedules count:
- Total weekend games per team
- Prime-time vs. off-peak slots
- Home/away balance

But they treat each constraint independently. Real inequity is *compositional*: a team that gets all early weekday games AND only strong opponents is doubly disadvantaged, but slot-counting methods miss the *interaction*.

Isolation Forest is built to find multivariate anomalies—points that don't cluster with peers—without defining what "fair" means. It detects teams whose *combined* exposure deviates from the population distribution, surfacing hidden biases that fairness metrics would miss.

## Data
**Synthetic.** 12 teams, 22 games each. Features per team:
- `avg_start_hour`: mean game time (proxy for primetime scarcity)
- `avg_opponent_strength`: ELO-like proxy for opponent quality
- `weekend_game_pct`: fraction of weekend vs. weekday games
- `primetime_pct`: fraction of 6–8pm slots

Two injected anomalies:
1. **Team_0** (disadvantaged): early slots, weak opponents, few weekend games
2. **Team_1** (privileged): late slots, strong opponents, many weekend games

Remaining 10 teams: fair, balanced schedules.

## What The Code Does
1. Generates synthetic team-schedule features.
2. Fits Isolation Forest (contamination=0.15).
3. Flags teams with anomalous multivariate schedule composition.
4. Outputs anomaly scores in rank order.

## Limitation
The synthetic data is tiny (12 teams) and injected anomalies are obvious. Real little leagues have 50–100+ teams and subtle biases. The key insight—*multivariate anomaly detection as a fairness audit*—generalizes, but you'd need:
- Actual league schedules (scrape from local league websites)
- More teams to avoid spurious clustering
- Feature engineering from detailed game logs (opponent strength via win%, not synthetic)

## Output
Teams flagged as anomalous have scheduling inequity detected *without* hand-coding fairness rules. The technique works because isolation forest finds points in the feature space that don't belong to the main cluster, which in this domain means "different treatment from peers."

---
*This is an auto-generated project seed for exploratory work. Most seeds are discarded.*
