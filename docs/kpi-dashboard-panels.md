# F-30: KPI Dashboard Panels

> 6-panel dashboard design for ORGANVM system observability.

## Panel Layout

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│   1. GOVERNANCE     │   2. DELIVERY       │   3. DOCUMENTATION  │
│                     │                     │                     │
│ Compliance rate     │ Cycle time          │ Doc coverage        │
│ Policy violations   │ Throughput          │ README quality      │
│ Audit findings      │ DORA-like metrics   │ Stale docs          │
├─────────────────────┼─────────────────────┼─────────────────────┤
│   4. FLOW           │   5. RELIABILITY    │   6. COMMUNITY &    │
│                     │                     │      DISTRIBUTION   │
│ WIP count           │ CI pass rate        │ ORGAN-VI events     │
│ Cycle time dist.    │ Test coverage       │ ORGAN-VII dispatches│
│ Blocked items       │ Error budgets       │ Reach metrics       │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

## Panel 1: Governance

**Purpose**: Are we following our own rules?

### Metrics

| Metric | Source | Refresh | Target |
|--------|--------|---------|--------|
| Governance compliance rate | `validate-deps.py` | Daily | >= 95% |
| Policy violations (back-edges) | `validate-deps.py` | Daily | 0 |
| seed.yaml coverage | Registry scan | Daily | 100% |
| Audit findings (open) | GitHub Issues with `audit` label | Weekly | Decreasing |
| Promotion pipeline health | registry.json status counts | Weekly | No LOCAL repos stalled > 30 days |

### Visualization

- **Primary**: Gauge showing compliance rate (green >= 95%, yellow >= 80%, red < 80%)
- **Secondary**: Bar chart of repos by promotion status (LOCAL, CANDIDATE, PUBLIC_PROCESS, GRADUATED, ARCHIVED)
- **Alert**: Red banner for any back-edge violation

### Data Query

```python
async def governance_panel():
    deps = run_validate_deps()
    registry = load_registry()
    return {
        "compliance_rate": deps.seed_coverage / deps.total_repos,
        "violations": deps.violations,
        "status_distribution": Counter(r["promotion_status"] for r in registry),
        "open_audit_findings": count_issues_with_label("audit"),
    }
```

## Panel 2: Delivery

**Purpose**: Are we shipping?

### Metrics

| Metric | Source | Refresh | Target |
|--------|--------|---------|--------|
| Cycle time (issue open → PR merged) | GitHub API | Weekly | <= 7 days median |
| Throughput (PRs merged/week) | GitHub API | Weekly | >= 3 |
| Lead time (commit → deploy) | GitHub Actions | Weekly | <= 1 day |
| Change failure rate | CI failures on main | Weekly | <= 15% |
| Deployment frequency | Releases/tags per week | Weekly | >= 1 |

### DORA-Like Metrics

Adapted for a solo-operator multi-repo system:

| DORA Metric | ORGANVM Adaptation |
|-------------|-------------------|
| Deployment frequency | Promotions + releases per week |
| Lead time for changes | Time from first commit to merge |
| Change failure rate | CI failures on main branch |
| Time to restore service | Time from issue creation to fix merge |

### Visualization

- **Primary**: Line chart of weekly throughput (PRs merged) over 12 weeks
- **Secondary**: Histogram of cycle time distribution
- **Sparklines**: 4 DORA metrics with trend arrows

## Panel 3: Documentation

**Purpose**: Does the documentation match reality?

### Metrics

| Metric | Source | Refresh | Target |
|--------|--------|---------|--------|
| Doc coverage (repos with docs/) | Filesystem scan | Weekly | >= 80% |
| README quality score | Custom linter | Weekly | >= 7/10 |
| Stale docs (> 90 days) | git log dates | Weekly | <= 20% |
| Feature backlog → doc ratio | FEATURE-BACKLOG.md vs docs/ | Monthly | Tracked |
| Broken internal links | Link checker | Weekly | 0 |

### README Quality Score (0-10)

Scoring rubric:
- Has description (2 pts)
- Has installation instructions (2 pts)
- Has usage examples (2 pts)
- Has contributing section or link (1 pt)
- Has license (1 pt)
- Updated within 30 days (1 pt)
- Has badges (CI, coverage) (1 pt)

### Visualization

- **Primary**: Heatmap of doc freshness by organ (rows) and repo (columns)
- **Secondary**: Bar chart of README quality scores, sorted ascending
- **Alert**: List of repos with no docs/ directory

## Panel 4: Flow

**Purpose**: Is work flowing or stuck?

### Metrics

| Metric | Source | Refresh | Target |
|--------|--------|---------|--------|
| WIP count (open PRs) | GitHub API | Daily | <= 5 |
| Cycle time distribution (p50, p90) | GitHub API | Weekly | p90 <= 14 days |
| Blocked items (issues with `blocked` label) | GitHub API | Daily | <= 3 |
| Queue depth (issues in `ready` state) | GitHub API | Daily | Tracked |
| Context switches (repos touched/day) | git log | Weekly | Tracked |

### Visualization

- **Primary**: Cumulative flow diagram (issues by state over time)
- **Secondary**: Scatter plot of cycle time per issue (spot outliers)
- **Alert**: Items blocked for > 7 days

### Little's Law Application

```
Avg. cycle time = WIP / Throughput

If WIP = 5 and Throughput = 3/week:
  Avg. cycle time = 5/3 = 1.67 weeks ≈ 12 days
```

Reducing WIP is the primary lever for reducing cycle time.

## Panel 5: Reliability

**Purpose**: Is the system stable?

### Metrics

| Metric | Source | Refresh | Target |
|--------|--------|---------|--------|
| CI pass rate (all repos) | GitHub Actions API | Daily | >= 90% |
| Test coverage trend | Coverage reports | Weekly | Non-decreasing |
| Flaky test count | CI logs (re-runs) | Weekly | <= 2 |
| Error budget remaining | Computed | Weekly | >= 50% |
| Build time trend | GitHub Actions timing | Weekly | Non-increasing |

### Error Budget Model

For each flagship repo, define an error budget:

```
Monthly error budget = (1 - SLO) × total_CI_runs
Example: SLO = 95%, 200 runs/month → budget = 10 failures

Budget consumed = actual_failures / budget
If budget consumed > 100% → freeze new features, fix reliability
```

### Visualization

- **Primary**: Line chart of CI pass rate per organ over 12 weeks
- **Secondary**: Test coverage trend lines for flagship repos
- **Alert**: Any repo with CI pass rate below 80%

## Panel 6: Community & Distribution

**Purpose**: Is the work reaching people?

### Metrics

| Metric | Source | Refresh | Target |
|--------|--------|---------|--------|
| ORGAN-VI events created | `koinonia-db/seed/` | Weekly | Tracked |
| ORGAN-VII dispatches sent | kerygma dispatch logs | Weekly | Tracked |
| Community engagement (issue comments from non-owner) | GitHub API | Weekly | Tracked |
| GitHub stars (total, delta) | GitHub API | Weekly | Tracked |
| Page views (GitHub Insights) | GitHub API | Monthly | Tracked |

### Visualization

- **Primary**: Bar chart of dispatch count by channel (Twitter, Bluesky, Mastodon, RSS)
- **Secondary**: Event timeline (ORGAN-VI events on a calendar view)
- **Sparkline**: Total stars trend

## Data Pipeline

```
GitHub API ──┐
             │
registry.json┤
             ├──→ Aggregator Script ──→ metrics.json ──→ Dashboard
organ-audit  │    (calculate-metrics.py)
             │
git log ─────┘
```

### Aggregator Output

```json
{
  "generated_at": "2026-03-08T12:00:00Z",
  "governance": {
    "compliance_rate": 0.97,
    "violations": 0,
    "seed_coverage": 105,
    "total_repos": 105
  },
  "delivery": {
    "prs_merged_this_week": 4,
    "median_cycle_time_days": 3.2,
    "promotions_this_week": 1
  },
  "documentation": {
    "doc_coverage": 0.82,
    "stale_docs_pct": 0.15,
    "avg_readme_score": 7.3
  },
  "flow": {
    "wip_count": 3,
    "blocked_count": 1,
    "p90_cycle_time_days": 11
  },
  "reliability": {
    "ci_pass_rate": 0.93,
    "flaky_tests": 1,
    "avg_build_time_sec": 45
  },
  "community": {
    "organ_vi_events": 23,
    "organ_vii_dispatches": 12,
    "total_stars": 47
  }
}
```

## Implementation Options

### Option A: Extend system-dashboard (meta-organvm)

Add panel routes to the existing FastAPI dashboard:

```python
# dashboard/panels.py
@app.get("/api/panels/{panel_name}")
async def get_panel(panel_name: str):
    metrics = load_metrics()
    return metrics[panel_name]
```

Frontend: simple HTML/JS with Chart.js or a lightweight charting library.

### Option B: Standalone Static Report

Generate a weekly HTML report from `metrics.json`:

```bash
python3 scripts/generate-dashboard.py > reports/dashboard-$(date +%Y-W%V).html
```

### Option C: GitHub Pages Dashboard

Publish `metrics.json` to a GitHub Pages site with a static JS frontend that renders charts client-side.

**Recommendation**: Start with Option B (static report) for immediate value, migrate to Option A when system-dashboard is mature.

## References

- `conductors-scorecard.md` (F-29) — the 4 core metrics this dashboard expands on
- `meta-organvm/system-dashboard/` — existing dashboard infrastructure
- `orchestration-start-here/scripts/calculate-metrics.py` — metrics calculation
- `orchestration-start-here/scripts/organ-audit.py` — system audit
- `cost-latency-monitoring.md` (F-31) — cost/latency metrics feed into Reliability panel
