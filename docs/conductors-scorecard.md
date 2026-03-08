# F-29: Conductor's Scorecard

> 4 core metrics for the weekly 10-minute review.

## Purpose

The Conductor's Scorecard distills ORGANVM system health into 4 metrics that can be reviewed in 10 minutes every Monday. Each metric answers a specific question about whether the system is progressing, governed, documented, and shipping.

## The 4 Metrics

### 1. Velocity / Promotion Throughput

**Question**: Is the system moving forward?

| Indicator | Target | Source |
|-----------|--------|--------|
| Repos promoted this week | >= 1 | registry.json diff |
| Issues closed this week | >= 5 | GitHub API |
| PRs merged this week | >= 3 | GitHub API |
| New repos created | Tracked (no target) | GitHub API |

**Calculation**:

```bash
# Promotions: diff registry.json for promotion_status changes
git diff HEAD~7..HEAD -- registry.json | grep promotion_status

# Issues closed (all ORGAN-IV orgs)
gh search issues --state closed --updated ">=$(date -v-7d +%Y-%m-%d)" --owner organvm-iv-taxis --json number | jq length

# PRs merged
gh search prs --merged --updated ">=$(date -v-7d +%Y-%m-%d)" --owner organvm-iv-taxis --json number | jq length
```

**Anomaly flags**:
- Zero promotions for 2+ consecutive weeks
- Issue close rate declining week over week
- PRs opened but not merged (growing review backlog)

### 2. Governance Compliance Rate

**Question**: Is the system governed?

| Indicator | Target | Source |
|-----------|--------|--------|
| Repos with valid seed.yaml | 100% | `validate-deps.py` output |
| PRs with governance checklist | >= 80% | PR template compliance |
| Back-edge violations detected | 0 | `validate-deps.py` |
| CI pass rate (latest run) | >= 90% | GitHub Actions API |

**Calculation**:

```bash
# Seed.yaml coverage
cd orchestration-start-here
python3 scripts/validate-deps.py --json | jq '.seed_coverage'

# CI pass rate
python3 scripts/organ-audit.py --json | jq '.ci_pass_rate'

# Back-edge violations
python3 scripts/validate-deps.py --json | jq '.violations | length'
```

**Anomaly flags**:
- Any back-edge violation (immediate action required)
- seed.yaml coverage drops below 95%
- CI pass rate below 85%

### 3. Doc-Implementation Gap

**Question**: Does the documentation match reality?

| Indicator | Target | Source |
|-----------|--------|--------|
| Features documented but not implemented | Tracked | Feature backlog vs codebase |
| Code without matching docs | Tracked | Module scan vs docs/ |
| Stale docs (no update in 90 days) | <= 20% | File modification dates |
| README freshness (updated within 30 days) | >= 80% | GitHub API |

**Calculation**:

```bash
# Stale docs: files in docs/ not modified in 90 days
find docs/ -name "*.md" -mtime +90 | wc -l

# README freshness: check last commit date for README.md
git log -1 --format="%ci" -- README.md
```

**Anomaly flags**:
- More than 5 documented features with zero implementation
- Core modules with no corresponding documentation
- README older than 60 days in any flagship repo

### 4. Spec-to-Merge Ratio

**Question**: Are ideas converting to shipped code?

| Indicator | Target | Source |
|-----------|--------|--------|
| Issues created this week | Tracked | GitHub API |
| PRs merged this week | Tracked | GitHub API |
| Ratio (merged / created) | >= 0.5 | Computed |
| Median issue age at close | <= 14 days | GitHub API |

**Calculation**:

```bash
# Issues created
CREATED=$(gh search issues --created ">=$(date -v-7d +%Y-%m-%d)" --owner organvm-iv-taxis --json number | jq length)

# PRs merged
MERGED=$(gh search prs --merged --updated ">=$(date -v-7d +%Y-%m-%d)" --owner organvm-iv-taxis --json number | jq length)

# Ratio
echo "scale=2; $MERGED / $CREATED" | bc
```

**Anomaly flags**:
- Ratio below 0.3 for 2+ consecutive weeks (creating faster than shipping)
- Ratio above 2.0 (shipping without planning — may indicate skipped governance)
- Median issue age exceeding 30 days (stale backlog)

## Data Sources

| Source | Access Method | Refresh |
|--------|--------------|---------|
| registry.json | Local file read | Real-time |
| GitHub Issues/PRs | `gh` CLI or GitHub API | Weekly |
| `validate-deps.py` | Local script execution | Weekly |
| `organ-audit.py` | Local script execution | Weekly |
| GitHub Actions | GitHub API | Weekly |
| File modification dates | `git log` / `find` | Weekly |

## Dashboard Integration

### Option A: Extend system-dashboard

Add a `/scorecard` route to the existing `meta-organvm/system-dashboard` FastAPI app:

```python
@app.get("/scorecard")
async def scorecard():
    return {
        "week": current_iso_week(),
        "velocity": await compute_velocity(),
        "governance": await compute_governance(),
        "doc_gap": await compute_doc_gap(),
        "spec_to_merge": await compute_spec_to_merge(),
    }
```

### Option B: Standalone Weekly Report

A script that generates a markdown report:

```bash
python3 scripts/weekly-scorecard.py > reports/scorecard-$(date +%Y-W%V).md
```

Output:

```markdown
# Scorecard — 2026-W10

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Promotions | 2 | >= 1 | OK |
| Issues closed | 8 | >= 5 | OK |
| Governance compliance | 97% | >= 80% | OK |
| Back-edge violations | 0 | 0 | OK |
| Stale docs | 12% | <= 20% | OK |
| Spec-to-merge ratio | 0.62 | >= 0.5 | OK |

**Anomalies**: None
```

## Review Ritual

**When**: Every Monday morning, 10 minutes maximum.

**Process**:
1. Run the scorecard script or check the dashboard (2 min)
2. Scan for anomaly flags — any red items? (2 min)
3. If anomalies: create issue(s) with `priority: high` label (3 min)
4. If clean: note in weekly log and move on (1 min)
5. Compare against previous week's scorecard for trends (2 min)

**Escalation**: If the same anomaly appears 3 consecutive weeks, it becomes a blocking item for the next sprint.

## References

- `orchestration-start-here/scripts/organ-audit.py` — system audit script
- `orchestration-start-here/scripts/validate-deps.py` — dependency validation
- `orchestration-start-here/scripts/calculate-metrics.py` — registry metrics
- `meta-organvm/system-dashboard/` — existing dashboard
- `kpi-dashboard-panels.md` (F-30) — detailed panel designs
