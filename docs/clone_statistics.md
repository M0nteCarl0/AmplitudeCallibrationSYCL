# Clone Statistics Badge

The README [`Clones`](../README.md) badge reports the **lifetime** number of clones of this repository.

## Why it needs automation

shields.io has no native clones badge and cannot get one: the GitHub
[Traffic API](https://docs.github.com/en/rest/metrics/traffic) exposes clone counts only to
principals with push access to the repository, and only for the **last 14 days**. A public image
service can therefore never read it directly.

[`.github/workflows/clone-stats.yml`](../.github/workflows/clone-stats.yml) closes that gap: once a
day it reads `/repos/{owner}/{repo}/traffic/clones` with a token, merges the window into a
never-shrinking per-day history, and pushes both files to the orphan **`traffic`** branch.

| File on `traffic` | Purpose |
| :--- | :--- |
| `clones.json` | shields.io [endpoint](https://shields.io/badges/endpoint-badge) payload (`schemaVersion`, `label`, `message`, `color`) rendered by the README badge |
| `clones-daily.json` | accumulated per-day history and lifetime totals — the source of truth |
| `README.md` | generated notice for the branch |

The badge itself is a stateless view of that branch:

```text
https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FM0nteCarl0%2FAmplitudeCallibrationSYCL%2Ftraffic%2Fclones.json
```

Because the badge is rendered from a file we control, no token ever reaches the browser and the
counter survives GitHub's 14-day retention window.

## One-time setup

1. Create a personal access token (PAT) that can read traffic data **and** push the branch:

   | Token type | Required access |
   | :--- | :--- |
   | Classic | `repo` scope |
   | Fine-grained | `Contents: read and write` + `Administration: read` |

   The workflow's default `GITHUB_TOKEN` is rejected by the traffic endpoints
   (*Resource not accessible by integration*), so a PAT is mandatory.

2. Add it to the repository as the secret **`GH_TRAFFIC_TOKEN`**
   (Settings → Secrets and variables → Actions → New repository secret).
   Until that secret exists, every run fails immediately with an explicit
   `GH_TRAFFIC_TOKEN is not set` annotation — nothing is published.

3. Run the workflow once: **Actions → Clone Statistics → Run workflow** (or wait for the daily
   `23 4 * * *` UTC run). The badge shows an error until the first successful run.

## Behaviour and caveats

- **Counting starts on the first successful run.** Traffic older than 14 days is already gone, so
  the badge cannot be back-dated.
- **Runs are idempotent.** Each stored day keeps the maximum value GitHub reports for it, so
  re-running the workflow, or seeing the same day in later windows, never double counts.
- **The history never shrinks.** Stored days are never dropped and per-day counts are only ever
  raised, so the lifetime total is monotone. The published file is validated before merging —
  a malformed or truncated history aborts the run instead of being overwritten.
- **Counts include automated clones** — bots, crawlers, mirrors and CI (including workflow
  checkouts), not only humans.
- **Scheduled workflows pause** after 60 days without repository activity; a gap means those days
  are lost permanently.
- The branch is rewritten as a **single force-pushed commit** on every run to keep it free of daily
  commit noise.

## Manual dry run

The script is stdlib-only and can be exercised offline with JSON fixtures instead of the API:

```bash
python3 .github/scripts/update_clone_stats.py --output /tmp/stats \
  --traffic-file traffic.json --history-file clones-daily.json
```
