# REV-110 reachability probe -- ATTEMPT 1 FAILED, kept beside its correction

**G11 `REV-110/02-reachability.json`, exit code 1, 0 survivors, cleanup success.**

Attempt 1 called `driver._kind`. There is no such attribute:

```
AttributeError: module 't2pw.batch.driver' has no attribute '_kind'
```

The function that maps evidence onto one `failure_kind` is named
**`driver._classify`** (`src/t2pw/batch/driver.py:1240`); `_kind` was my own
misreading of the call sites, which pass `kind=` as a keyword.

The probe was corrected to `driver._classify` and re-run as
`REV-110/03-reachability-r2.json` (exit 0). **No conclusion was drawn from
attempt 1** -- it never reached the branch it was written to measure, which is
exactly the failure mode the C-108 verification probe had. `rev110_reachability.log`
holds the corrected run.
