# Claude Code session — resuming on another machine

`session.jsonl` is the full transcript of the session that produced the M0–M2 work
on this branch. It is here so the work can be picked up on a different PC.

## Read this first: the transcript alone may not resume

Claude Code stores transcripts under a per-project directory whose name is derived
from the project's **absolute path**. On the machine this was recorded on:

```
C:\Users\<you>\.claude\projects\D--Automation-labelling-SPM\<session-uuid>.jsonl
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^
                                 derived from D:\Automation_labelling_SPM
```

So a copy of the file is only found by `claude --resume` if it sits at the matching
path for *that* machine's checkout. This is the mechanism as understood at the time
of writing — verify rather than rely on it.

## To try resuming on PC2

1. Clone this repo to **the same absolute path**, `D:\Automation_labelling_SPM`.
   A different drive or folder produces a different directory name and the
   session will not be listed.
2. Copy the transcript into place, keeping the original UUID filename:

   ```powershell
   $dst = "$env:USERPROFILE\.claude\projects\D--Automation-labelling-SPM"
   New-Item -ItemType Directory -Force $dst | Out-Null
   Copy-Item .claude-session\session.jsonl `
     "$dst\38db9755-31ba-4a4a-a496-83668b06bcfa.jsonl"
   ```

3. From `D:\Automation_labelling_SPM`, run `claude --resume` and pick the session.

If the session does not appear, the path or the storage layout does not match — use
the fallback below rather than fighting it.

## Fallback, and honestly the better handoff

`DECISIONS_LOG.md` in the repo root is the portable record and does not depend on
any of the above. It carries every decision from the pipeline team, what each one
settles, the consequences traced from it, what is built, and what is still open.
Paired with `MLFLOW_INTEGRATION_ANALYSIS.md` (the original gap analysis) and
`DOCS_REVIEW_FINDINGS.md`, a fresh session can be brought up to speed by reading
those three and the commit messages on this branch.

Starting a fresh session and pointing it at those files loses the conversational
history but none of the reasoning, because the reasoning was deliberately written
down as it happened rather than left in the chat.

## Where the work stopped

Done: M0.1, M0.2, W-1, W-2, W-3, W-4, M1, M2. 107 tests passing.

Open, in priority order:

- **M0.3** — `dataset_key` is an absolute host filesystem path used as the DB key
  in five tables, so moving hosts or changing the bind mount orphans every
  annotation, review and exemption row. Parked because it needs a `pg_dump` taken
  on the deploy host first.
- **M3** — split integrity is currently *measured* into the manifest but not
  enforced. M3 forces propagated and auto-accepted images into the train split and
  refuses to publish a snapshot that violates it.
- **M4** — frozen golden eval set in separate, curator-write-only storage. Accepted
  by the pipeline team; still needs named domain experts to curate it.
- **W-5** — wheel log-polar unwrap generated at export time from raw-space masks,
  with the unwrap parameters versioned per snapshot. Deliberately not annotated
  directly in log-polar space, because provisional wheel geometry constants will
  change and that would invalidate every mask drawn under the old ones.

Nothing on this branch has been verified against real data yet, and it includes
three unapplied migrations plus a new `boto3` dependency — so the backend image
needs a rebuild, not just a restart, and a database dump before first boot.
