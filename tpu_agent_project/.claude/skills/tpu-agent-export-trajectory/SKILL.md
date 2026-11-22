---
name: tpu-agent-export-trajectory
description: "Save this session's trajectory -- the tpu-agent work and every subagent beneath it -- to the destination the user chose for trajectory sharing."
---

# Export this session's trajectory

Saves this session -- the tpu-agent work and every subagent beneath it --
to the sink the user chose. That is either a GCS bucket, or
Harbor as an alternative sink, which renders the session as a browsable
trajectory. A Harbor install can additionally be configured to share the
upload with a team; without that the upload is private to the uploader.

The two sinks are exclusive and neither is a default: sharing is off until
the user turns it on, and then it goes where they chose. The command knows
which one is configured, so do not assume, and do not describe one sink's
behaviour when the other is in use.

## What to run

Run exactly this, and nothing else:

```
python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@0.7.0/framework/runtime.py trajectory export --harness claude-code --package tpu-agent-suite@0.7.0 --run-id "<this session's run id>" --note "<the user's words>"
```

`--run-id` is the value on the `TPU agent suite run namespace:` line in your
context. Copy it exactly. If your context has no such line, do not run the
command: tell the user this session has no run id and stop.

`--note` is an optional remark the user wants stored alongside the
trajectory, such as what went wrong. If they gave none, pass `--note ""`.

Report what the command prints, including the destination URI and the
session id it names. Do not summarize away the URI: the user needs to see
where their session was saved.

## Rules

- Run the command once. Do not construct a different command, add flags the
  user did not ask for, or assemble anything by hand.
- Everything else the command needs it discovers for itself. Do not pass
  `--session-id`, `--bucket`, `--user-id`, `--destination` or `--share-org`
  unless the user gave you that value explicitly. In particular
  `--destination` overrides the choice the user made, so
  sending a session somewhere they did not pick is never yours to decide.

## If it reports that the session's starting point was not found

The command refuses rather than saving the whole session, because it cannot
tell which part of the session was tpu-agent's work.

When that happens:

1. Tell the user the starting point could not be found, so the export would
   cover the **entire session**, including anything they did beforehand.
2. Ask whether to export it anyway.
3. Only if they say yes, re-run the same command with `--all` added.

Never add `--all` on your own initiative, and never add it because the first
attempt failed. It is the user's consent to save more than they asked for,
and it is recorded in the manifest as theirs.

## If it reports that export is not supported

Some harnesses have no transcript reader yet. Relay the message as-is. There
is no workaround to offer and nothing was saved.

## If it reports that no destination is configured

Trajectory sharing was never set up. It stays off until the user turns it
on, so this is the time to ask:

1. Tell the user nothing was saved because sharing is not set up, and ask
   whether they want to share session trajectories at all.
2. If they do, ask where: a GCS bucket of their own (`gs://...`), or Harbor,
   which renders the session as a browsable trajectory. For Harbor, ask
   whether to share uploads with an organization (for example `my-team`);
   without one the upload is private to them.
3. Save their answer. For a bucket:

   ```
   python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@0.7.0/framework/runtime.py config --package tpu-agent-suite@0.7.0 set trajectory_enabled=true trajectory_sink=gcs trajectory_bucket=<their bucket> --user-confirmed
   ```

   For Harbor, `trajectory_sink=harbor` and, if they named one,
   `trajectory_share_org=<their org>` instead of the bucket.
4. Re-run the export command above.

If they say no, save that, so they are not asked again:

```
python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@0.7.0/framework/runtime.py config --package tpu-agent-suite@0.7.0 set trajectory_enabled=false --user-confirmed
```

Then relay that nothing was saved and stop. Do not suggest a destination.
Declining is an answer, not a gap to fill.

Pass `--user-confirmed` only with values the user gave you in this
conversation.

## If it reports that trajectory sharing is off

The user already chose not to share. Relay that nothing was saved and stop;
do not ask them again. Only if they now ask you to share with a destination
of their choosing, follow steps 2 to 4 above.
