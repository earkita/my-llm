---
name: diagnose-r9700-log
description: Inspect and diagnose the newest R9700 inference runtime log or a user-supplied log path, correlate it with managed service state, and distinguish application errors, OOM or signals, graceful shutdown, and unexplained external termination. Use when the service stopped, startup failed, a model exited, or the user asks what happened in the last log.
---

# Diagnose R9700 Log

1. Work from the repository containing this skill.
2. Run the bundled read-only collector:

   ```bash
   skills/diagnose-r9700-log/scripts/diagnose-last-log.sh
   ```

   Pass `--log PATH` for a specific file or `--lines N` to change the displayed tail.
3. Read the classification, matched evidence, final log lines, service status, and stale/live PID evidence together.
4. Report the most likely cause with exact supporting log lines. Separate facts from inference and state confidence.

Classify conservatively:

- Treat traceback, explicit fatal/error, OOM, signal, abort, or shutdown messages as direct evidence.
- Treat a clean shutdown sequence as intentional or graceful termination.
- If the log ends after successful readiness, contains no failure/shutdown marker, and the recorded PID is absent, report an abrupt external termination as likely but unproven. The application log cannot identify an external SIGKILL, cgroup cleanup, host OOM killer, reboot, or power loss.
- Treat ordinary warnings about eager mode, optional CUDA features on ROCm, generation defaults, or deprecated APIs as nonfatal unless followed by a failure.

Do not start, stop, restart, delete state, or alter logs. Ask before expanding diagnosis to host evidence such as kernel or system journal logs. Never claim a root cause that the available evidence cannot prove.
