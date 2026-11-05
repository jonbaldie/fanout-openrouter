# Exploratory pass: completions and policy reload

Date: 2026-11-05. Base: `a397666` on `origin/main`. Python 3.14.4, uv 0.9.26, and just 1.59.0 on Linux. Dependencies were installed in a separate worktree with `uv sync --locked --all-extras`. The pass used no dotenv file, server key, or client credential. The server listened on loopback port 8000. [Captured HTTP results and teardown checks](2026-11-05-output.txt) accompany this report.

## Discover and change a policy

The [policy guide](../policies.md#reloading-policies) says `just dev` reloads JSON saved in the project directory. A disposable policy file contained one model, `test/first`, with candidate `openai/gpt-4o-mini`, fanout count 1, and created timestamp 1710000000. `FANOUT_POLICY_FILE` pointed to that file. `GET /api/v1/models` returned `test/first` after startup. Saving `test/second` made the endpoint return `test/second`. Restoring `test/first` and saving `test/second` again produced the same results. This checks model discovery after reload, not the upstream routing outcome. The original [stale policy issue #2](https://github.com/jonbaldie/fanout-openrouter/issues/2) is closed by [PR #7](https://github.com/jonbaldie/fanout-openrouter/pull/7). This pass did not reproduce the stale discovery result.

## Request a completion

The README promises non-streaming and streaming completions. A valid `POST /api/v1/chat/completions` with model `test/first` and user message `Hello` returned HTTP 401 with `No cookie auth credentials found`. The same request with `stream: true` returned the same JSON error. A request with the model but no messages returned HTTP 400 with `Input required: specify "prompt" or "messages"`.

These are missing-credential and validation paths, not successful completions. No OpenRouter key was available in the environment or isolated worktree. An authenticated ordinary request and the streaming variation remain blocked until a maintainer provides a valid key. Fan-out, synthesis, fallback, tool calls, streaming progress, and direct parity with OpenRouter were not exercised. The message mentions cookie authentication even though the documented client configuration uses a bearer token. This is a usability observation, not a confirmed contract defect, because the pass did not compare that error with OpenRouter.

## Findings and checks

No new bug was confirmed or filed. There are no unresolved suspected failures from the exercised paths. [Issue #4](https://github.com/jonbaldie/fanout-openrouter/issues/4) tracks a separate rate-limit retention problem that this pass did not test. The existing Beads export and GitHub issues, labels, comments, and linked PRs were checked. `docs/agents/issue-tracker.md` and the `bd` command were unavailable. GitHub has no `needs-triage` label.

`ruff check .` passed, `ruff format --check .` reported 17 files already formatted, and `CI=true uv run --locked pytest -m 'not live' -q` reported 11 passed and 28 deselected. These checks do not replace live parity testing.

The server process group was registered before readiness polling and sent SIGTERM in a `finally` block. The immediate check ran before the reloader and worker finished exiting, so it still saw the group and listening port. A delayed check three seconds later found neither the process group nor a listener on port 8000. No browser, container, Docker image, or Docker volume was started by this pass.
