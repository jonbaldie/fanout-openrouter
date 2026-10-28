# Exploratory pass: API and policy configuration

Date: 2026-10-28. Base: `1fc6cf1` on `origin/main`. Python 3.14.4, uv 0.13.6, dependencies installed with `uv sync --locked --all-extras`. The server ran on loopback port 18761 in an isolated worktree. No real API key or dotenv file was available. OpenRouter's public models endpoint returned HTTP 200, but this does not establish authenticated completion access.

## Journeys

### Discover a virtual model and request a completion

The README promises `GET /api/v1/models` and chat completions. The default model list returned seven IDs, including `fanout/minimal`. A non-streaming completion without credentials returned 401 with `No cookie auth credentials found`. The streaming variation also returned 401. A request missing messages returned 400 with an explanatory error. These are observable error paths, not proof of a successful completion. Real fan-out, pass-through, tool calling, synthesis, and SSE delivery remain blocked by the missing OpenRouter key. A maintainer-provided key is needed to complete them.

### Edit a policy and see it take effect

The [policy guide](../policies.md#reloading-policies) says saving JSON in development mode automatically restarts the server. In a disposable `fanout_policies.json`, a single policy named `test/first` appeared in `/api/v1/models`. With Uvicorn `--reload` running, changing it to `test/second` left the old model visible after four seconds. Restarting the server showed the new model. The same sequence was replayed from the original policy with the same result. The original file was restored. This is confirmed in [issue #2](https://github.com/jonbaldie/fanout-openrouter/issues/2). The exact configuration and output are in the issue and [captured output](2026-10-28-output.txt).

Users following the guide may unknowingly continue routing requests with stale configuration. Restarting manually works as a workaround. The experiment used the locked dependencies, which do not include the optional `watchfiles` package.

## Checks and limitations

`CI=true uv run --locked pytest -m 'not live' -q` reported 10 passed and 29 deselected. These tests are supplementary to the HTTP journeys and do not establish live API parity. No authenticated request or live model generation was attempted without a key. No browser, container, Docker image, or volume was created. Each Uvicorn process group was registered with an exit cleanup handler before readiness polling, stopped after use, and the loopback port was checked after teardown. Process group IDs and results are in the captured output.

GitHub had no existing issues or prior rejected issues at the start of the pass. The checked-in Beads export contained no matching policy-reload issue. `docs/agents/issue-tracker.md` was absent, and `bd` was not installed, so the finding was filed on GitHub rather than silently dropped. The repository had no `needs-triage` label. No product fix was made.
