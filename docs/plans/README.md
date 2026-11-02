# Plans

[Next stage: models beyond RAM](beyond-ram-models.md) is the current priority:
extend the existing expert cache with bounded storage reads so a previously
non-fitting model can run. It contains bounded Luna-max assignments and Codex
correctness, hardware and useful-answer gates.

[Measured improvements](next-stage-measured-improvements.md) is a supporting
backlog, superseded as the next-stage priority by the capability plan above.

[PS5LM improvement plan](ps5lm-improvements.md) describes the current Luna
implementation packages, acceptance checks and measurement gates. Codex owns
integration and final validation; the plan does not imply a release or console
deployment.

The next phase, split into one goal per branch. Each branch starts from
`main`, merges back into `main` when its "done when" holds on the console,
and bumps the version in `ps5/app/version.hpp` with a CHANGELOG entry.

| Order | Branch | Plan | Version |
|---|---|---|---|
| 1 | `feature/dashboard-pages`, `feature/power-usage` | [Dashboard pages](dashboard-pages.md): widgets, model details, Settings and Logs pages on L1/R1; USAGE and POWER cards | 0.3.0, 0.4.0 (released) |
| 2 | `feature/cli-access` | [CLI access](cli-access.md): opencode, codex, Claude Code and others against the console | 0.5.0 (released) |
| 3 | `feature/chat-tools` | [Chat tools](chat-tools.md): tools, files, images in the browser chat; a scratch folder with a size warning | 0.6.0 (released) |
| 4 | `feature/model-market` | [Model market](model-market.md): search and download from Hugging Face, with fit shown | 0.6.0 (released) |

`main` is this fork's integration branch. Upstream pull requests go from a
branch cut from `cobanov/PS5LM:main` with only the commits meant for it.
