# Curated Project Skill Routing

This repository keeps task-specific Skills in .agents/skills. Codex and Claude entry directories point to the same project-owned copies where links are present.

This table records the Skills reviewed or added by the 2026-09-11 audit; it does not remove or replace other project-owned Skills already present in the repository.

Load only the Skill whose trigger matches the current task. A Skill may be reused by several projects, and this project may use several Skills. Existing repository instructions and the user's current request take priority over a Skill.

| Skill | Use here for |
|---|---|
| `lingmao-experiment-closeout` | Default experiment reproduction, resume provenance, report and archive evidence |
| `pytorch-patterns` | Existing model-training workflow |
| `managing-python-dependencies` | Python environment and dependency changes |
| `github-actions` | Training-project CI and security workflows |

## Maintenance rules

- Keep domain and implementation Skills in the project instead of the global Codex Skill directory.
- Preserve project-specific Skills as the source of truth; do not replace them with an archived global copy.
- Prefer links for IDE-specific discovery so Codex and Claude read the same maintained content.
- Add or expand a Skill only when it captures repeatable project knowledge that is not already clear from code or repository documentation.
- For small changes, run focused checks first and expand testing only when failures, risk, or new scope justify it.
