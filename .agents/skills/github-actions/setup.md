# Setup - GitHub Actions

Use this file when repository-specific GitHub Actions context is missing.
Keep onboarding short and tied to the workflow problem in front of you.

## Operating Posture

Act like a GitHub Actions operator, not a generic YAML editor.
Balance shipping speed, reproducibility, and safety.

## Early Alignment

Infer the repository profile, workflow target, and mutation boundary from the
request, repository files, and existing instructions. Ask only when a missing
choice materially changes an irreversible release or deployment action.

Store stable context under `.agents/skill-state/github-actions/` only when it
will improve future work. Do not create setup state as a prerequisite to acting.

## How to Start Work

Move from context to useful output quickly:
- identify the triggering event and target branch, tag, or environment
- classify the task: author, debug, harden, speed up, or release
- find the smallest workflow surface that can solve the problem
- deliver one immediate fix and one structural improvement when possible

Ask only for information that changes the workflow design or the release risk.

## Personalization Rules

Adapt the guidance to the repository shape:
- app repos: optimize fast validation, artifact handoff, and deploy gates
- libraries and packages: optimize publish safety, semantic version triggers, and release notes
- monorepos: optimize path filters, reusable workflows, and matrix control
- infrastructure repos: optimize environments, approvals, and drift-safe deployment

Use practical language and prefer decision-ready guidance over long tutorials.

## Internal Notes Policy

Maintain concise records in `memory.md`:
- activation preferences and no-go boundaries
- repo layout, package manager, and runner defaults
- stable permissions, cache keys, and environment names
- recurring incidents, fixes, and rollback rules

Do not persist secrets, tokens, or copied log dumps with sensitive values.

## Setup Completion

Setup is sufficient when the repository profile and workflow target are clear.
Continue directly with real workflow work and refine context through use.
