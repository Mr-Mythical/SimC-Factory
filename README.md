# Mr. Mythical: SimC Factory

Mr. Mythical: SimC Factory is an internal operations tool for producing, refreshing, and validating SimulationCraft training data at scale. It coordinates profile generation, AWS Batch simulation runs, result collection, model evaluation, and addon export from one controlled server-side workflow.

This is not a public web service. The FastAPI dashboard is an internal operator console for running simulations, inspecting model health, managing AWS-backed workflows, and preparing addon releases. The exported addon, Mr. Mythical: DPS Predictor, is the only artifact intended for end users.

## Project Goal

The goal is to keep the user-facing addon lightweight, current, and easy to validate while moving the expensive work into a repeatable internal pipeline.

That means this project is built to:

- scale SimulationCraft runs without running them on user machines
- keep cloud infrastructure and credentials away from addon users
- refresh stale or incomplete spec data with minimal manual work
- evaluate model quality before new addon data is exported
- make Mr. Mythical: DPS Predictor releases reproducible through export and offline parity checks

## What the Tool Does

- Runs high-volume SimulationCraft jobs across AWS Batch array workers.
- Generates and refreshes per-spec training data for DPS prediction.
- Tracks stale, incomplete, or underperforming specs so they can be regenerated.
- Evaluates model quality and supports local/offline addon parity checks.
- Exports trained model data into Lua tables consumed by Mr. Mythical: DPS Predictor.
- Provides an internal dashboard for operators to launch jobs, inspect pipelines, manage credentials, and review model/spec status.

## Intended Shape

The system is split into internal tooling and a user-facing addon:

- **Internal server or workstation:** runs the orchestrator, dashboard, model tooling, AWS polling, result merging, and addon export.
- **AWS Batch workers:** run only SimulationCraft plus a small shell worker, keeping cloud jobs lightweight and disposable.
- **S3:** acts as temporary transport for chunk inputs, manifests, and output zips.
- **Mr. Mythical: DPS Predictor:** ships the exported model data and prediction logic to users without exposing the dashboard, AWS infrastructure, or training workflow.

This keeps Python, pandas, numpy, boto3, training artifacts, credentials, and orchestration logic on a trusted internal host. The user-facing addon is a simple Lua implementation that only consumes the final model data, so it can be safely distributed without exposing any of the internal complexity or infrastructure.
