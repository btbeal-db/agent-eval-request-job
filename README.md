# Agent Eval Review Session Creator

A Databricks job that automatically pulls the most recent traces from an MLflow experiment and creates a human labeling session via the MLflow Review App. Optionally sends a Slack notification to the job owner when the session is ready.

## What It Does

1. Fetches the last N traces from a specified MLflow experiment
2. Writes them to a Unity Catalog dataset
3. Registers a 1–5 numeric `conversation_quality` label schema
4. Creates an MLflow labeling session and assigns reviewers
5. Optionally DMs `brennan.beal@databricks.com` on Slack with a summary and link

Reviewers access the session via the [MLflow Review App](https://docs.databricks.com/aws/en/mlflow3/genai/human-feedback/concepts/labeling-sessions) and rate each conversation on a 1–5 scale.

## Prerequisites

### Unity Catalog

The target **catalog** must already exist — the job creates the schema if it doesn't exist, but it cannot create catalogs.

### Databricks Secret Scope

If Slack notifications are enabled, a secret scope must exist with the following key:

| Scope (default: `agent-eval-review`) | Key               | Value                                      |
|--------------------------------------|-------------------|--------------------------------------------|
| `agent-eval-review`                  | `slack-bot-token` | Slack bot token (`xoxb-...`)               |

Create the secret:
```bash
databricks secrets put-secret agent-eval-review slack-bot-token --string-value "xoxb-..." --profile FEVM
```

### Slack App Scopes

The Slack bot token must come from an app with the following OAuth scopes:

| Scope              | Purpose                                      |
|--------------------|----------------------------------------------|
| `chat:write`       | Send DMs to users                            |
| `users:read.email` | Look up a Slack user ID by email address     |

Set these up at [api.slack.com/apps](https://api.slack.com/apps) under **OAuth & Permissions → Bot Token Scopes**.

## Job Parameters

### Deploy-time Variables

Configured in `databricks.yml` and baked in at `databricks bundle deploy`. Override with `-v key=value` at deploy time.

| Parameter              | Variable              | Default                        | Description                                              |
|------------------------|-----------------------|--------------------------------|----------------------------------------------------------|
| `--experiment-id`      | `experiment_id`       | `3284073597979440`             | MLflow experiment ID to pull traces from                 |
| `--num-traces`         | `num_traces`          | `10`                           | Number of most recent traces to include                  |
| `--reviewer-emails`    | `reviewer_emails`     | `brennan.beal@databricks.com`  | Comma-separated list of reviewer emails                  |
| `--session-name-prefix`| `session_name_prefix` | `agent_review`                 | Prefix for the session name (timestamp is appended)      |
| `--uc-catalog`         | `uc_catalog`          | `hls_amer_catalog`             | Unity Catalog catalog for storing review datasets        |
| `--uc-schema`          | `uc_schema`           | `appeals-review`               | Unity Catalog schema for storing review datasets         |
| `--slack-secret-scope` | `slack_secret_scope`  | `agent-eval-review`            | Databricks secret scope holding the Slack bot token      |

### Runtime Parameters

Configured as a Databricks job parameter in `resources/eval_review_job.yml`. Override with `--params key=value` at run time — no redeployment needed.

| Parameter        | Job Parameter  | Default  | Description                                             |
|------------------|----------------|----------|---------------------------------------------------------|
| `--notify-users` | `notify_users` | `false`  | Set to `true` to send a Slack DM after session creation |

## Deployment

```bash
# Deploy to dev
databricks bundle deploy --profile FEVM

# Run manually
databricks bundle run agent_eval_review_job --profile FEVM

# Run with Slack notifications enabled (runtime override — no redeploy needed)
databricks bundle run agent_eval_review_job --profile FEVM --params notify_users=true
```

## Schedule

The job is configured to run every **Monday at 9:00 AM PT** (currently unpaused). Adjust the cron expression in `resources/eval_review_job.yml` as needed.
