"""
Databricks Job: Create MLflow Review Session

Fetches the last N traces from an MLflow experiment and creates a labeling
session for business partners to review via the MLflow Review App. Assigned
users receive an email invitation and write access to the session.

Usage (local / manual):
    python create_review_session.py \
        --experiment-id 3284073597979440 \
        --num-traces 10 \
        --reviewer-emails "reviewer1@company.com,reviewer2@company.com" \
        --session-name-prefix agent_review \
        --uc-catalog main \
        --uc-schema agent_review

This script is intended to be run as a Databricks job task via the DAB defined
in resources/eval_review_job.yml.
"""
import argparse
import sys
from datetime import datetime

import mlflow
from mlflow.genai import create_labeling_session
from mlflow.genai import label_schemas
from mlflow.genai.datasets import create_dataset, get_dataset


# ── Label schema ──────────────────────────────────────────────────────────────
# A single 1-5 numeric rating makes it easy for non-technical reviewers to
# score conversations without requiring ML background.
LABEL_SCHEMA_NAME = "conversation_quality"
LABEL_SCHEMA_INSTRUCTION = (
    "Review the conversation and rate the quality of the agent's response on a "
    "1–5 scale:\n"
    "  1 – Very poor: response is incorrect, off-topic, or harmful\n"
    "  2 – Poor: response partially addresses the question but has major gaps\n"
    "  3 – Acceptable: response is generally correct but could be clearer\n"
    "  4 – Good: response is helpful, accurate, and easy to understand\n"
    "  5 – Excellent: response fully and clearly addresses the user's need\n\n"
    "Please add a comment explaining your rating, especially for scores of 1 or 2."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an MLflow labeling session from recent experiment traces"
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="MLflow experiment ID to pull traces from",
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=10,
        help="Number of most recent traces to include (default: 10)",
    )
    parser.add_argument(
        "--reviewer-emails",
        default="",
        help="Comma-separated reviewer email addresses",
    )
    parser.add_argument(
        "--session-name-prefix",
        default="agent_review",
        help="Prefix for the labeling session name (a timestamp is appended)",
    )
    parser.add_argument(
        "--uc-catalog",
        required=True,
        help="Unity Catalog catalog for storing the review dataset",
    )
    parser.add_argument(
        "--uc-schema",
        required=True,
        help="Unity Catalog schema for storing the review dataset",
    )
    parser.add_argument(
        "--slack-secret-scope",
        default="",
        help=(
            "Databricks secret scope containing Slack credentials. "
            "Expected key: slack-bot-token. "
            "If omitted, Slack notifications are skipped."
        ),
    )
    parser.add_argument(
        "--notify-users",
        default="false",
        help="Send a Slack notification after session creation. Pass 'true' to enable (default: false).",
    )
    return parser.parse_args()


def _init_spark(uc_catalog: str, uc_schema: str):
    """Ensure a SparkSession exists and that the target UC schema exists.

    We only create the schema — catalog creation requires admin privileges and
    a storage location, so the catalog must already exist before running this job.
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{uc_catalog}`.`{uc_schema}`")


def _get_secret(scope: str, key: str) -> str:
    """Retrieve a secret from a Databricks secret scope via dbutils."""
    from databricks.sdk.runtime import dbutils
    return dbutils.secrets.get(scope=scope, key=key)


def _send_slack_notification(
    secret_scope: str,
    session_name: str,
    session_url: str,
    reviewer_emails: list,
    num_traces: int,
) -> None:
    """Send a Slack DM to brennan.beal@databricks.com with a summary of the new session."""
    from slack_sdk import WebClient

    token = _get_secret(secret_scope, "slack-bot-token")
    client = WebClient(token=token)

    notify_email = "brennan.beal@databricks.com"
    user = client.users_lookupByEmail(email=notify_email)
    channel = user["user"]["id"]

    reviewers_str = ", ".join(reviewer_emails) if reviewer_emails else "(none)"
    text = (
        f"*New agent review session created* :clipboard:\n"
        f"• *Session:* `{session_name}`\n"
        f"• *Traces:* {num_traces} conversations to review\n"
        f"• *Reviewers:* {reviewers_str}\n"
        f"• *Link:* {session_url}"
    )
    client.chat_postMessage(channel=channel, text=text)
    print(f"  Slack notification sent to {notify_email}")


def _fetch_traces(experiment_id: str, num_traces: int):
    """Return a DataFrame of the most recent N traces from the experiment."""
    traces_df = mlflow.search_traces(
        experiment_ids=[experiment_id],
        max_results=num_traces,
        order_by=["attributes.timestamp_ms DESC"],
        return_type="pandas",
    )

    if traces_df.empty:
        print("ERROR: No traces found in this experiment.")
        sys.exit(1)

    # merge_records() expects columns named 'inputs' and 'outputs'.
    # mlflow.search_traces() may return them as 'request' / 'response'.
    if "request" in traces_df.columns and "inputs" not in traces_df.columns:
        traces_df = traces_df.rename(columns={"request": "inputs"})
    if "response" in traces_df.columns and "outputs" not in traces_df.columns:
        traces_df = traces_df.rename(columns={"response": "outputs"})

    return traces_df


def _get_or_create_dataset(uc_table_name: str):
    """Return an existing MLflow-managed UC dataset or create a new one."""
    try:
        dataset = get_dataset(name=uc_table_name)
        print(f"  Reusing existing dataset: {uc_table_name}")
    except Exception:
        dataset = create_dataset(name=uc_table_name)
        print(f"  Created new dataset: {uc_table_name}")
    return dataset


def _ensure_label_schema() -> None:
    """Create (or overwrite) the label schema used by this review session."""
    label_schemas.create_label_schema(
        name=LABEL_SCHEMA_NAME,
        type="feedback",
        title="Conversation Quality (1–5)",
        input=label_schemas.InputNumeric(min_value=1.0, max_value=5.0),
        instruction=LABEL_SCHEMA_INSTRUCTION,
        enable_comment=True,
        overwrite=True,
    )


def main() -> None:
    args = parse_args()

    experiment_id = args.experiment_id
    num_traces = args.num_traces
    reviewer_emails = [e.strip() for e in args.reviewer_emails.split(",") if e.strip()]

    # Build a safe UC table name: lowercase alphanumeric + underscores only
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join(c if c.isalnum() or c == "_" else "_" for c in args.session_name_prefix)
    session_name = f"{safe_prefix}_{timestamp}"
    uc_table_name = f"{args.uc_catalog}.{args.uc_schema}.{session_name}"

    print("=" * 55)
    print("  Agent Eval Review Session Creator")
    print("=" * 55)
    print(f"  Experiment ID  : {experiment_id}")
    print(f"  Traces         : {num_traces} most recent")
    print(f"  Reviewers      : {reviewer_emails or '(none — set --reviewer-emails)'}")
    print(f"  Session name   : {session_name}")
    print(f"  Dataset path   : {uc_table_name}")
    print(f"  Secret scope   : {args.slack_secret_scope or '(not set — Slack notification skipped)'}")
    print("=" * 55)
    print()

    # ── 1. Configure MLflow ───────────────────────────────────────────────────
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_id=experiment_id)

    # ── 2. Initialise Spark and ensure UC catalog/schema exist ───────────────
    print("Initialising Spark session and ensuring UC catalog/schema exist...")
    _init_spark(args.uc_catalog, args.uc_schema)

    # ── 3. Fetch the last N traces ────────────────────────────────────────────
    print(f"Fetching {num_traces} most recent traces from experiment {experiment_id}...")
    traces_df = _fetch_traces(experiment_id, num_traces)
    print(f"  Found {len(traces_df)} traces.")

    # ── 4. Persist traces to a Unity Catalog dataset ─────────────────────────
    print(f"Writing traces to UC dataset...")
    dataset = _get_or_create_dataset(uc_table_name)
    dataset.merge_records(traces_df)
    print(f"  Merged {len(traces_df)} records.")

    # ── 5. Register the label schema for reviewers ───────────────────────────
    print("Registering label schema...")
    _ensure_label_schema()
    print(f"  Schema '{LABEL_SCHEMA_NAME}' ready.")

    # ── 6. Create the labeling session and attach the dataset ─────────────────
    print(f"Creating labeling session '{session_name}'...")
    labeling_session = create_labeling_session(
        name=session_name,
        assigned_users=reviewer_emails,
        label_schemas=[LABEL_SCHEMA_NAME],
    )
    labeling_session = labeling_session.add_dataset(dataset_name=uc_table_name)

    # ── 7. Notify via Slack ───────────────────────────────────────────────────
    if args.notify_users.lower() == "true" and args.slack_secret_scope:
        print("Sending Slack notification...")
        _send_slack_notification(
            secret_scope=args.slack_secret_scope,
            session_name=session_name,
            session_url=labeling_session.url,
            reviewer_emails=reviewer_emails,
            num_traces=len(traces_df),
        )
    else:
        print("Skipping Slack notification (notify_users not set or no secret scope configured).")

    # ── 8. Done ───────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  Review session created successfully!")
    print("=" * 55)
    print(f"  Session URL : {labeling_session.url}")
    print(f"  Reviewers   : {reviewer_emails}")
    print(f"  Traces      : {len(traces_df)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
