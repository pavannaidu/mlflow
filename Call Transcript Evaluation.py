# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Notebook Overview
# MAGIC %md
# MAGIC ## Call Transcript Evaluation
# MAGIC
# MAGIC Example of using **MLflow GenAI** with `ai_query()` to extract structured data from unstructured text, then running **offline evaluations** (LLM-as-a-Judge + deterministic scorers) to compare model quality.
# MAGIC
# MAGIC **What this covers:**
# MAGIC * Structured output extraction via `ai_query()` with JSON schema constraints
# MAGIC * Registering evaluation datasets in MLflow (Unity Catalog-backed)
# MAGIC * Defining scorers — built-in judges, custom Guidelines judges, and code-based validators
# MAGIC * Multi-model comparison with `mlflow.genai.evaluate()`
# MAGIC
# MAGIC **Compute:** Serverless (CPU), Environment 5

# COMMAND ----------

# DBTITLE 1,Setup & Configuration
# MAGIC %pip install mlflow[databricks]==3.15 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration
# ---- Update these for your environment ----
CATALOG = "pavan_naidu_catalog"
SCHEMA = "mlflow"
WAREHOUSE_ID = "fb3b5c1dee0b9d55"
EXPERIMENT_NAME = "/Users/pavan.naidu@databricks.com/Call Transcript Evaluation Experiment"

MODELS = [
    "system.ai.meta-llama-3-3-70b-instruct",
    "system.ai.gpt-5-4-mini",
    "system.ai.gemini-3-5-flash",
]

# Derived paths
SCHEMA_FULL = f"{CATALOG}.{SCHEMA}"
TABLE_TRANSCRIPTS = f"{SCHEMA_FULL}.call_transcripts"
TABLE_INSIGHTS = f"{SCHEMA_FULL}.call_insights"
DATASET_NAME = f"{SCHEMA_FULL}.call_insights_eval_dataset"

print(f"Catalog:    {CATALOG}")
print(f"Schema:     {SCHEMA_FULL}")
print(f"Models:     {len(MODELS)}")
print(f"Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# DBTITLE 1,Create Source Table: call_transcripts
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_FULL}")

spark.sql(f"""
CREATE OR REPLACE TABLE {TABLE_TRANSCRIPTS} (
  call_id STRING,
  call_date TIMESTAMP,
  agent_id STRING,
  customer_id STRING,
  transcript STRING
)
COMMENT 'Raw customer call transcripts for AI-powered analysis'
""")

spark.sql(f"""
INSERT INTO {TABLE_TRANSCRIPTS} VALUES
  ('CALL-001', '2026-08-18T09:15:00', 'AGT-101', 'CUST-5001',
   'Agent: Thank you for calling Acme Support, how can I help you today?\nCustomer: Hi, I purchased the ProMax Router last week and I cannot get it to connect to my network. I have tried resetting it multiple times.\nAgent: I understand how frustrating that must be. Let me walk you through an advanced reset procedure. Please hold the reset button for 30 seconds while the device is powered on.\nCustomer: Okay, doing that now... Oh wait, the lights are blinking differently now. Let me try connecting again... It works! Thank you so much.\nAgent: Great to hear! Is there anything else I can help you with?\nCustomer: No, that is all. Thanks again!'),

  ('CALL-002', '2026-08-18T10:32:00', 'AGT-102', 'CUST-5002',
   'Agent: Acme Support, this is Sarah. How may I assist you?\nCustomer: I am absolutely livid. This is my FOURTH call about the same issue. I have been charged twice for my subscription three months in a row and every time I call, someone says they will fix it and nothing happens. I have screenshots of every duplicate charge. I want to speak to a manager.\nAgent: I completely understand your frustration, and I sincerely apologize for the repeated failures. Let me pull up your full billing history right now. I can see three months of duplicate charges totaling $89.97. I am going to process all three refunds immediately and escalate an internal ticket to our billing engineering team.\nCustomer: You say that but the last three agents said the same thing. Why should I believe you?\nAgent: That is a completely fair question. Here is what I am going to do differently. I am giving you a case reference number BIL-2026-4847. I am also adding a $50 account credit for the time you have spent on this. And I am setting a calendar reminder to personally follow up with you via email in 48 hours with a status update. If for any reason the refunds do not appear within 5 business days, you can reference that case number and skip the queue entirely.\nCustomer: Okay, well, at least you are being specific about it. The case number helps. I just cannot keep calling about this every month.\nAgent: You absolutely should not have to. I have also flagged your account with a duplicate-charge prevention hold. Our billing system will now block any duplicate processing on your account. You will receive an email confirmation of all three refunds within the hour.\nCustomer: Fine. I will wait for the email and your follow-up. If this is not resolved by Friday I am disputing everything with my bank.\nAgent: Completely understood. You will hear from me before then.'),

  ('CALL-003', '2026-08-19T14:05:00', 'AGT-103', 'CUST-5003',
   'Agent: Welcome to Acme Support, my name is James. What can I do for you today?\nCustomer: Hi James. So my situation is a bit complicated. I am currently on the Basic tier at $9.99 but I also have the legacy Analytics Add-on from 2024 at $4.99. I need more storage for sure, but I want to make sure I am not paying for analytics twice if I upgrade to Enterprise.\nAgent: Great question. Let me look at your account. I can confirm you have the Basic plan plus the legacy Analytics v1 Add-on. If you move to Enterprise at $29.99, that includes Analytics v2 which is a superset of what you currently have. So we would cancel the $4.99 add-on automatically. Your net increase would be $15.01 per month but you would get 500GB storage, full Analytics v2, priority support, and API access.\nCustomer: Hmm, that is more than I expected. Is there anything in between? Like can I just get the storage upgrade without the full Enterprise?\nAgent: We do have a Professional tier at $19.99 that includes 200GB and Analytics v1, but honestly since you already have Analytics v1 through the add-on, the only real benefit for you would be the storage jump from 50GB to 200GB for $10 more per month.\nCustomer: Yeah, that does not seem worth it. Let me just go with Enterprise then. The API access alone would save me time with our integrations.\nAgent: Perfect. I will switch you now. Your legacy add-on will be canceled effective immediately with no pro-rated charge, and Enterprise activates now. You should see all features within the hour. I will also send you our Enterprise onboarding guide which covers the API documentation.\nCustomer: Great. One more thing - will my existing dashboard configurations carry over?\nAgent: Yes, everything migrates seamlessly. Your dashboards, saved reports, and user permissions all remain intact.\nCustomer: Perfect, thanks James. Very helpful.'),

  ('CALL-004', '2026-08-19T16:45:00', 'AGT-101', 'CUST-5004',
   'Agent: Acme Support, how can I help?\nCustomer: My SmartHome Hub has been dropping connection every few hours for the past week. All my connected devices go offline when it happens. I have already tried factory resetting it twice, updated my router firmware, and even moved the hub closer to the router. Nothing works.\nAgent: I appreciate you trying those troubleshooting steps already. Can you tell me which firmware version your hub is running?\nCustomer: It says version 2.3.1.\nAgent: Okay, there is a known issue with 2.3.1 on hubs manufactured between March and June 2026. I can push a firmware update to version 2.4.0 remotely. However, I want to be transparent - the update fixes the disconnection for about 80 percent of affected units. For the remaining 20 percent, it is a hardware issue with the Wi-Fi antenna that requires a replacement unit.\nCustomer: So there is a chance this will not fix it?\nAgent: Correct. What I recommend is this: I will push the update now, which should take about 10 minutes. Then monitor it for 48 hours. If you see even one disconnection in that window, call us back and we will expedite a replacement unit with next-day shipping at no cost. I am noting your account now so you will not need to re-explain the situation.\nCustomer: I guess that is the best you can do. It is just frustrating because I have a home security system tied to this hub and every time it drops, my cameras go offline.\nAgent: That is a serious concern and I completely understand. As a temporary measure, I can enable the hub fallback mode which will keep your security devices on a degraded but functional connection even during the primary signal drops. Would you like me to activate that?\nCustomer: Yes, please do that. At least my cameras will stay up.\nAgent: Done. Fallback mode is active immediately. You will see a yellow indicator light on the hub which is normal. Let me push that firmware update now too.\nCustomer: Okay. I will give it 48 hours. Thanks.'),

  ('CALL-005', '2026-08-20T08:20:00', 'AGT-104', 'CUST-5005',
   'Agent: Thank you for calling Acme Support.\nCustomer: I want to cancel my account entirely. The service has been terrible and I am done.\nAgent: I am sorry to hear that. May I ask what specific issues have been driving this decision?\nCustomer: Where do I start? The app crashes constantly on both my iPhone and my iPad. The sync feature between devices has not worked properly in at least three weeks. And last week my automated reports just stopped generating entirely. I am paying $29.99 a month for a product that does not work.\nAgent: I hear you, and those are all legitimate issues. Let me look into your account to see what is happening. I can see your account is on Enterprise plan, and I notice you are running app version 4.2.1. We released version 4.3.0 two days ago that specifically addresses the crash issue on iOS 19 and the sync failures. The reports issue is separate - it looks like your API token expired on August 14th and needs to be regenerated.\nCustomer: Nobody told me any of this. I have been struggling with these issues for weeks.\nAgent: You are right, and I apologize for the lack of proactive communication. Here is what I can do right now: I can regenerate your API token which should immediately restore your automated reports. For the app issues, I strongly recommend updating to 4.3.0. If those three fixes do not resolve your experience within the next 72 hours, I will process a full refund for this entire billing cycle and handle the cancellation with no further calls needed.\nCustomer: You can fix the reports thing right now?\nAgent: Yes, let me regenerate that token. Done - your reports should resume processing within the next 15 minutes. You will get a test report email shortly. For the app, you will need to manually update through the App Store.\nCustomer: Alright. I will update the app and see if the reports come back. But I am serious - if this is not working by Thursday I am canceling.\nAgent: Completely fair. I have noted your account with a 72-hour follow-up. If you do decide to cancel, you can reference case CAN-2026-1105 and we will process it immediately with a full cycle refund.\nCustomer: Fine. We will see.'),

  ('CALL-006', '2026-08-20T11:05:00', 'AGT-105', 'CUST-5006',
   'Agent: Acme Support, this is Michael. How can I help?\nCustomer: Hi, I have two separate issues I need help with. First, my bill shows a charge for Premium API Access at $49.99 but I never signed up for that. Second, my data export jobs have been failing with a timeout error since Monday.\nAgent: I can definitely help with both. Let me start with the billing concern. Pulling up your account now. I see the Premium API Access was added on August 12th. It looks like this was triggered automatically when your API call volume exceeded 10,000 requests in a single day, which is our auto-upgrade threshold on the Enterprise plan.\nCustomer: Wait, there is an auto-upgrade? I never agreed to that. Our integration had a bug that caused a spike in API calls. That should not trigger charges I did not approve.\nAgent: You are absolutely right to be concerned. Let me check the terms. Under the Enterprise plan section 4.3, auto-upgrades require email notification 24 hours before activation. Let me verify if that notification was sent. I can see the system generated the email but it went to an old address - admin@oldcompany.com. That is clearly not current.\nCustomer: We changed our admin email six months ago. We updated it in account settings.\nAgent: I see the discrepancy - your account contact email is correct but the API notifications are still routed to the old address. This is a known issue with our notification routing for accounts migrated before January 2026. I am going to reverse the $49.99 charge, disable the auto-upgrade for your account, and submit a ticket to fix the notification routing. You will not be charged again without explicit consent.\nCustomer: Good. Now what about my export jobs?\nAgent: Let me look at the export logs. I can see your jobs started timing out on Monday at approximately 2 AM UTC. This correlates with our infrastructure migration that affected batch processing for accounts in the US-East region. The fix was deployed yesterday but requires a manual cache flush on your account. Let me do that now. Your next scheduled export should complete successfully.\nCustomer: Can you trigger a manual export so I can verify?\nAgent: Done. I have kicked off an export of your most recent dataset. You should receive it within 20 minutes. If it completes, your scheduled jobs will resume normally. If it fails, call back and reference ticket TECH-2026-8832 and we will escalate to infrastructure engineering.\nCustomer: Okay, I will watch for that export. Thanks for handling both issues.'),

  ('CALL-007', '2026-08-20T13:30:00', 'AGT-103', 'CUST-5007',
   'Agent: Acme Support, James speaking.\nCustomer: Hi James. I am evaluating Acme for a potential enterprise deployment across our organization, about 200 users. I have some specific questions about your security and compliance capabilities.\nAgent: Happy to help. What specific areas are you looking into?\nCustomer: First, do you support SAML-based SSO with Okta? Second, we need SOC 2 Type II compliance - do you have that? And third, we have strict data residency requirements. Our data must stay within the EU.\nAgent: Great questions. For SSO, yes we fully support SAML 2.0 with Okta, Azure AD, and OneLogin. That is available on our Enterprise plan and above. For SOC 2 Type II, we completed our latest audit in March 2026. I can send you our compliance report through our secure document portal if you provide your work email.\nCustomer: That would be great. What about data residency?\nAgent: For data residency, we currently have data centers in US-East, US-West, and EU-West specifically Ireland. For EU-based storage, you would need our Enterprise Plus plan which starts at $49.99 per user per month for commitments over 100 users. However, I want to be upfront - our EU region does not yet support the real-time analytics feature. That is on our roadmap for Q1 2027.\nCustomer: That is a problem. Real-time analytics is a key requirement for us. Is there any workaround?\nAgent: The data at rest stays in EU, but real-time analytics processing currently routes through US-East with results cached back to EU. If your compliance requirements are specifically about data storage and not processing transit, that may work. However, I would recommend connecting you with our enterprise solutions architect who can do a proper compliance review against your specific requirements.\nCustomer: Yes, please set that up. Can we do a call this week?\nAgent: I will have our solutions team reach out within 24 hours to schedule. I will also send you our security whitepaper and the SOC 2 summary in the meantime. Can you confirm your email?\nCustomer: It is procurement@techcorp.eu.\nAgent: Got it. Expect the documents within the hour and a call from our solutions team by tomorrow.\nCustomer: Perfect. Thanks for the thorough answers, James.'),

  ('CALL-008', '2026-08-20T15:45:00', 'AGT-106', 'CUST-5008',
   'Agent: Acme Support, this is Kevin.\nCustomer: I am going to make this very simple. I have been without service for five days. FIVE DAYS. I run a business that depends on your platform and I have lost actual revenue because of your outage. I have called every single day, been promised callbacks that never come, and I am now considering legal action.\nAgent: I am very sorry to hear about the impact on your business. Let me look at your account right away.\nCustomer: Do not give me a script. I have heard sorry from five different agents. I want to know exactly what is wrong with my account and exactly when it will be fixed. Not approximately, not soon, not we are working on it. An exact time.\nAgent: I understand. Let me pull up the full case history. I can see your account has been flagged with a data migration failure from our August 15th infrastructure update. The migration for your account specifically failed due to a schema conflict with your custom integrations.\nCustomer: So why has nobody fixed it in five days?\nAgent: Looking at the case notes, it was escalated to Tier 3 on day two, but the engineer assigned was handling the broader migration rollback and your specific case fell through the cracks. That is unacceptable and I take full responsibility for this call.\nCustomer: Taking responsibility does not restore my service. My clients are threatening to leave. Every day costs me roughly $2,000 in lost productivity.\nAgent: Here is what I can do right now. I am escalating this to our VP of Engineering with a P1 severity tag. That guarantees a response within 2 hours, not 2 days. I am also initiating a service credit request for the full 5 days of downtime which on your Enterprise Plus plan would be approximately $250. Additionally, for the business impact, I am flagging this for our customer recovery team who handle revenue loss claims up to $10,000.\nCustomer: $250 in credits does not cover $10,000 in lost revenue.\nAgent: You are right, it does not. That is why I mentioned the customer recovery team - they handle claims beyond standard service credits. They will reach out within 24 hours with a dedicated case manager. For the immediate technical issue, expect a call from the engineering VP team within 2 hours. Can I confirm the best number to reach you?\nCustomer: Use this number. And if I do not get that call in 2 hours, I am filing a complaint with the BBB and posting this entire experience on every review platform I can find.\nAgent: I hear you. I am setting a personal timer and if you do not receive that call, I will escalate further myself. Your case reference is P1-2026-0089.')
""")

print("Source table call_transcripts created with 8 records.")

# COMMAND ----------

# DBTITLE 1,Process Call Transcripts with AI Extraction
RESPONSE_FORMAT = '''{
  "type": "json_schema",
  "json_schema": {
    "name": "call_insights",
    "schema": {
      "type": "object",
      "properties": {
        "summary": {"type": "string", "description": "A concise 2-3 sentence summary capturing the reason for the call, the key discussion points, and the final outcome or next steps agreed upon"},
        "issue": {"type": "string", "description": "A single clear sentence describing the primary problem or request raised by the customer, including any relevant product names, error symptoms, or account details mentioned"},
        "resolution": {"type": "string", "description": "A single sentence describing the specific action the agent took or committed to in order to address the reported issue, including any timelines, follow-ups, or escalations"},
        "sentiment": {"type": "string", "enum": ["Positive", "Negative", "Neutral"], "description": "The overall emotional tone of the customer throughout the call, considering their language, frustration level, and satisfaction with the outcome"},
        "category": {"type": "string", "enum": ["Technical Support", "Billing", "Account Management", "Product Inquiry", "Cancellation", "General Inquiry"], "description": "The primary support category that best classifies the nature of the request, based on the core topic discussed regardless of any secondary issues raised"}
      },
      "required": ["summary", "issue", "resolution", "sentiment", "category"],
      "additionalProperties": false
    },
    "strict": true
  }
}'''

spark.sql(f"""
CREATE OR REPLACE TABLE {TABLE_INSIGHTS} AS
SELECT
  call_id, call_date, agent_id, customer_id,
  insights:summary::STRING AS summary,
  insights:issue::STRING AS issue,
  insights:resolution::STRING AS resolution,
  insights:sentiment::STRING AS sentiment,
  insights:category::STRING AS category
FROM (
  SELECT *, ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    CONCAT('Analyze the following customer support call transcript and extract structured information.\n\nTranscript:\n', transcript),
    responseFormat => '{RESPONSE_FORMAT}'
  ) AS insights
  FROM {TABLE_TRANSCRIPTS}
)
""")

# COMMAND ----------

# DBTITLE 1,Verify Output
df = spark.table(TABLE_INSIGHTS)
print(f"Row count: {df.count()}")
display(df)

# COMMAND ----------

# DBTITLE 1,Set MLflow Experiment
import os
import mlflow
from mlflow.entities.trace_location import UnityCatalog

os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = WAREHOUSE_ID

# set_experiment upserts; trace_location binds traces to UC (immutable once set)
try:
    experiment = mlflow.set_experiment(
        experiment_name=EXPERIMENT_NAME,
        trace_location=UnityCatalog(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
        ),
    )
except mlflow.MlflowException:
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

EXPERIMENT_ID = experiment.experiment_id

print(f"Experiment: {experiment.name}")
print(f"Experiment ID: {EXPERIMENT_ID}")
print(f"Trace Location: {experiment.trace_location}")

# COMMAND ----------

# DBTITLE 1,Register Evaluation Dataset
from mlflow.genai.datasets import create_dataset, get_dataset

eval_df = spark.sql(f"""
  SELECT call_id, transcript
  FROM {TABLE_TRANSCRIPTS}
""").toPandas()

try:
    dataset = create_dataset(name=DATASET_NAME)
    print("Created new evaluation dataset.")
except Exception:
    dataset = get_dataset(name=DATASET_NAME)
    print("Loaded existing evaluation dataset.")

# Ground-truth labels for each call
GROUND_TRUTH = {
    "CALL-001": {"expected_sentiment": "Positive", "expected_category": "Technical Support"},
    "CALL-002": {"expected_sentiment": "Negative", "expected_category": "Billing"},
    "CALL-003": {"expected_sentiment": "Positive", "expected_category": "Account Management"},
    "CALL-004": {"expected_sentiment": "Negative", "expected_category": "Technical Support"},
    "CALL-005": {"expected_sentiment": "Negative", "expected_category": "Cancellation"},
    "CALL-006": {"expected_sentiment": "Neutral", "expected_category": "Billing"},
    "CALL-007": {"expected_sentiment": "Positive", "expected_category": "Product Inquiry"},
    "CALL-008": {"expected_sentiment": "Negative", "expected_category": "Technical Support"},
}

records = [
    {
        "inputs": {"transcript": row["transcript"]},
        "expectations": GROUND_TRUTH.get(row["call_id"], {}),
    }
    for _, row in eval_df.iterrows()
]

dataset.merge_records(records)

print(f"Dataset: {dataset.name}")
print(f"Records: {len(records)}")
print(f"Ground truth labels: expected_sentiment, expected_category")

# COMMAND ----------

# DBTITLE 1,Define Scorers
import json
from mlflow.genai.scorers import Guidelines, Safety, scorer
from mlflow.entities import Feedback, AssessmentSource, AssessmentSourceType

# --- Built-in LLM Judges ---
safety = Safety()

print("TYPE 1 - Built-in LLM Judges:")
print(f"  - {safety.name} (predefined, zero-config)")

# --- Custom LLM Judges (Guidelines-based, pass/fail with rationale) ---

faithfulness = Guidelines(
    name="faithfulness",
    guidelines=[
        "Every claim in the summary, issue, and resolution fields must be directly supported by the transcript.",
        "The extraction must not introduce facts, names, numbers, or outcomes not present in the source transcript.",
        "If the transcript is ambiguous about a detail, the extraction should not state it as definitive.",
    ],
)

completeness = Guidelines(
    name="completeness",
    guidelines=[
        "The summary must mention the reason for the call, the key action taken, and the outcome or next step.",
        "The issue field must capture the specific product or service problem, not just a vague description.",
        "The resolution field must include the concrete action (not just that the issue was resolved).",
    ],
)

classification_accuracy = Guidelines(
    name="classification_accuracy",
    guidelines=[
        "The sentiment must accurately reflect the customer emotional tone: Positive if satisfied, Negative if frustrated or angry, Neutral if matter-of-fact.",
        "The category must match the primary topic: Technical Support for device/software issues, Billing for charges/refunds, Account Management for plan changes, Cancellation for cancel requests.",
    ],
)

print("\nTYPE 2 - Custom LLM Judges (Guidelines):")
for s in [faithfulness, completeness, classification_accuracy]:
    print(f"  - {s.name} ({len(s.guidelines)} guidelines)")

# --- Custom LLM-Powered Scorer (numeric 1-5, bring your own prompt) ---

@scorer
def professionalism(inputs, outputs) -> Feedback:
    """Scores extraction professionalism on a 1-5 scale via LLM."""
    judge_prompt = (
        "Rate the professionalism of the following customer support insight extraction "
        "on a scale of 1-5 (1=Poor, 5=Excellent). Consider clarity, conciseness, "
        "objectivity, and appropriate business tone.\n\n"
        f"Summary: {outputs.get('summary', '')}\n"
        f"Issue: {outputs.get('issue', '')}\n"
        f"Resolution: {outputs.get('resolution', '')}\n\n"
        "Respond with ONLY a JSON object: {\"score\": <int>, \"rationale\": \"<text>\"}"
    )
    result = spark.sql(f"""
        SELECT ai_query(
            'databricks-meta-llama-3-3-70b-instruct',
            '{judge_prompt.replace(chr(39), chr(39)+chr(39))}'
        ) AS judgment
    """).collect()[0]["judgment"]

    try:
        parsed = json.loads(result)
        return Feedback(
            value=parsed["score"],
            rationale=parsed.get("rationale", ""),
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id="databricks-meta-llama-3-3-70b-instruct",
            ),
        )
    except (json.JSONDecodeError, KeyError):
        return Feedback(value=3, rationale=f"Could not parse judge response: {result[:200]}")

print("\nTYPE 3 - Custom LLM-Powered Scorer:")
print(f"  - professionalism (numeric 1-5, custom prompt, explicit LLM call)")

# --- Deterministic / Code-Based Scorers (no LLM, fast + reproducible) ---

VALID_SENTIMENTS = {"Positive", "Negative", "Neutral"}
VALID_CATEGORIES = {"Technical Support", "Billing", "Account Management", "Product Inquiry", "Cancellation", "General Inquiry"}

@scorer
def schema_validity(inputs, outputs) -> Feedback:
    """Checks required fields are present and enum values are valid."""
    issues = []
    if outputs.get("sentiment") not in VALID_SENTIMENTS:
        issues.append(f"Invalid sentiment: {outputs.get('sentiment')}")
    if outputs.get("category") not in VALID_CATEGORIES:
        issues.append(f"Invalid category: {outputs.get('category')}")
    for field in ["summary", "issue", "resolution"]:
        if not outputs.get(field):
            issues.append(f"Missing or empty field: {field}")
    passed = len(issues) == 0
    return Feedback(
        value=passed,
        rationale="All fields valid" if passed else "; ".join(issues),
    )

@scorer
def ground_truth_match(inputs, outputs, expectations) -> Feedback:
    """Compares predicted sentiment/category against ground truth labels."""
    issues = []
    if expectations.get("expected_sentiment"):
        if outputs.get("sentiment") != expectations["expected_sentiment"]:
            issues.append(
                f"Sentiment mismatch: predicted '{outputs.get('sentiment')}', "
                f"expected '{expectations['expected_sentiment']}'"
            )
    if expectations.get("expected_category"):
        if outputs.get("category") != expectations["expected_category"]:
            issues.append(
                f"Category mismatch: predicted '{outputs.get('category')}', "
                f"expected '{expectations['expected_category']}'"
            )
    passed = len(issues) == 0
    return Feedback(
        value=passed,
        rationale="Matches ground truth" if passed else "; ".join(issues),
    )

@scorer
def summary_length(inputs, outputs) -> Feedback:
    """Validates summary is 50-500 characters."""
    summary = outputs.get("summary", "")
    length = len(summary)
    passed = 50 <= length <= 500
    return Feedback(
        value=passed,
        rationale=f"Summary length: {length} chars ({'within' if passed else 'outside'} 50-500 range)",
    )

print("\nTYPE 4 - Deterministic / Code-Based Scorers:")
print(f"  - schema_validity (enum + required field validation)")
print(f"  - ground_truth_match (exact match vs human labels)")
print(f"  - summary_length (character count range check)")

ALL_SCORERS = [
    safety,
    faithfulness,
    completeness,
    classification_accuracy,
    professionalism,
    schema_validity,
    ground_truth_match,
    summary_length,
]

print(f"\nTotal scorers: {len(ALL_SCORERS)}")

# Register scorers to the experiment (shareable via get_scorer())
from mlflow.genai import list_scorers, get_scorer

for s in ALL_SCORERS:
    try:
        s.register(experiment_id=EXPERIMENT_ID)
        print(f"  Registered: {s.name}")
    except (ValueError, AttributeError):
        pass

registered = list_scorers(experiment_id=EXPERIMENT_ID)
print(f"\nRegistered scorers: {[s.name for s in registered]}")

# COMMAND ----------

# DBTITLE 1,Multi-Model Evaluation
import pandas as pd
from mlflow.genai.datasets import get_dataset

dataset = get_dataset(name=DATASET_NAME)
print(f"Loaded registered dataset: {dataset.name}")

def make_predict_fn(model_endpoint: str):
    def predict_fn(transcript: str) -> dict:
        import json
        transcript_escaped = transcript.replace("'", "''")
        result = spark.sql(f"""
          SELECT ai_query(
            '{model_endpoint}',
            CONCAT(
              'Analyze the following customer support call transcript and extract structured information.\n\nTranscript:\n',
              '{transcript_escaped}'
            ),
            responseFormat => '{RESPONSE_FORMAT}'
          ) AS insights
        """).collect()[0]["insights"]
        return json.loads(result)
    return predict_fn

comparison_results = {}

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"Evaluating: {model}")
    print(f"{'='*60}")

    model_short_name = model.split(".")[-1]
    run_name = f"call-insights-eval-{model_short_name}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_endpoint": model,
            "task": "call_transcript_extraction",
            "response_format": "json_schema_strict",
            "num_transcripts": dataset.to_df().shape[0],
        })

        result = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=make_predict_fn(model),
            scorers=ALL_SCORERS,
        )
        comparison_results[model] = result.metrics

print("\n\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

comparison_df = pd.DataFrame(comparison_results).T
comparison_df.index.name = "model"
display(comparison_df)