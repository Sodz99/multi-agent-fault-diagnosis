# Fault Diagnosis Demo Guide

## Overview
- Scripted, demo-ready multi-agent assistant that showcases deterministic telecom fault diagnosis without requiring live production access.
- Illustrates how CrewAI, LangGraph, and an Amazon Bedrock RAG stack collaborate to analyze synthesized alerts, generate grounded hypotheses, and deliver remediation guidance.
- Emphasizes reproducibility: every run replays the same fixtures, seed, and guardrails so stakeholders can focus on reasoning quality instead of environment drift.

## Running the Demo
### Quick Start
```
$ telecom-ops fault-diagnosis --demo --session alpha
```
- Launches the `fault-diagnosis` workflow with a deterministic seed and creates a timestamped session folder for artifacts.
- CLI banner prints workflow name, Crew roster, random seed, and the relative path where artifacts will be written.
- Fixture loader spins up immediately, echoing each fixture ID as embeddings are written to the Bedrock-backed vector store.

### What Appears in the CLI
- Line-by-line transcript with speaker tags so you can watch Planner, Retriever, Reasoner, and Reporter agents hand off tasks.
- Guardrail verdicts, grounding citations, and confidence bars inline with each hypothesis update.
- Validator commentary (e.g., `traffic_probe_agent`, `config_diff_checker`) streaming beside the hypothesis they validate or reject.
- Final manifest listing reports, transcripts, plots, and synthetic data artifacts saved under the active session directory.

### Sample Transcript Snippet
```
Workflow: Fault Diagnosis | Crew: Planner, Retriever, Reasoner, Reporter
Deterministic seed: 4242 | Artifacts: artifacts/sessions/2024-05-18T15-04-22Z/

[Planner] Seeding alert context from fixture FD-ALRT-017
[Retriever] Indexed fixtures → runbook_rf_002.md, incident_core_011.json
[Reasoner] Hypothesis 1 (0.82 confidence) grounded ✔ via incident_core_011.json
[Validator: traffic_probe_agent] Packet loss spike confirmed (p95 latency 180ms)
[Reporter] Drafting remediation steps → artifacts/.../fault_diagnosis_report.html
```

## Stage-by-Stage Walkthrough
| Stage | CLI Experience | Generated Artifacts |
| --- | --- | --- |
| Launch | Workflow banner, Crew roster, artifact path, deterministic seed | `artifacts/sessions/<ts>/session.log` |
| Fixture Replay | Progress spinner with fixture IDs and vector-store load confirmation | `artifacts/sessions/<ts>/rag_index.json` |
| Alert Intake | Markdown alert summary with severity, KPIs, and impacted assets | `artifacts/sessions/<ts>/alert_context.json` |
| Evidence Sweep | Status rows for each tool (`Started → Completed`) with saved file paths | `log_bundle.json`, `topology_view.json` |
| Hypothesis Board | Ranked hypotheses, confidence bars, guardrail verdict (`Grounded ✔/Retry ✖`) | `hypothesis_board.md` |
| Validation Loop | Streaming validator output showing pass/fail and re-retrieval messages | `validation_trace.json` |
| Resolution Snapshot | Remediation steps, MTTR estimate, escalation status | `remediation_plan.md` |
| Wrap-Up | Manifest of reports, plots, transcripts, synthetic inputs | Reports, KPI plots, JSON transcripts |

## CrewAI Orchestration
- CrewAI runs four primary roles, each wrapping a telecom persona for domain-specific prompts:
  - **Planner (`noc_sentinel`)** aligns on objectives, seeds LangGraph shared state, and schedules downstream tools.
  - **Retriever (`core_network_analyst`)** calls the Bedrock RAG pipeline, fetching telecom runbooks, incident histories, and fixture artifacts from the vector store.
  - **Reasoner (`hypothesis_chair`)** synthesizes evidence, drafts hypotheses, and coordinates guardrailed validation loops.
  - **Reporter (`postmortem_writer`)** packages summaries, remediation guidance, and stakeholder-ready artifacts.
- Validator subgraphs (e.g., `traffic_probe_agent`, `config_diff_checker`) enforce grounded reasoning before hypotheses advance.
- Shared state lives in LangGraph so each role can read/write deterministic slots (`alert_context`, `log_bundle`, `hypothesis_board`, etc.).

## Deterministic Workflow
1. **Alert Intake** – Planner parses alarms, classifies severity, and writes summary metrics to `alert_context` in shared state.
2. **Routing** – LangGraph decision nodes branch by asset type/priority, pre-loading RAG snippets for specialists like `rf_specialist`.
3. **Evidence Gathering** – Tool subgraphs collect logs, topology snapshots, and config fetches, persisting outputs (`log_bundle`, `topology_view`).
4. **Hypothesis Generation** – Reasoner drafts three ranked hypotheses, attaches required validations, and emits `hypothesis_board`.
5. **Guardrailed Validation** – Validators run sequentially; any ungrounded claim triggers re-retrieval with narrower filters.
6. **Resolution Decision** – Aggregator node compiles validated hypotheses, MTTR, and impact; low confidence routes to `escalation_queue`.
7. **Remediation Plan** – Reporter assembles change steps, maintenance windows, and notifications; `change_manager` persona checks compliance.
8. **Post-Mortem Trigger** – Final package (report, citations, transcripts) is prepared for knowledge-base updates and stakeholder delivery.

## Bedrock RAG & Guardrails
- Vector store (OpenSearch or in-memory fallback) populates from `data/fault_diagnosis/fixtures/` at launch.
- Retriever obtains Bedrock embeddings, executes similarity search, and attaches citation metadata to every response.
- Guardrails reject answers lacking citations, prompting targeted re-retrieval and preventing hallucinated remediation steps.
- Vector store snapshots version alongside the CLI binary so demos rehydrate identical fixture sets over time.

## Session Artifacts & Reporting
- HTML/PDF report generator captures evidence tables, hypothesis board, remediation plan, and KPI plots under `artifacts/sessions/<ts>/`.
- Plotting utility writes KPI visuals (traffic load, error rates, MTTR trends) to `artifacts/plots/` for embedding in reports.
- CLI serializes transcripts, guardrail outcomes, validation traces, and synthetic data bundles in JSON/Markdown for later review.
- Manifest printed at wrap-up lists everything produced so stakeholders can open artifacts immediately.

## Data Management
- Synthetic session data is generated at runtime and saved as `artifacts/data/fault_diagnosis_<timestamp>.json` for reproducibility.
- Fixtures (alarms, KPIs, topology snippets) live under `data/fault_diagnosis/fixtures/` and feed the vector store loader.
- Reports append fixture IDs and RAG document hashes per hypothesis, enabling auditors to verify each claim quickly.

## CLI & Package Layout
- Package root `telecom_ops/` contains CrewAI tasks, LangGraph orchestration, and Bedrock utilities; `telecom_ops/shared/` holds logging, plotting, and reporting helpers.
- CLI entry point `telecom-ops` exposes the `fault-diagnosis` workflow flag, deterministic seed option, and session directory creation.
- Workflow modules remain importable for notebooks or integration tests, while the CLI stitches them together for the scripted demo.

## Implementation Roadmap
1. **CLI Experience** – Scaffold entry point, launch banner, fixture loader spinner, transcript streaming, and artifact manifest.
2. **CrewAI Modules** – Define roles, telecom personas, and validator subgraphs with deterministic prompts and hand-offs.
3. **LangGraph Flow** – Encode the deterministic graph, guardrail checks, and escalation routing logic.
4. **Bedrock RAG** – Implement vector store loader, embedding pipeline, similarity search, and retrieval snapshot persistence.
5. **Artifact Generation** – Build HTML/PDF report builder, plotting hooks, and serialization for transcripts/validation traces.
6. **Data Utilities** – Create synthetic data generator and support libraries for KPI plots and fixture metadata.
7. **Testing** – Add smoke tests or notebooks to replay the CLI flow, verifying deterministic ordering and artifact emission.
8. **Documentation** – Record manual verification steps and demo scripts for stakeholder walkthroughs.



📁 Current folder (manually rename to "fault-diagnosis-agent")
  ├── run_demo.bat                       # ← Updated for new structure
  ├── README.md
  ├── requirements.txt
  ├── .env & .env.example
  ├── 📁 docs/                          # Documentation
  │   └── Fault_Diagnosis.md
  ├── 📁 src/                           # Source code (Python standard)
  │   └── 📁 fault_diagnosis/           # Main package
  │       ├── cli.py                    # Command interface
  │       ├── 📁 agents/                # CrewAI agents
  │       ├── 📁 workflow/              # LangGraph workflow  
  │       ├── 📁 rag/                   # RAG pipeline
  │       ├── 📁 validation/            # Hypothesis validators
  │       ├── 📁 data/                  # Data handling
  │       ├── 📁 artifacts/             # Artifact generation
  │       └── 📁 shared/                # Utilities
  ├── 📁 fixtures/                      # Telecom data (flattened)
  └── 📁 outputs/                       # Generated artifacts
      ├── sessions/                     # Session outputs  
      ├── data/                         # Export data
      └── vector_store/                 # ChromaDB cache