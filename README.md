# Fault Diagnosis Multi-Agent System

A comprehensive, production-ready implementation of a multi-agent fault diagnosis system for telecom networks, featuring:

- **CrewAI** for intelligent agent orchestration
- **LangGraph** for stateful workflow management
- **AWS Bedrock** for RAG-powered knowledge retrieval
- **Hypothesis validation** with specialized telecom validators
- **Deterministic demo mode** for reproducible demonstrations

## 🚀 Quick Start


<script src="https://asciinema.org/a/VOGlgOddp6dyECuz8RhdCmdeU.js" id="asciicast-VOGlgOddp6dyECuz8RhdCmdeU" async="true"></script>


### Prerequisites

- Python 3.9 or higher
- AWS account with Bedrock access (for full RAG functionality)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd fault-diagnosis-multi-agent
```

2. **Run the setup script**
```bash
python setup.py
```

This will:
- Create a virtual environment
- Install all dependencies
- Set up configuration files
- Verify the installation

3. **Configure AWS credentials** (optional for demo mode)
```bash
# Edit .env file with your AWS credentials
cp .env.example .env
# Edit .env with your AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
```

4. **Activate virtual environment**
```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

5. **Run the demo**
```bash
telecom-ops fault-diagnosis --demo --session alpha
```

## 📖 Usage

### Basic Demo Run

Run a deterministic fault diagnosis demo:

```bash
telecom-ops fault-diagnosis --demo --session my_demo
```

This creates a timestamped session directory under `artifacts/sessions/` with:
- Hypothesis analysis and validation results
- HTML/PDF reports
- KPI plots and visualizations
- Complete session transcripts
- Synthetic data for reproducibility

### Command Options

```bash
telecom-ops fault-diagnosis [OPTIONS]

Options:
  --demo              Run scripted demo flow (default)
  --session NAME      Session label for artifact directory
  --seed INT          Deterministic seed (default: 4242)
  --help              Show help message
```

### Sample Output

```
Workflow: Fault Diagnosis (Integrated) | Crew: Planner, Retriever, Reasoner, Reporter
Deterministic seed: 4242 | Session: artifacts/sessions/2024-05-18T15-04-22Z_alpha/

[2024-05-18T15:04:22Z] Fixture Loading      Started
[RAG] Loading 5 fixtures into vector store...
[RAG] Processed FD-ALRT-017: 3 chunks
[Integration] RAG pipeline initialized
[2024-05-18T15:04:23Z] Fixture Loading      Completed

[2024-05-18T15:04:23Z] Alert Processing     Started
[2024-05-18T15:04:24Z] Alert Processing     Completed

[2024-05-18T15:04:24Z] LangGraph Workflow   Started
[LangGraph] Starting fault diagnosis workflow...
[Workflow] Alert Intake - Processing incoming alert...
[Workflow] Evidence Gathering - Collecting relevant data...
[CrewAI] Starting Hypothesis Generation phase...
[2024-05-18T15:04:28Z] LangGraph Workflow   Completed

[2024-05-18T15:04:28Z] Enhanced Validation  Started
[Validator] H1: Grounded (confidence: 0.85)
[Validator] H2: Retry (confidence: 0.62)
[Validator] H3: Rejected (confidence: 0.34)
[2024-05-18T15:04:29Z] Enhanced Validation  Completed

Demo complete. Artifacts are available under: artifacts/sessions/2024-05-18T15-04-22Z_alpha/
```

## 🏗️ Architecture

### System Components

1. **CrewAI Agents**
   - **NOC Sentinel Planner**: Alert triage and workflow coordination
   - **Core Network Analyst**: Evidence retrieval and knowledge search
   - **Hypothesis Chair**: Root cause analysis and hypothesis generation
   - **Postmortem Writer**: Report generation and documentation

2. **LangGraph Workflow**
   - Stateful workflow orchestration
   - Decision routing based on confidence thresholds
   - Escalation handling for low-confidence scenarios
   - Guardrailed validation loops

3. **Bedrock RAG Pipeline**
   - AWS Bedrock embeddings for semantic search
   - ChromaDB vector storage with persistence
   - Grounded retrieval with proper citations
   - Fallback to local text search if Bedrock unavailable

4. **Hypothesis Validators**
   - **Traffic Probe Validator**: Validates network performance claims
   - **Config Diff Validator**: Checks configuration-related hypotheses
   - **Topology Validator**: Verifies network topology references

### Workflow Stages

1. **Alert Intake** → Parse and classify incoming alerts
2. **Routing Decision** → Route to appropriate specialist workflows
3. **Evidence Gathering** → Collect logs, KPIs, and historical data
4. **Hypothesis Generation** → Generate ranked root cause hypotheses
5. **Guardrailed Validation** → Validate hypotheses with multiple validators
6. **Resolution Decision** → Decide between remediation and escalation
7. **Remediation Planning** → Generate actionable remediation steps
8. **Post-Mortem Documentation** → Create reports and update knowledge base

## 🧪 Testing

### Smoke Test

Verify the complete system works:

```bash
python scripts/smoke_fault_diagnosis.py
```

### Manual Testing

Test individual components:

```bash
# Test CLI directly
python -m telecom_ops.cli fault-diagnosis --demo --session test

# Test with specific seed
python -m telecom_ops.cli fault-diagnosis --demo --seed 1234 --session reproducible
```

## 📁 Project Structure

```
fault-diagnosis-multi-agent/
├── telecom_ops/                    # Main package
│   ├── cli.py                     # CLI entry point
│   ├── fault_diagnosis/           # Fault diagnosis workflow
│   │   ├── agents.py              # CrewAI agent definitions
│   │   ├── tasks.py               # CrewAI task definitions
│   │   ├── crew_orchestration.py # CrewAI crew setup
│   │   ├── langgraph_workflow.py  # LangGraph state machine
│   │   ├── rag_pipeline.py        # Bedrock RAG implementation
│   │   ├── validators.py          # Hypothesis validation logic
│   │   ├── integrated_workflow.py # Main integrated workflow
│   │   ├── artifacts.py           # Artifact generation
│   │   └── data.py                # Fixture data loading
│   └── shared/                    # Shared utilities
│       ├── console.py             # Console output helpers
│       └── files.py               # File system utilities
├── data/fault_diagnosis/fixtures/ # Sample fixture data
├── scripts/                       # Utility scripts
├── artifacts/                     # Generated artifacts
│   ├── sessions/                  # Session-specific artifacts
│   ├── plots/                     # KPI visualizations
│   └── data/                      # Synthetic data exports
├── .env.example                   # Environment configuration template
├── pyproject.toml                 # Python package configuration
├── setup.py                       # Setup script
└── README.md                      # This file
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# AWS Configuration
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here

# Bedrock Models
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v1
BEDROCK_LLM_MODEL=anthropic.claude-3-sonnet-20240229-v1:0

# Workflow Settings
FAULT_DIAGNOSIS_CONFIDENCE_THRESHOLD=0.7
FAULT_DIAGNOSIS_ENABLE_RAG=true
FAULT_DIAGNOSIS_ENABLE_CREWAI=true
FAULT_DIAGNOSIS_ENABLE_LANGGRAPH=true
FAULT_DIAGNOSIS_ENABLE_VALIDATORS=true
```

### Component Toggle

You can selectively enable/disable components for different deployment scenarios:

- **Demo Mode**: All components enabled with fallbacks
- **Production Mode**: Full AWS Bedrock integration
- **Offline Mode**: CrewAI + LangGraph only (no RAG)
- **Basic Mode**: Simple workflow without advanced validation

## 📊 Output Artifacts

Each session generates comprehensive artifacts:

### Reports
- `fault_diagnosis_report.html` - Stakeholder-ready HTML report
- `fault_diagnosis_report.pdf` - PDF version for distribution
- `hypothesis_board.md` - Detailed hypothesis analysis
- `remediation_plan.md` - Step-by-step remediation guide

### Data Files
- `alert_context.json` - Processed alert data
- `rag_index.json` - RAG retrieval results with citations
- `validation_trace.json` - Validator results and verdicts
- `manifest.json` - Complete list of generated artifacts

### Session Logs
- `session.log` - Complete session transcript
- Synthetic data bundle for reproducibility
- KPI trend visualizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the smoke test (`python scripts/smoke_fault_diagnosis.py`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the Proprietary License - see the pyproject.toml file for details.

## 🆘 Troubleshooting

### Common Issues

**AWS Credentials Not Found**
```bash
# Set AWS credentials in .env file or use AWS CLI
aws configure
# Or set environment variables directly
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

**Bedrock Access Denied**
- Ensure your AWS account has Bedrock access enabled
- Check that your region supports the Bedrock models you're using
- Verify IAM permissions include Bedrock actions

**Import Errors**
```bash
# Reinstall in development mode
pip install -e .
# Or activate virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

**No Artifacts Generated**
- Check that the `artifacts/` directory is writable
- Verify the session completed without errors
- Look for error messages in the console output

### Getting Help

- Check the smoke test: `python scripts/smoke_fault_diagnosis.py`
- Review session logs in `artifacts/sessions/<timestamp>/session.log`
- Enable verbose mode in `.env`: `FAULT_DIAGNOSIS_VERBOSE=true`
- For AWS/Bedrock issues, check CloudTrail logs

For more detailed information, see `Fault_Diagnosis.md`.
