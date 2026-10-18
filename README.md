# InsightForgeEnterprise
### An Evidence-Grounded GenAI Analytics Platform with RAG Evaluation and Release Quality Gates

InsightForgeEnterprise is a Python-based GenAI analytics platform that combines Retrieval-Augmented Generation (RAG), document retrieval, evidence-grounded answer generation, answerability checks, and automated evaluation. It is designed to explore how LLM-powered analytics systems can retrieve relevant business information, support answers with source citations, and abstain when available evidence is insufficient.

The project uses synthetic customer, campaign, transaction, and feedback data to demonstrate an end-to-end analytics and responsible AI workflow.

## Key Features

- **Retrieval-Augmented Generation (RAG):** Retrieves relevant passages from a curated business knowledge base to support question answering.
- **Evidence-Grounded Answers:** Associates generated answers with source documents to improve traceability.
- **Answerability and Abstention:** Evaluates whether the available evidence is sufficient to answer a question and supports abstaining when it is not.
- **RAG Evaluation:** Measures retrieval hit rate, Mean Reciprocal Rank (MRR), and expected-source retrieval.
- **Answer Quality Evaluation:** Evaluates answerability accuracy, abstention recall, source citation coverage, grounded citations, and expected-term coverage.
- **Responsible AI:** Includes documentation and evaluation scenarios related to privacy, bias, and responsible analytics.
- **Business Analytics:** Uses synthetic customer, campaign, transaction, and feedback datasets to represent practical analytics questions.
- **Automated Release Quality Gates:** Generates a release report that compares evaluation metrics against configured quality thresholds.
- **API-Based Architecture:** Exposes application functionality through FastAPI.
- **Automated Testing and Code Quality:** Uses pytest and Ruff for regression testing, linting, and formatting checks.

## Architecture

```text
Synthetic Business Data
          |
          v
     Data Ingestion
          |
          v
  Curated Knowledge Base
          |
          v
  Document Chunking
          |
          v
    RAG Retrieval
          |
          v
 Evidence-Grounded Answering
          |
          +--------------------+
          |                    |
          v                    v
  Source Citations      Answerability Check
          |                    |
          +---------+----------+
                    |
                    v
          Evaluation Framework
                    |
                    v
       Release Quality Report
```

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| API | FastAPI |
| Data Processing | Pandas, synthetic datasets |
| GenAI Architecture | Retrieval-Augmented Generation (RAG) |
| Retrieval | Document chunking, relevance ranking, top-k retrieval |
| Evaluation | Hit Rate@K, MRR, answerability, abstention, citation and term coverage |
| Testing | pytest |
| Code Quality | Ruff |
| Documentation | Markdown |
| Version Control | Git and GitHub |

*The specific retrieval backend and generation strategy are determined by the project configuration and implementation. This repository should not be interpreted as evidence of a deployed commercial LLM service or production-scale vector database.*

## Project Structure

```text
InsightForgeEnterprise/
├── data/
│   ├── raw/                 # Synthetic business datasets and evaluation inputs
│   └── documents/           # Curated RAG knowledge-base documents
├── reports/
│   └── rag/                 # RAG evaluation and quality reports
├── scripts/
│   ├── generate_data.py
│   ├── generate_quality_summary.py
│   ├── generate_release_report.py
│   └── smoke_test_api.py
├── src/
│   └── insightforge/
│       ├── api/
│       ├── evaluation/
│       ├── rag/
│       └── main.py
├── tests/
│   ├── fixtures/
│   └── unit/
├── .gitignore
├── pyproject.toml
└── README.md
```

*Directory contents may evolve as the project develops.*

## Getting Started

### Prerequisites

- Python 3.11 or a compatible version supported by the project configuration
- Git
- Windows, macOS, or Linux

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/InsightForgeEnterprise.git
cd InsightForgeEnterprise
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Python 3.11 is unavailable, use a Python version supported by `pyproject.toml`.

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If the project does not define a `dev` optional dependency group, install the dependencies using the method specified in `pyproject.toml`.

### 4. Configure the Environment

Review the project configuration and `.env.example`, if provided. Add required local settings to `.env` and never commit credentials or API keys.

### 5. Generate Synthetic Data

```powershell
python scripts\generate_data.py
```

This generates synthetic business datasets for local experimentation. Generated files and evaluation inputs may vary with the project configuration.

### 6. Start the API

```powershell
uvicorn insightforge.main:app --reload
```

Open:

- API base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

### 7. Run the Tests

```powershell
python -m pytest -q
```

### 8. Run Code Quality Checks

```powershell
ruff check .
ruff format --check .
```

### 9. Generate the Release Quality Report

```powershell
python scripts\generate_release_report.py
```

Review the generated report and its configured thresholds before treating a release as passing.

## Evaluation Framework

InsightForgeEnterprise evaluates several complementary aspects of retrieval and answer quality.

| Metric | Purpose |
|---|---|
| Hit Rate@K | Checks whether an expected source appears among the top-k retrieved results |
| Mean Reciprocal Rank (MRR) | Rewards retrieving the expected source near the top of the results |
| Answerability Accuracy | Measures answerable versus unanswerable classification accuracy |
| Abstention Recall | Measures whether the system abstains on unanswerable evaluation cases |
| Expected-Source Citation Rate | Checks whether answers cite the expected source |
| Grounded Citation Rate | Checks whether cited evidence supports the generated answer under the evaluator's rules |
| Expected-Term Coverage | Measures coverage of predefined terms relevant to an evaluation question |

### Latest Local Validation

The latest recorded local run completed with:

- **58 automated tests passing**
- **Ruff lint checks passing**
- **Ruff formatting checks passing**
- **Release quality gate: PASS**
- Retrieval Hit Rate@3: **100%**
- Mean Reciprocal Rank: **1.000**
- Answerability Accuracy: **100%**
- Abstention Recall on unanswerable cases: **100%**
- Expected-Source Citation Rate: **83.3%**
- Mean Expected-Term Coverage: **91.7%**

These figures describe a small, hand-authored evaluation suite over synthetic data and local documents. They are not estimates of production performance, real-world factual accuracy, or generalization to unseen business questions. Results can change with code, configuration, dependencies, and evaluation data.

## Responsible AI Considerations

The project explores several safeguards relevant to business-facing GenAI applications:

- Ground answers in retrieved evidence and expose source citations.
- Abstain when the evidence is insufficient to support a reliable answer.
- Evaluate citation behavior and answerability rather than relying on retrieval scores alone.
- Include privacy and bias considerations in the business knowledge base.
- Use synthetic data for reproducible development and demonstration.

These mechanisms are prototypes and evaluation exercises, not a certification of regulatory compliance or a guarantee that all generated answers are correct.

## Engineering Practices

- Modular application and evaluation components
- Reproducible local development workflow
- Automated regression tests
- Linting and formatting checks
- Metric-based release quality gates
- Explicit documentation of evaluation limitations

## Future Improvements

- Add a larger, independently reviewed evaluation dataset.
- Support configurable embedding models and vector retrieval backends.
- Introduce systematic comparisons between retrieval and ranking strategies.
- Expand adversarial, hallucination, and citation-faithfulness tests.
- Add latency, throughput, and resource-usage benchmarks.
- Add CI workflows for automated testing and release reports.
- Explore deployment on Google Cloud with appropriate security, monitoring, and cost controls.

## Intended Use

InsightForgeEnterprise is an educational and portfolio project demonstrating practical GenAI engineering, RAG evaluation, API development, testing, and responsible analytics concepts. It is not intended to make autonomous business decisions or to be used as a production decision system without further validation and safeguards.

## Author

**Yash Kalbhile**

GitHub: https://github.com/yash45615

LinkedIn: https://www.linkedin.com/in/yash-kalbhile-697727231/

## License

Choose and add a license file before granting others permission to reuse or redistribute the project. If no license is included, the default copyright restrictions apply.
