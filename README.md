# 🔎 ResearchVault

**An evidence-grounded research assistant for exploring and understanding research papers.**

ResearchVault is a RAG-based application that lets users ask questions about research PDFs and inspect the retrieved passages behind generated answers. The project is being developed toward a broader research workspace for organizing and reasoning across collections of papers.

> **Project status:** The core RAG pipeline and retrieval evaluation are implemented. Multi-paper workflows and the broader research-workspace features are planned. This README distinguishes implemented capabilities from future work.

---

## Table of contents

- [Problem](#problem)
- [What ResearchVault does](#what-researchvault-does)
- [Current capabilities](#current-capabilities)
- [Architecture](#architecture)
- [Evaluation](#evaluation)
- [Technology stack](#technology-stack)
- [Getting started](#getting-started)
- [Project structure](#project-structure)
- [Responsible AI and limitations](#responsible-ai-and-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Problem

Working with research papers often means searching through long PDFs, locating supporting evidence, remembering what each paper contributes, and comparing findings across sources. AI-generated summaries and answers can also be difficult to trust when citations are missing or do not support the claims.

ResearchVault explores a more transparent workflow: retrieve relevant passages from research documents, generate answers grounded in those passages, and make the source material inspectable.

---

## What ResearchVault does

At its core, ResearchVault follows a retrieval-augmented generation (RAG) workflow:

1. Load research PDFs.
2. Extract text and document metadata.
3. Split the documents into chunks.
4. Generate vector embeddings for the chunks.
5. Store the vectors in Chroma.
6. Retrieve relevant chunks for a user question.
7. Generate an answer using the retrieved evidence.
8. Return source information so users can inspect where the answer came from.

Conversational query rewriting converts context-dependent follow-up questions into standalone retrieval queries.

---

## Current capabilities

- **PDF ingestion:** Load research papers using LangChain PDF-loading components.
- **Metadata-aware chunking:** Split documents while retaining source and page metadata, with unique chunk IDs.
- **Embedding generation:** Generate document embeddings using Hugging Face embeddings.
- **Vector storage:** Persist and query embeddings with Chroma.
- **Semantic retrieval:** Retrieve relevant document chunks for a question, optionally filtered to one paper.
- **Grounded answer generation:** Use retrieved passages as the evidence base for LLM responses, with per-claim paper and page citations.
- **Source extraction:** Return document identifiers, page information, and relevant passages.
- **Conversational retrieval:** Rewrite context-dependent follow-up questions before retrieval.
- **Evaluation dashboard:** Display retrieval metrics and per-question retrieval results.

> Implementation maturity varies by component. The application is still being developed and tested; these capabilities should not be interpreted as a claim of production readiness.

---

## Architecture

```
                 RESEARCH PDFs
                       |
                       v
                PDF INGESTION
          Text extraction + metadata
                       |
                       v
                   CHUNKING
          Text chunks + page metadata
                       |
                       v
               EMBEDDING MODEL
                       |
                       v
             CHROMA VECTOR STORE
                       |
                       |                    USER QUESTION
                       |                          |
                       |                          v
                       |                   QUERY REWRITING
                       |                   (for follow-ups)
                       |                          |
                       +----------+---------------+
                                  |
                                  v
                              RETRIEVAL
                         Top-K relevant chunks
                                  |
                                  v
                         RETRIEVED EVIDENCE
                          Text + source + page
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
              GROUNDED LLM               SOURCE EXTRACTION
               GENERATION                Document + page +
                    |                    relevant passage
                    +-------------+-------------+
                                  |
                                  v
                          ANSWER + SOURCES
                                  |
                                  v
                           EVALUATION LOOP
                   Retrieval metrics and answer
                     evaluation experiments
```

### Evidence-first response design

ResearchVault keeps generated answers connected to retrieved source material. A response makes it possible to inspect:

- Which document was retrieved.
- Which page contains the passage.
- Which passage was supplied as evidence.

The system acknowledges when the available documents do not contain enough information rather than filling gaps with unsupported claims.

---

## Evaluation

A retrieval evaluation run was performed on 58 questions covering answerable, unanswerable, multi-paper, trick, and ambiguous cases. The dashboard reports these aggregate results:

| Metric | Result |
|---|---|
| Recall@10 | 0.9511 |
| MRR | 0.8240 |
| nDCG@10 | 0.8346 |
| Questions evaluated | 58 |

The dashboard also displays per-question retrieval results, including recall, reciprocal rank, and nDCG.

> **These figures describe the retrieval evaluation run only.** They are not answer accuracy, factuality, or citation-correctness scores. Answer evaluation using an LLM-as-a-judge workflow is part of the evaluation dashboard; results will be reported separately once that evaluation has been run and inspected.

---

## Technology stack

| Component | Technology |
|---|---|
| Language | Python |
| RAG framework | LangChain |
| Vector database | Chroma |
| Embeddings | Hugging Face (`all-MiniLM-L6-v2`) |
| Language model | OpenAI API |
| PDF processing | LangChain PDF loaders (PyMuPDF) |
| Text splitting | `RecursiveCharacterTextSplitter` |
| Interface / evaluation dashboard | Gradio |
| Development | VS Code, Jupyter, Git, GitHub |

Exact package versions and model configuration depend on the project environment.

---

## Getting started

### Prerequisites

- Python 3.11
- Git
- An OpenAI API key
- The project dependencies (see `requirements.txt`)

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ResearchVault
```

### 2. Create and activate an environment

```bash
conda create -n researchvault python=3.11
conda activate researchvault
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

Never commit `.env` or expose API keys.

### 5. Add research PDFs

Place PDFs in the document directory read by `src/ingestion.py` (for example `data/papers/`). PDFs are not committed to the repository.

### 6. Build the vector store

Ingest, chunk, embed, and persist the papers to `chroma_db/`:

```bash
python -m src.vectorstore
```

### 7. Run

```bash
# Chat in the terminal: shows the rewritten query, answer, and sources
python -m src.generation

# Inspect retrieval only: chunks, scores, pages, chunk IDs
python -m src.retrieval_ui
```

To launch the main application and evaluation dashboard, run `python app.py`.

---

## Project structure

```
ResearchVault/
├── data/
│   ├── evaluations.csv      # Evaluation question set
│   └── papers/              # Local research PDFs (not committed)
├── notebooks/               # Experiments and exploration
├── src/
│   ├── ingestion.py         # PDF loading and metadata
│   ├── chunking.py          # Document splitting and chunk IDs
│   ├── embeddings.py        # Embedding configuration
│   ├── vectorstore.py       # Chroma setup and persistence
│   ├── retrieval.py         # Similarity retrieval
│   ├── generation.py        # Query rewriting and grounded answers
│   ├── prompts.py           # Prompt templates
│   ├── retrieval_lab.py     # Batch retrieval inspection
│   └── retrieval_ui.py      # Retrieval Lab (Gradio)
├── app.py                   # Application entry point
├── requirements.txt
├── .env                     # Local secrets; never commit
└── .gitignore
```

---

## Responsible AI and limitations

ResearchVault is intended to support researchers, not replace their judgment or close reading of original papers.

**Principles**

- **Source transparency:** Retrieved documents and passages are inspectable.
- **Grounding:** Evidence from the selected papers is preferred over unsupported model knowledge.
- **Uncertainty:** The system indicates when the retrieved evidence is insufficient.
- **Careful interpretation:** Explicit statements from a paper are distinguished from interpretations of its findings.
- **Verification:** Citations must be checked against the cited passage; the presence of a citation alone does not prove a claim.
- **Human oversight:** Users should consult the original paper before relying on important conclusions.

**Known limitations**

- Dependence on PDF text-extraction quality.
- Sensitivity to chunking choices.
- Embedding and retrieval quality, including multi-paper questions where one paper can dominate the top-K results.
- Prompt behaviour and the language model's ability to interpret retrieved passages.
- Retrieval metrics do not by themselves establish answer correctness.
- No automated tests yet; dependency pinning and configuration management are still in progress.

---

## Roadmap

The following items are planned; they are not yet implemented.

### Next

1. **Paper library:** upload and manage papers, and record why each paper was saved.
2. **Multi-paper Q&A:** question answering across selected papers with per-paper source attribution.
3. **Methodology comparison:** evidence-grounded comparison of methods, datasets, results, and limitations, with every claim tied to its paper and page.
4. **Engineering hardening:** automated tests, pinned dependencies, and cleaner configuration.

### Later

- **Stronger evidence experience:** clickable source passages and PDF page navigation, claim-to-evidence mapping, better handling of missing or conflicting evidence.
- **Research connections:** relationships between papers, claims, methods, and findings; a visual evidence graph; carefully qualified detection of possible disagreements between papers.
- **Advanced research assistance:** literature-review support, research trend exploration, and research-gap exploration with clear separation between evidence and hypotheses.
- **Deployment:** robust document update and deletion workflows, logging, error handling, and user/session management as needed.

---

## Contributing

ResearchVault is an evolving learning and engineering project. Suggestions, issue reports, and constructive feedback are welcome.

When reporting a problem, include:

- What you were trying to do.
- The steps to reproduce it.
- The expected behaviour.
- The actual behaviour and relevant error output.

---

*ResearchVault: turning research papers into searchable, inspectable evidence.*