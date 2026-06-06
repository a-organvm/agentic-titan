# Agentic Titan

[![CI](https://github.com/organvm-iv-taxis/agentic-titan/actions/workflows/ci.yml/badge.svg)](https://github.com/organvm-iv-taxis/agentic-titan/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-pending-lightgrey)](https://github.com/organvm-iv-taxis/agentic-titan)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/organvm-iv-taxis/agentic-titan/blob/main/LICENSE)
[![Organ IV](https://img.shields.io/badge/Organ-IV%20Taxis-10B981)](https://github.com/organvm-iv-taxis)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](https://github.com/organvm-iv-taxis/agentic-titan)
[![Python](https://img.shields.io/badge/lang-Python-informational)](https://github.com/organvm-iv-taxis/agentic-titan)


[![ORGAN-IV: Orchestration](https://img.shields.io/badge/ORGAN--IV-Orchestration-e65100?style=flat-square)](https://github.com/organvm-iv-taxis)
[![Tests](https://img.shields.io/badge/tests-1312%20passing-brightgreen?style=flat-square)]()
[![Python](https://img.shields.io/badge/python-%E2%89%A53.11-blue?style=flat-square)]()
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--hardened-e65100?style=flat-square)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)](deploy/)
[![Redis](https://img.shields.io/badge/Redis-state%20%26%20events-DC382D?style=flat-square&logo=redis)]()
[![Celery](https://img.shields.io/badge/Celery-task%20queue-37814A?style=flat-square&logo=celery)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-audit%20log-4169E1?style=flat-square&logo=postgresql)]()
[![Grafana](https://img.shields.io/badge/Grafana-6%20dashboards-F46800?style=flat-square&logo=grafana)](deploy/grafana/)
[![Helm](https://img.shields.io/badge/Helm-k8s%20charts-0F1689?style=flat-square&logo=helm)](deploy/helm/)

> A polymorphic, model-agnostic multi-agent orchestration framework with nine topology patterns, 22 agent archetypes, and production-grade safety infrastructure — from two agents on a laptop to 100+ agents across Firecracker microVMs.

[Problem Statement](#problem-statement) | [Core Architecture](#core-architecture) | [Key Concepts](#key-concepts) | [Installation & Setup](#installation--setup) | [Quick Start](#quick-start) | [Working Examples](#working-examples) | [Testing & Validation](#testing--validation) | [Getting Started](docs/getting-started.md) | [Downstream Implementation](#downstream-implementation) | [Cross-References](#cross-references) | [Contributing](#contributing) | [License & Author](#license--author)

---

## Problem Statement

Multi-agent AI systems face a coordination problem that existing frameworks consistently undersolve. The standard approach treats topology as a fixed architectural decision: you choose a pipeline, or a hierarchy, or a swarm, and your agents live within that structure for the duration of the task. This works when the problem is well-characterized in advance. It fails — often silently — when the nature of the work shifts mid-execution.

Consider a team of agents tasked with researching a technical question and producing a report. The research phase benefits from a swarm topology: agents explore independently, share findings through a common memory, and surface relevant patterns without bottleneck. But the synthesis phase demands a pipeline — raw findings must be filtered, structured, reviewed, and assembled in sequence. And if the research reveals contradictions, the team needs a consensus mechanism (ring topology with voting) before synthesis can proceed. A fixed topology forces the system to use one pattern where three are needed, or forces the developer to hand-code topology transitions that are specific to one workflow and fragile to changes.

The second failure mode is model lock-in. Most orchestration frameworks are built around a single LLM provider's API conventions. The agent definitions, tool bindings, memory patterns, and error handling all assume a specific provider. Switching from OpenAI to Anthropic — or running local models for cost-sensitive development — requires rewriting infrastructure, not just changing an API key. This is an artificial constraint. An agent's cognitive function (what it does) should be separable from its execution substrate (which model performs it).

The third failure mode is the absence of production safety infrastructure. Research frameworks demonstrate impressive multi-agent coordination in demos, but production deployment requires human-in-the-loop approval gates, role-based access control, budget enforcement, audit logging, and explicit stopping conditions. These are not optional features to add later — they are structural requirements that, when absent, make the system unusable for any task where an agent's actions have real consequences.

Agentic Titan addresses all three. It implements a **polymorphic topology engine** that supports nine distinct coordination patterns and can switch between them at runtime based on task analysis. It provides a **model-agnostic adapter layer** that routes requests across Ollama, Anthropic, OpenAI, and Groq based on configurable strategies (cost-optimized, quality-first, speed-first, cognitive-task-aware). And it ships with a **production safety stack** — human-in-the-loop approval gates with risk classification, RBAC, JWT authentication, API key management, budget tracking, content filtering, and full PostgreSQL audit logging — that treats safety as first-class architecture rather than an afterthought.

The result is a system that scales from local development (two agents, one laptop, Ollama) to production deployment (100+ agents across Docker containers or Firecracker microVMs with Prometheus observability, Grafana dashboards, and Celery batch processing) without changing agent definitions or rewriting coordination logic.

---

## Core Architecture

Agentic Titan is organized into four architectural layers that compose vertically:

```
                    +-------------------------------------------------+
                    |              SAFETY & GOVERNANCE                 |
                    |   HITL Gates  |  RBAC  |  Budget  |  Audit      |
                    +-------------------------------------------------+
                             |               |              |
                    +-------------------------------------------------+
                    |              HIVE MIND LAYER                     |
                    |   Redis State  |  ChromaDB Vectors  |  Events   |
                    +-------------------------------------------------+
                             |               |              |
          +------------------+---------------+--------------+---------+
          |                  |               |              |         |
  +-------v--------+ +------v------+ +------v------+ +----v-------+ |
  | TOPOLOGY ENGINE| | LLM ADAPTER | | AGENT FORGE | | WORKFLOWS  | |
  |  9 topologies  | | 4 providers | | 22 archetypes| | DAG engine | |
  |  dynamic switch| | cost router | | YAML DSL    | | cognitive  | |
  +----------------+ +-------------+ +-------------+ +------------+ |
          |                  |               |              |         |
  +-------v------------------v---------------v--------------v--------+
  |                   RUNTIME FABRIC                                  |
  |   Local Python  |  Docker  |  OpenFaaS  |  Firecracker MicroVM   |
  +-----------------------------------------------------------------+
```

**Layer 1: Runtime Fabric.** Agents execute within runtimes selected by an intelligent runtime selector. Local Python processes serve development. Docker containers provide production isolation with resource limits. OpenFaaS enables serverless burst scaling. Firecracker microVMs (Phase 18) provide hardware-level isolation with sub-second boot times — each agent runs in its own lightweight VM with VSOCK communication, TAP/NAT networking, and custom rootfs images.

**Layer 2: Core Engine.** Four subsystems operate at this layer. The **Topology Engine** implements nine coordination patterns (swarm, hierarchy, pipeline, mesh, ring, star, rhizomatic, fission-fusion, stigmergic) with runtime switching driven by task analysis and criticality detection. The **LLM Adapter** provides a uniform interface across four providers (Ollama, Anthropic, OpenAI, Groq) with routing strategies including cost-optimized, quality-first, speed-first, round-robin, and cognitive-task-aware selection. The **Agent Forge** parses declarative YAML agent specifications (the Agent Spec DSL) into executable agent instances with capabilities, personality traits, tool bindings, memory configuration, and LLM preferences. The **Workflow Engine** orchestrates multi-stage inquiries as directed acyclic graphs with context compaction, epistemic signature tracking, and contradiction detection.

**Layer 3: Hive Mind.** Shared intelligence infrastructure. Redis provides fast key-value state management and pub/sub event distribution. ChromaDB stores vector embeddings for semantic memory search. Together they implement the "collective consciousness" that enables real-time coordination: agents write findings to shared memory, subscribe to relevant event channels, and query accumulated knowledge without direct coupling.

**Layer 4: Safety and Governance.** Every agent action passes through a safety chain before execution. The Human-in-the-Loop handler classifies actions by risk level and routes high-risk operations to approval gates. Role-based access control restricts what each agent can do. Budget tracking enforces spending limits per agent, per session, and per workflow. Content filtering catches unsafe outputs. PostgreSQL audit logging records every decision for post-hoc review. JWT authentication and API key management secure the external API surface.

The system has completed 18 development phases, from foundational DSL and runtime work through advanced topology dynamics, RLHF learning pipelines, and production hardening. All quality gates are green: zero lint errors, zero type errors across the full codebase, 1,312 tests passing, CI security and dependency-integrity checks clean.

---

## Key Concepts

### Concept 1: Polymorphic Topology and Runtime Switching

The central design insight of Agentic Titan is that topology is not an architectural constant — it is a runtime variable. Different phases of a task demand different coordination patterns, and a system that cannot switch between them forces developers to choose suboptimal patterns or hand-code brittle transitions.

The Topology Engine implements nine distinct patterns. The six classical topologies — **swarm** (all-to-all, emergent behavior for brainstorming and consensus), **hierarchy** (tree-structured delegation for command chains), **pipeline** (sequential stages for workflows), **mesh** (resilient grid for fault-tolerant tasks), **ring** (token-passing for voting and sequential processing), and **star** (hub-and-spoke for coordinator patterns) — cover the standard multi-agent coordination literature. Three advanced topologies extend this vocabulary: **rhizomatic** (lateral, non-hierarchical connections inspired by Deleuze and Guattari's philosophical model), **fission-fusion** (dynamic clustering inspired by crow roost dynamics where swarms split into independent exploration clusters and reconverge for collective decision-making), and **stigmergic** (environment-mediated coordination where agents communicate through shared traces rather than direct messaging, modeled on insect pheromone systems).

Topology switching is driven by two mechanisms. The first is explicit: an LLM-powered task analyzer examines a task description and produces a `TaskProfile` that maps to the optimal topology based on attributes like consensus requirements, sequential stages, fault tolerance needs, and parallelism. The second is emergent: the **criticality detection system** (based on statistical physics models of phase transitions) monitors correlation length, susceptibility, relaxation time, and fluctuation size across the agent network. When the system detects movement toward a phase transition — the "edge of chaos" where collective behavior is maximally adaptive — it can trigger topology changes to maintain optimal coordination dynamics.

This is not theoretical apparatus bolted onto a simple system. The topology engine's `AgentNode` structure tracks capabilities, roles, parent-child relationships, neighbor sets, and metadata per agent. The `TaskProfile` classifier uses both keyword analysis and LLM-powered reasoning to map natural-language task descriptions to topology recommendations. And the fission-fusion manager implements genuine cluster dynamics: tracking correlation metrics across agent groups, triggering fission (splitting into independent subclusters for parallel exploration) and fusion (reconverging for information sharing) based on measured task correlation and coordination demand.

### Concept 2: Model-Agnostic Cognitive Routing

Agentic Titan treats LLM providers as interchangeable execution substrates with measurable characteristics, not as frameworks to build around. The adapter layer defines a uniform `LLMAdapter` interface that four provider-specific implementations (Ollama, Anthropic, OpenAI, Groq) conform to. Every adapter exposes the same methods for text generation, tool use, and streaming. Agent specifications declare LLM preferences (`preferred: claude-sonnet`, `fallback: [gpt-4o, llama3.2]`) without coupling to provider-specific APIs.

The LLM Router selects providers based on configurable strategies. **Cost-optimized** prefers local models (Ollama, cost tier 1) then cheap cloud models (Groq, tier 2) before premium providers (Anthropic/OpenAI, tier 3). **Quality-first** inverts this, selecting the highest-quality available model. **Speed-first** routes to the fastest provider (Groq at tier 4). **Round-robin** distributes load across available providers. **Fallback** follows the agent's declared preference chain.

The Cognitive Router extends this with task-type-aware selection. Eight cognitive task types — structured reasoning, creative synthesis, mathematical analysis, cross-domain connection, meta-analysis, pattern recognition, code generation, and research synthesis — map to ranked lists of preferred models based on empirical cognitive-strength profiles. Claude excels at structured reasoning, ethical analysis, and consistent narrative. GPT-4 excels at creative synthesis and cross-domain connections. Gemini excels at mathematical reasoning and structured data. The cognitive router selects the optimal model for each stage of a multi-perspective inquiry, producing genuinely heterogeneous cognitive output rather than stylistic variation from a single provider.

This separation means switching from cloud development (Anthropic + OpenAI) to air-gapped deployment (Ollama with local models) requires changing environment variables, not rewriting agent definitions or workflow logic. The same agent specification produces equivalent behavior across providers, with graceful degradation when preferred models are unavailable.

### Concept 3: Agent Archetypes as Composable Primitives

Rather than requiring developers to implement agent behavior from scratch, Agentic Titan ships 22 pre-built agent archetypes organized into four categories that span operational, governance, biological, and philosophical models of coordination.

**Core archetypes** (10) cover standard software development and knowledge work: Orchestrator (coordinates multi-agent workflows), Researcher (gathers and analyzes information), Coder (writes and tests code), Reviewer (quality assurance), Paper2Code (converts research papers to implementations), CFO (budget management and cost optimization), DevOps (infrastructure automation), SecurityAnalyst (code security scanning), DataEngineer (ETL and schema management), and ProductManager (requirements and roadmap planning).

**Governance archetypes** (5) model institutional decision-making structures: JuryAgent (deliberative body with evidence evaluation and voting), ExecutiveAgent (decision implementation and execution leadership), LegislativeAgent (policy proposal and debate), JudicialAgent (compliance review and dispute resolution), and BureaucracyAgent (rule-based processing with specialized departmental roles).

**Biological archetypes** (2) model living-systems coordination: EusocialColonyAgent (superorganism with castes, division of labor, and stigmergic communication — modeled on eusocial insects) and CellAgent (multicellular patterns with apoptosis, signaling cascades, and differentiation — modeled on biological cell behavior).

**Philosophical archetypes** (3) model theoretical coordination frameworks: AssemblageAgent (heterogeneous assembly with territorialization and deterritorialization dynamics from Deleuze-Guattarian philosophy), ActorNetworkAgent (actor-network theory-based actant enrollment and translation following Latour), and SwarmIntelligenceAgent (particle swarm optimization and ant colony optimization algorithms), plus DAOAgent (decentralized autonomous governance with proposal mechanisms and on-chain-style voting).

Every archetype extends `BaseAgent`, which provides lifecycle management (initialize, work, shutdown), hive mind integration, topology-aware communication, resilience patterns (circuit breaker, retry with backoff), PostgreSQL audit logging, explicit stopping conditions (success, failure, max turns, timeout, budget exhaustion, user cancellation, checkpoint required, stuck detection, error threshold), and checkpointing for recovery. Archetypes are composable: a workflow can deploy a JuryAgent for deliberation, route its verdict to an ExecutiveAgent for implementation, and have a JudicialAgent review compliance — modeling a complete institutional process as an executable agent pipeline.

### Concept 4: Production Safety as First-Class Architecture

Agentic Titan implements safety not as guardrails added to a finished system but as structural constraints woven into the execution path. Every agent action flows through the safety chain before it executes.

The **Human-in-the-Loop (HITL) handler** receives action requests from agents, classifies them by risk level using a configurable `ActionClassifier`, and routes high-risk actions to approval gates. Low-risk actions can be auto-approved for throughput. Medium and high-risk actions block execution until a human approves or denies them, with configurable timeout and notification channels (WebSocket for real-time dashboards, Redis for distributed state). The HITL handler maintains a full `GateRegistry` of approval gates, each with its own criteria and notification callbacks.

**Role-based access control** restricts agent capabilities by role. An agent assigned the "researcher" role can query external APIs and write to shared memory but cannot execute code or modify infrastructure. Role assignments are declarative (specified in agent YAML specs) and enforced at the framework level — bypassing RBAC requires bypassing the `BaseAgent` execution path itself.

**Budget tracking** enforces cost limits at three granularities: per-agent, per-session, and per-workflow. The CFO archetype provides budget management as an agent capability, but the budget enforcement layer operates independently — an agent that exceeds its budget triggers a `BUDGET_EXHAUSTED` stopping condition regardless of what the agent itself wants to do.

**Content filtering and audit logging** complete the safety stack. Content filters catch unsafe outputs before they reach users or downstream systems. PostgreSQL audit logging with Alembic migrations records every decision, every action, and every approval/denial for post-hoc review and compliance. The combination ensures that production deployments of multi-agent systems are inspectable, controllable, and auditable — requirements that separate production AI infrastructure from research demonstrations.

### Concept 5: Declarative Agent Specification (The Agent Spec DSL)

Agent definitions in Agentic Titan are declarative YAML documents, not imperative code. The Agent Spec DSL provides a portable, human-readable format for specifying what an agent is, what it can do, and how it should behave — decoupled from the runtime, provider, or topology it operates within.

```yaml
apiVersion: titan/v1
kind: Agent
metadata:
  name: researcher
  labels:
    tier: cognitive
spec:
  capabilities:
    - web_search
    - summarization
  personality:
    traits: [thorough, curious, skeptical]
    communication_style: academic
  llm:
    preferred: claude-sonnet
    fallback: [gpt-4o, llama3.2]
  tools:
    - name: web_search
      protocol: native
  memory:
    short_term: 10
    long_term: hive_mind
```

The DSL is parsed by a Pydantic-validated spec engine (`titan/spec.py`) that enforces structural correctness at load time. Every field maps to a typed model: `LLMPreference` with preferred and fallback chains, `ToolSpec` with protocol variants (MCP, native, HTTP), `MemorySpec` with short-term window size and long-term backend selection, `PersonalitySpec` with trait lists and communication styles, and `RuntimeSpec` with per-environment overrides for local, container, and serverless execution.

The Kubernetes-influenced `apiVersion` and `kind` fields are deliberate. Agent specs are designed to be versionable, diffable, and storable alongside infrastructure configuration. A team can review agent definition changes in pull requests the same way they review code changes — because the specification *is* the source of truth, not an opaque model configuration hidden inside Python classes.

---

## Installation & Setup

### Prerequisites

| Requirement | Version | Purpose |
|------------|---------|---------|
| Python | >= 3.11 | Runtime and type system features (StrEnum, modern typing) |
| Redis | >= 7.0 | Working memory, event bus, rate limiting |
| Docker | >= 24.0 | Container runtime, infrastructure services |
| pip | latest | Package management |

Optional infrastructure for full-stack deployment:

| Component | Purpose |
|-----------|---------|
| ChromaDB | Vector memory for semantic search |
| PostgreSQL | Audit logging, authentication storage |
| Prometheus + Grafana | Observability stack |
| NATS | High-throughput event streaming (alternative to Redis Pub/Sub) |
| Ray | Distributed compute backend |
| Firecracker | MicroVM isolation |

### Installation

```bash
# Clone the repository
git clone https://github.com/organvm-iv-taxis/agentic-titan.git
cd agentic-titan

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core package
pip install -e .

# Install with optional feature sets
pip install -e ".[dev]"          # pytest, ruff, mypy
pip install -e ".[dashboard]"    # FastAPI web dashboard
pip install -e ".[metrics]"      # Prometheus instrumentation
pip install -e ".[vector-db]"    # ChromaDB integration
pip install -e ".[documents]"    # PDF, DOCX, XLSX, PPTX tools
pip install -e ".[postgres]"     # PostgreSQL persistence
pip install -e ".[langfuse]"     # Langfuse observability
pip install -e ".[memori]"       # SQL-native memory backend

# Or install everything for development
pip install -e ".[dev,dashboard,metrics,vector-db,documents,postgres,langfuse,memori]"
```

### Start Infrastructure

```bash
# Start Redis and ChromaDB (minimal)
docker compose -f deploy/compose.yaml up -d redis chromadb

# Start with monitoring (adds Prometheus + Grafana)
docker compose -f deploy/compose.yaml --profile monitoring up -d

# Start full stack (all services)
docker compose -f deploy/compose.yaml --profile full up -d

# Verify services
titan status
```

### LLM Provider Configuration

Set environment variables for the providers you want to use:

```bash
# Local models (no API key needed)
# Requires Ollama running locally: https://ollama.ai
export OLLAMA_HOST="http://localhost:11434"

# Cloud providers (set one or more)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
```

The LLM router auto-detects available providers at startup. No provider configuration is required beyond setting the relevant environment variable — the router discovers capabilities, model lists, and tool support automatically.

---

## Quick Start

### Run a Single Agent

```bash
# Initialize a project directory
titan init my-project
cd my-project

# Run the researcher agent against a prompt
titan run specs/researcher.titan.yaml -p "Analyze the current state of multi-agent orchestration frameworks"
```

### Start a Multi-Agent Swarm

```bash
# Auto-select topology based on task analysis
titan swarm "Build a REST API for a bookstore" --topology auto --agents 5

# Explicitly select topology
titan swarm "Review and approve these pull requests" --topology pipeline --agents 3

# LLM-powered task analysis (recommends topology, agents, runtime)
titan analyze "Conduct a comprehensive security audit of this codebase"
```

### Start the Observability Stack

```bash
# Start the web dashboard
titan dashboard start --port 8080

# Start Prometheus metrics endpoint
titan metrics start --port 9100

# Start full observability stack (Prometheus + Grafana + dashboard)
titan observe start

# Access:
#   Dashboard:  http://localhost:8080 (real-time WebSocket agent monitoring)
#   Grafana:    http://localhost:3000 (admin/titan)
#   Prometheus: http://localhost:9090
```

---

## Working Examples

### Example 1: Multi-Perspective Research Inquiry

The Inquiry Engine orchestrates multi-model collaborative research by routing different cognitive tasks to different LLM providers. This example runs a five-stage inquiry where each stage uses the model best suited to its cognitive requirement.

```bash
# Start a multi-perspective inquiry via CLI
titan inquiry start \
  --topic "The relationship between emergence in complex systems and creative practice" \
  --stages 5 \
  --output report.md
```

Or programmatically:

```python
import asyncio
from titan.workflows.inquiry_engine import InquiryEngine, InquirySession
from titan.workflows.cognitive_router import CognitiveRouter

async def run_inquiry():
    engine = InquiryEngine()

    session = await engine.start_inquiry(
        topic="The relationship between emergence in complex systems and creative practice",
        workflow=None,  # Uses default EXPANSIVE_INQUIRY_WORKFLOW
    )

    # Stream results as each stage completes
    async for stage_result in engine.stream_results(session.session_id):
        print(f"[{stage_result.role}] ({stage_result.model_used})")
        print(stage_result.content[:200])
        print("---")

asyncio.run(run_inquiry())
```

The inquiry engine routes each stage through the Cognitive Router: structured reasoning stages go to Claude, creative synthesis stages go to GPT-4, mathematical analysis stages go to Gemini. The result is a genuine multi-perspective analysis, not a single model rephrasing itself.

### Example 2: Self-Organizing Development Team

This example spawns a team of agents that self-organize their topology based on the task phase. The team starts in swarm mode for research, switches to pipeline for implementation, and uses ring topology for code review.

```python
import asyncio
from titan.spec import AgentSpecLoader
from hive.memory import HiveMind, MemoryConfig
from hive.topology import TopologyEngine, TopologyType

async def development_team():
    # Initialize shared infrastructure
    hive = HiveMind(MemoryConfig())
    await hive.initialize()

    topology = TopologyEngine(hive_mind=hive)

    # Load agent specs from YAML
    loader = AgentSpecLoader()
    specs = [
        loader.load("specs/researcher.titan.yaml"),
        loader.load("specs/coder.titan.yaml"),
        loader.load("specs/reviewer.titan.yaml"),
        loader.load("specs/orchestrator.titan.yaml"),
    ]

    # Phase 1: Research (swarm topology)
    await topology.set_topology(TopologyType.SWARM)
    print(f"Phase 1: {topology.current_topology.value} topology")
    # Agents explore independently, share findings via HiveMind

    # Phase 2: Implementation (pipeline topology)
    await topology.set_topology(TopologyType.PIPELINE)
    print(f"Phase 2: {topology.current_topology.value} topology")
    # Research -> Code -> Review -> Merge

    # Phase 3: Review (ring topology)
    await topology.set_topology(TopologyType.RING)
    print(f"Phase 3: {topology.current_topology.value} topology")
    # Token-passing review: each agent reviews the previous agent's work

    await hive.cleanup()

asyncio.run(development_team())
```

### Example 3: Stress Testing with Chaos Engineering

This example runs a 30-agent chaos test that injects random failures and topology switches to validate system resilience under adversarial conditions.

```bash
# Run chaos stress test: 30 agents, 10% failure rate, 120 seconds
titan stress chaos --agents 30 --failure-rate 0.1 --duration 120 --output chaos_results.json

# Run scale test: maximum agents, minimal work per agent
titan stress scale --agents 100 --duration 60

# Run swarm stress test with detailed metrics
titan stress swarm --agents 50 --duration 120 --output swarm_results.json
```

The stress testing framework (`titan/stress/`) provides five built-in scenarios: swarm (all-to-all communication), pipeline (sequential stage processing), hierarchy (tree delegation), chaos (random failures + topology switches), and scale (maximum agents with minimal work). Each scenario produces structured JSON results with latency distributions, failure rates, topology switch counts, and throughput metrics.

---

## Testing & Validation

Agentic Titan ships with **1,312 tests passing** across 108 test files organized into 18 test categories:

| Category | Files | Coverage |
|----------|-------|----------|
| Agent Archetypes | 9 | All 22 archetypes — lifecycle, capabilities, governance patterns |
| Hive Mind & Topology | 9 | Criticality detection, fission-fusion, information centers, multi-scale neighbors |
| Workflows | 9 | Inquiry engine, DAG execution, conversational flows, temporal patterns, narrative synthesis |
| Integration | 8 | Cross-component interaction, topology transitions under load |
| Batch Processing | 7 | Celery integration, worker lifecycle, stall detection, recovery, cleanup |
| Learning Pipeline | 7 | RLHF samples, reward model, DPO trainer, evaluation suite, preference pairs |
| Prompts | 7 | Auto-prompt generation, token optimization, prompt metrics, examples |
| API | 6 | Admin routes, auth endpoints, batch WebSocket, inquiry WebSocket, rate limiting |
| Authentication | 6 | JWT creation/verification, API key management, middleware, RBAC |
| Adversarial | 5 | Prompt injection resistance, boundary testing, malformed input handling |
| Analysis | 4 | Contradiction detection, dialectic synthesis |
| Chaos | 4 | Fission recovery, resilience under random failure injection |
| E2E | 4 | Full workflow execution: swarm, topology switching, budget enforcement, RLHF pipeline |
| Evaluation | 4 | Model comparison, cognitive signature assessment |
| MCP | 4 | MCP server protocol, notifications, prompts, resources |
| Runtime | 4 | Firecracker config, VM lifecycle, runtime isolation |
| Performance | 3 | Load testing, throughput benchmarking |
| Ray | 3 | Ray Serve integration, actor lifecycle, backend selection |

### Quality Gate Status

The repository has completed a comprehensive quality program ("Omega Closure") with six tranche checkpoints:

- **Tranche 0 (Environment Reproducibility):** Fresh `.venv` validated.
- **Tranche 1 (Baseline Snapshot):** Ruff, mypy, pytest baselines captured.
- **Tranche 2 (Full-Lint Blocking):** 1,270 initial lint errors reduced to **0**. Full-repo `ruff check` and `ruff format --check` pass.
- **Tranche 3 (Full-Typecheck Blocking):** 28 initial mypy errors across 23 files reduced to **0**. All quarantine overrides removed. Full-repo strict mypy passes.
- **Tranche 4 (Runtime Test Completion):** 1,312 passed, 16 skipped (infrastructure-dependent tests). Zero RuntimeWarning violations.
- **Tranche 5 (Security and Deploy):** CI security gate green. Dependency integrity gate green. Deploy smoke artifacts captured for Docker Compose and k3s.
- **Tranche 6 (Documentation and Release Closure):** Changelog, release notes, governance audit workflow, and signoff records complete.

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run full test suite
REDIS_URL=redis://localhost:6379/0 pytest tests/ -q

# Run with coverage
pytest --cov=titan --cov=agents --cov=hive

# Run specific categories
pytest tests/adversarial/        # Adversarial safety tests
pytest tests/chaos/              # Chaos engineering tests
pytest tests/e2e/                # End-to-end workflow tests
pytest tests/integration/        # Cross-component integration
pytest tests/performance/        # Load and throughput tests
pytest tests/mcp/                # MCP protocol tests
pytest tests/ray/                # Ray backend tests
```

---

## Downstream Implementation

Agentic Titan's orchestration patterns propagate through the eight-organ system. ORGAN-IV (Orchestration) provides the coordination infrastructure that other organs instantiate for their domain-specific needs.

### ORGAN-I: Theory

[**RE:GE (Recursive Engine: Generative Entity)**](https://github.com/organvm-i-theoria/recursive-engine--generative-entity) implements a 21-organ symbolic processing system where each organ is a specialized handler for a domain of meaning (narrative, reflection, ceremony, economy). RE:GE's organ routing system — the Soul Patchbay that routes fragments between processors based on charge dynamics and mode selection — is a specialized instance of the topology-aware coordination pattern implemented generically in Agentic Titan. Where Titan's topology engine switches between swarm, hierarchy, pipeline, and mesh patterns for agent coordination, RE:GE's patchbay switches between organ-specific processing paths for symbolic material. The abstraction is the same: dynamic routing of work units through heterogeneous processors based on runtime characteristics.

### ORGAN-II: Art

[**Metasystem Master**](https://github.com/organvm-ii-poiesis/metasystem-master) uses multi-agent coordination for performance orchestration — coordinating generative audio, visual, and interactive systems in real-time creative performance. The base agent lifecycle (`BaseAgent` with initialize/work/shutdown, circuit breaker resilience, and checkpoint recovery) was originally extracted from metasystem-core and generalized into Agentic Titan's framework. The creative domain adds requirements that stress-tested the framework in productive ways: real-time coordination demands sub-second topology switching, generative processes require stigmergic communication patterns (agents coordinating through shared environmental traces rather than direct messages), and performance contexts need graceful degradation under load rather than hard failure.

### ORGAN-III: Commerce

Commercial products in ORGAN-III ([organvm-iii-ergon](https://github.com/organvm-iii-ergon)) leverage Agentic Titan's orchestration patterns for production AI workflows. The batch processing system (Celery integration with stall detection, recovery, and cleanup), the authentication infrastructure (JWT + API keys + rate limiting), and the observability stack (Prometheus metrics, Grafana dashboards) were built to support commercial deployment requirements. The CFO agent archetype's budget tracking was designed specifically for SaaS contexts where per-customer cost management is a business requirement, not just an engineering concern.

---

## Cross-References

### Within ORGAN-IV (Orchestration)

- [**agent--claude-smith**](https://github.com/organvm-iv-taxis/agent--claude-smith) — Claude-specific orchestrator, session management, and security hooks. Agentic Titan's `AnthropicAdapter` was informed by claude-smith's provider-specific patterns.
- [**a-i--skills**](https://github.com/organvm-iv-taxis/a-i--skills) — YAML-based skill definitions. The Agent Spec DSL's Kubernetes-influenced YAML format was inspired by the skills repository's declarative approach.
- [**petasum-super-petasum**](https://github.com/organvm-iv-taxis/petasum-super-petasum) — Network architecture patterns. Universal node network concepts inform the mesh and rhizomatic topology implementations.

### Across the Organ System

- **ORGAN-V (Public Process):** [How I Used 4 AI Agents to Cross-Validate an Eight-Organ System](https://github.com/organvm-v-logos) — Essay on multi-agent orchestration methodology, drawing on Agentic Titan as a primary case study.
- **ORGAN-VI (Community):** The governance archetypes (JuryAgent, ExecutiveAgent, LegislativeAgent, JudicialAgent) model institutional decision-making patterns relevant to community governance infrastructure.
- **System Context:** [meta-organvm](https://github.com/meta-organvm) — The umbrella organization coordinating all eight organs. Agentic Titan's role as ORGAN-IV infrastructure means it provides orchestration patterns consumed by the other seven organs.

### Source Lineage

Agentic Titan synthesizes patterns from seven precursor systems:

| Source | Contribution |
|--------|-------------|
| agent--claude-smith | Orchestrator patterns, session management, security hooks |
| metasystem-core | BaseAgent lifecycle, circuit breaker, knowledge graph patterns |
| my--father-mother | Dual-persona logging, MCP bridge |
| a-i-council--coliseum | Decision engine, voting protocols, communication protocol |
| a-i--skills | YAML DSL patterns for agent specification |
| iGOR | Episodic learning system |
| aionui | LLM auto-detection and fallback routing |

---

## Contributing

### Adding a New Agent Archetype

1. Create a spec file: `specs/myagent.titan.yaml`
2. Implement the archetype: `agents/archetypes/myagent.py` (extend `BaseAgent`)
3. Register the archetype in `agents/archetypes/__init__.py`
4. Add tests: `tests/archetypes/test_myagent.py`

### Adding a New LLM Provider

1. Implement the `LLMAdapter` interface in `adapters/base.py`
2. Add auto-detection logic in `adapters/router.py`
3. Update `DEFAULT_MODELS` and `PROVIDER_INFO` dictionaries
4. Add integration tests in `tests/integration/`

### Adding a New Topology

1. Add the topology type to `TopologyType` enum in `hive/topology.py`
2. Implement the topology's coordination logic as a topology class
3. Update the `TaskProfile.from_task()` analyzer to recognize task patterns for the new topology
4. Add tests in `tests/test_hive/`

### Code Quality Requirements

All contributions must pass the full quality gate:

```bash
# Lint (zero tolerance)
ruff check .
ruff format --check .

# Type check (zero tolerance, no quarantine overrides)
mypy --ignore-missing-imports hive agents titan mcp dashboard

# Tests (1,312+ passing)
REDIS_URL=redis://localhost:6379/0 pytest tests/ -q
```

See `docs/ci-quality-gates.md` for the complete CI quality gate model and `plans/completion_program.md` for the tranche gate methodology.

---

## License & Author

**License:** MIT

**Author:** [@4444J99](https://github.com/4444J99)

**Organization:** [organvm-iv-taxis](https://github.com/organvm-iv-taxis) (ORGAN-IV: Orchestration)

**System:** [meta-organvm](https://github.com/meta-organvm) — The eight-organ creative-institutional system

---

*523 files. 3.7M tokens of Python. 18 development phases. 22 agent archetypes. 9 topology patterns. 1,312 tests. Zero lint errors. Zero type errors. Built to orchestrate.*

<!-- SYSTEM-NAV-START -->

---

<sub>[Case Study](https://4444j99.github.io/portfolio/projects/agentic-titan/) · [Portfolio](https://4444j99.github.io/portfolio/) · [System Directory](https://4444j99.github.io/portfolio/directory/) · [ORGAN IV · Taxis](https://organvm-iv-taxis.github.io/) · Part of the <a href="https://4444j99.github.io/portfolio/directory/">ORGANVM eight-organ system</a></sub>

<!-- SYSTEM-NAV-END -->
