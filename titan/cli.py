"""
Agentic Titan CLI - Command-line interface for the agent swarm.

Commands:
- titan run <spec>     - Run an agent from a spec file
- titan swarm <task>   - Start a swarm for a task
- titan status         - Show swarm status
- titan list           - List available agents
- titan health         - Health check all services

Inspired by: kimi-cli mode-switching patterns
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from adapters.base import LLMMessage
from adapters.router import get_router
from agents.personas import ORCHESTRATOR, report_error, report_success, say
from hive.memory import HiveMind
from hive.topology import TopologyEngine
from titan.spec import AgentSpec, get_spec_registry

if TYPE_CHECKING:
    from adapters.router import LLMRouter
    from agents.framework.base_agent import BaseAgent

# Initialize
app = typer.Typer(
    name="titan",
    help="Agentic Titan - Polymorphic Agent Swarm",
    add_completion=False,
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("titan.cli")


# ============================================================================
# Helper Functions
# ============================================================================


def print_banner() -> None:
    """Print the Titan banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗║
    ║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝║
    ║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║     ║
    ║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║     ║
    ║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗║
    ║    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝║
    ║                                                           ║
    ║          ████████╗██╗████████╗ █████╗ ███╗   ██╗          ║
    ║          ╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║          ║
    ║             ██║   ██║   ██║   ███████║██╔██╗ ██║          ║
    ║             ██║   ██║   ██║   ██╔══██║██║╚██╗██║          ║
    ║             ██║   ██║   ██║   ██║  ██║██║ ╚████║          ║
    ║             ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝          ║
    ║                                                           ║
    ║        Polymorphic Agent Swarm Architecture v0.1.0        ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="cyan")


async def check_infrastructure() -> dict[str, bool]:
    """Check infrastructure services."""
    import httpx

    checks = {
        "redis": False,
        "chromadb": False,
        "nats": False,
        "ollama": False,
    }

    async with httpx.AsyncClient() as client:
        # Redis
        try:
            import redis.asyncio as redis_lib

            from_url = cast(Any, redis_lib.from_url)
            r = from_url("redis://localhost:6379")
            await r.ping()
            checks["redis"] = True
            await r.close()
        except Exception:
            pass

        # ChromaDB
        try:
            resp = await client.get("http://localhost:8000/api/v1/heartbeat", timeout=2)
            checks["chromadb"] = resp.status_code == 200
        except Exception:
            pass

        # NATS
        try:
            resp = await client.get("http://localhost:8222/healthz", timeout=2)
            checks["nats"] = resp.status_code == 200
        except Exception:
            pass

        # Ollama
        try:
            resp = await client.get("http://localhost:11434/api/tags", timeout=2)
            checks["ollama"] = resp.status_code == 200
        except Exception:
            pass

    return checks


# ============================================================================
# Commands
# ============================================================================


@app.command()
def run(
    spec_path: str = typer.Argument(..., help="Path to agent spec file"),
    prompt: str = typer.Option(None, "--prompt", "-p", help="Task prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run an agent from a spec file."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    say(ORCHESTRATOR, f"Loading spec from {spec_path}")

    async def run_agent() -> None:
        spec = AgentSpec.from_file(spec_path)
        say(ORCHESTRATOR, f"Loaded agent: {spec.name}")

        console.print(
            Panel(
                f"[green]Agent loaded successfully![/green]\n\n"
                f"Name: {spec.name}\n"
                f"Capabilities: {', '.join(spec.capabilities)}\n"
                f"LLM: {spec.llm.get('preferred', 'default')}",
                title="Agent Spec",
            )
        )

        if not prompt:
            say(ORCHESTRATOR, "No prompt provided. Use --prompt to run a task.")
            return

        say(ORCHESTRATOR, f"Task: {prompt}")

        # Create agent based on spec type
        from agents.archetypes import (
            CoderAgent,
            OrchestratorAgent,
            ResearcherAgent,
            ReviewerAgent,
        )

        # Map spec names to agent classes
        agent_map = {
            "researcher": ResearcherAgent,
            "coder": CoderAgent,
            "reviewer": ReviewerAgent,
            "orchestrator": OrchestratorAgent,
        }

        agent_class = agent_map.get(spec.name.lower())
        if not agent_class:
            report_error(ORCHESTRATOR, Exception(f"Unknown agent type: {spec.name}"), "")
            return

        # Initialize Hive Mind for agent
        hive = HiveMind()
        await hive.initialize()

        try:
            # Create agent with appropriate kwargs
            if spec.name.lower() == "researcher":
                agent = agent_class(topic=prompt, hive_mind=hive)
            elif spec.name.lower() == "coder":
                agent = agent_class(task_description=prompt, hive_mind=hive)
            elif spec.name.lower() == "reviewer":
                agent = agent_class(content=prompt, hive_mind=hive)
            elif spec.name.lower() == "orchestrator":
                agent = agent_class(task=prompt, hive_mind=hive)
            else:
                agent = agent_class(hive_mind=hive)

            # Run agent
            say(ORCHESTRATOR, "Starting agent...")
            result = await agent.run(prompt)

            # Display result
            console.print(
                Panel(
                    f"[green]Agent completed![/green]\n\n"
                    f"Status: {result.state.value}\n"
                    f"Turns: {result.turns_taken}\n"
                    f"Duration: {result.execution_time_ms / 1000:.2f}s",
                    title="Result",
                )
            )

            if result.result:
                output_str = str(result.result)
                console.print(
                    Panel(
                        output_str[:2000] + ("..." if len(output_str) > 2000 else ""),
                        title="Output",
                    )
                )

        finally:
            await hive.shutdown()

    try:
        asyncio.run(run_agent())
    except Exception as e:
        report_error(ORCHESTRATOR, e, "Failed to run agent")
        raise typer.Exit(1)


@app.command()
def swarm(
    task: str = typer.Argument(..., help="Task for the swarm"),
    topology: str = typer.Option("auto", "--topology", "-t", help="Topology type"),
    agents: int = typer.Option(3, "--agents", "-a", help="Number of agents"),
    max_tokens: int = typer.Option(50000, "--max-tokens", help="Max tokens budget"),
    timeout: int = typer.Option(300, "--timeout", help="Timeout in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Start an agent swarm for a task."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print_banner()

    async def run_swarm() -> None:
        from agents.archetypes import (
            CoderAgent,
            DataEngineerAgent,
            OrchestratorAgent,
            ProductManagerAgent,
            ResearcherAgent,
            ReviewerAgent,
            SecurityAnalystAgent,
        )

        say(ORCHESTRATOR, "Initializing swarm...")

        # Initialize components
        hive = HiveMind()
        await hive.initialize()

        engine = TopologyEngine(hive)

        # Select topology
        if topology == "auto":
            suggestion = engine.suggest_topology(task)
            selected = suggestion["recommended"]
            say(ORCHESTRATOR, f"Auto-selected topology: {selected}")
            console.print(f"Reasons: {', '.join(suggestion['reasons'])}")
        else:
            selected = topology

        # Create topology
        topo = engine.create_topology(selected)
        say(ORCHESTRATOR, f"Created {selected} topology")

        # Check LLM availability
        router = get_router()
        await router.initialize()
        providers = router.list_available_providers()

        console.print(
            Panel(
                f"Task: {task}\n"
                f"Topology: {selected}\n"
                f"Agents: {agents}\n"
                f"Max Tokens: {max_tokens}\n"
                f"LLM Providers: {', '.join(p.value for p in providers)}",
                title="Swarm Configuration",
            )
        )

        # Analyze task to determine agent mix
        agent_mix = await _analyze_task_for_agents(task, router)
        say(ORCHESTRATOR, f"Suggested agents: {', '.join(agent_mix[:agents])}")

        # Map archetype names to classes
        archetype_map = {
            "researcher": ResearcherAgent,
            "coder": CoderAgent,
            "reviewer": ReviewerAgent,
            "orchestrator": OrchestratorAgent,
            "data_engineer": DataEngineerAgent,
            "product_manager": ProductManagerAgent,
            "security_analyst": SecurityAnalystAgent,
            "analyst": ResearcherAgent,  # Alias
            "writer": ResearcherAgent,  # Map to closest
            "planner": ProductManagerAgent,
            "executor": CoderAgent,
        }

        # Create agents
        created_agents = []
        for i, archetype_name in enumerate(agent_mix[:agents]):
            archetype_class = archetype_map.get(archetype_name.lower(), ResearcherAgent)
            agent_name = f"{archetype_name}-{i + 1}"

            # Create agent with appropriate initialization
            agent: BaseAgent
            if archetype_class == ResearcherAgent:
                agent = archetype_class(
                    topic=task,
                    name=agent_name,
                    hive_mind=hive,
                )
            elif archetype_class == CoderAgent:
                agent = archetype_class(
                    task_description=task,
                    name=agent_name,
                    hive_mind=hive,
                )
            elif archetype_class == ReviewerAgent:
                agent = archetype_class(
                    content=task,
                    name=agent_name,
                    hive_mind=hive,
                )
            elif archetype_class == OrchestratorAgent:
                agent = archetype_class(
                    task=task,
                    name=agent_name,
                    hive_mind=hive,
                )
            else:
                agent = archetype_class(
                    name=agent_name,
                    hive_mind=hive,
                )

            created_agents.append(agent)
            topo.add_agent(agent.agent_id, agent.name, list(agent.capabilities))
            say(ORCHESTRATOR, f"{agent_name} ready ({archetype_name})")

        # Run the swarm
        say(ORCHESTRATOR, f"Starting swarm with {len(created_agents)} agents...")

        results = []
        total_tokens = 0
        start_time = asyncio.get_event_loop().time()

        for agent in created_agents:
            try:
                # Check timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    say(ORCHESTRATOR, "Timeout reached, stopping early")
                    break

                # Check token budget (rough estimate)
                if total_tokens > max_tokens:
                    say(ORCHESTRATOR, "Token budget exhausted")
                    break

                # Initialize and run agent
                await agent.initialize()
                result = await agent.run(task)
                results.append(
                    {
                        "agent": agent.name,
                        "state": result.state.value,
                        "turns": result.turns_taken,
                        "duration_ms": result.execution_time_ms,
                    }
                )

                # Estimate tokens (rough)
                total_tokens += result.turns_taken * 2000

                say(ORCHESTRATOR, f"{agent.name} completed: {result.state.value}")

            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {e}")
                results.append(
                    {
                        "agent": agent.name,
                        "state": "failed",
                        "error": str(e),
                    }
                )

        # Display results
        successful = sum(1 for r in results if r.get("state") != "failed")
        total_duration = int((asyncio.get_event_loop().time() - start_time) * 1000)

        console.print(
            Panel(
                f"Agents: {len(created_agents)}\n"
                f"Successful: {successful}\n"
                f"Estimated Tokens: {total_tokens}\n"
                f"Duration: {total_duration / 1000:.2f}s",
                title="Swarm Result",
                border_style="green" if successful == len(created_agents) else "yellow",
            )
        )

        # Show per-agent results
        table = Table(title="Agent Results")
        table.add_column("Agent", style="cyan")
        table.add_column("State")
        table.add_column("Turns")
        table.add_column("Duration")

        for r in results:
            state_style = "green" if r.get("state") == "completed" else "red"
            duration_ms = r.get("duration_ms")
            duration_text = (
                f"{float(duration_ms) / 1000:.1f}s"
                if isinstance(duration_ms, int | float)
                else "-"
            )
            table.add_row(
                str(r["agent"]),
                f"[{state_style}]{r.get('state', 'unknown')}[/{state_style}]",
                str(r.get("turns", "-")),
                duration_text,
            )

        console.print(table)

        await hive.shutdown()

    asyncio.run(run_swarm())


async def _analyze_task_for_agents(task: str, router: LLMRouter) -> list[str]:
    """
    Analyze task to determine optimal agent archetypes.

    Args:
        task: Task description to analyze
        router: LLM router for analysis

    Returns:
        List of recommended archetype names
    """
    try:
        prompt = f"""Analyze this task and suggest 3-5 agent archetypes from this list:
- researcher: For information gathering, analysis, and synthesis
- coder: For writing and debugging code
- reviewer: For reviewing work quality and providing feedback
- analyst: For data analysis and insights
- planner: For planning and requirements
- executor: For executing defined tasks

Task: {task}

Return ONLY the archetype names in a comma-separated list, nothing else."""

        response = await router.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            max_tokens=100,
        )

        # Parse response
        archetypes = [
            a.strip().lower().replace("-", "_") for a in response.content.split(",") if a.strip()
        ]

        # Validate and deduplicate
        valid_archetypes = [
            a
            for a in archetypes
            if a in ["researcher", "coder", "reviewer", "analyst", "planner", "executor"]
        ]

        if not valid_archetypes:
            return ["researcher", "analyst", "reviewer"]

        return valid_archetypes

    except Exception as e:
        logger.warning(f"Task analysis failed, using defaults: {e}")
        return ["researcher", "analyst", "reviewer"]


@app.command()
def status() -> None:
    """Show swarm status."""
    print_banner()

    async def show_status() -> None:
        checks = await check_infrastructure()

        table = Table(title="Infrastructure Status")
        table.add_column("Service", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("URL")

        services = [
            ("Redis", checks["redis"], "localhost:6379"),
            ("ChromaDB", checks["chromadb"], "localhost:8000"),
            ("NATS", checks["nats"], "localhost:4222"),
            ("Ollama", checks["ollama"], "localhost:11434"),
        ]

        for name, available, url in services:
            status = "[green]●[/green]" if available else "[red]○[/red]"
            table.add_row(name, status, url)

        console.print(table)

        # LLM providers
        router = get_router()
        await router.initialize()

        table2 = Table(title="LLM Providers")
        table2.add_column("Provider", style="cyan")
        table2.add_column("Status", justify="center")
        table2.add_column("Models")

        for info in router.list_providers():
            status = "[green]●[/green]" if info.available else "[red]○[/red]"
            models = ", ".join(info.models[:3]) + ("..." if len(info.models) > 3 else "")
            table2.add_row(info.provider.value, status, models or "-")

        console.print(table2)

    asyncio.run(show_status())


@app.command("list")
def list_agents(
    directory: str = typer.Option("./specs", "--dir", "-d", help="Specs directory"),
) -> None:
    """List available agent specs."""
    registry = get_spec_registry()

    specs_dir = Path(directory)
    if specs_dir.exists():
        registry.load_directory(specs_dir)

    if not registry:
        console.print("[yellow]No agent specs found[/yellow]")
        console.print(f"Create specs in {directory}/*.titan.yaml")
        return

    table = Table(title="Available Agents")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Capabilities")
    table.add_column("LLM")

    for spec in registry.list():
        caps = ", ".join(spec.capabilities[:3])
        if len(spec.capabilities) > 3:
            caps += "..."
        table.add_row(
            spec.id,
            spec.name,
            caps,
            spec.llm.get("preferred", "default"),
        )

    console.print(table)


@app.command()
def health() -> None:
    """Health check all services."""

    async def run_health() -> None:
        say(ORCHESTRATOR, "Running health checks...")

        # Infrastructure
        infra = await check_infrastructure()

        # Hive Mind
        hive = HiveMind()
        try:
            await hive.initialize()
            hive_health = await hive.health_check()
            await hive.shutdown()
        except Exception as e:
            hive_health = {"error": str(e)}

        # LLM Router
        router = get_router()
        llm_health: dict[str, bool] = {}
        llm_error: str | None = None
        try:
            await router.initialize()
            llm_health = await router.health_check()
        except Exception as e:
            llm_error = str(e)

        all_healthy = all(infra.values()) and all(llm_health.values()) and llm_error is None

        if all_healthy:
            report_success(ORCHESTRATOR, "All systems operational")
        else:
            report_error(ORCHESTRATOR, "Some systems unhealthy", "Health check")
            console.print(f"Infrastructure: {infra}")
            console.print(f"Hive Mind: {hive_health}")
            if llm_error:
                console.print(f"LLM: {{'error': {llm_error!r}}}")
            else:
                console.print(f"LLM: {llm_health}")

    asyncio.run(run_health())


@app.command()
def init(
    directory: str = typer.Argument(".", help="Directory to initialize"),
) -> None:
    """Initialize a new Titan project."""
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Create directories
    (dir_path / "specs").mkdir(exist_ok=True)
    (dir_path / "agents").mkdir(exist_ok=True)

    # Create example spec
    example_spec = """apiVersion: titan/v1
kind: Agent
metadata:
  name: researcher
  labels:
    tier: cognitive
    domain: knowledge
spec:
  capabilities:
    - web_search
    - summarization
    - research

  personality:
    traits: [thorough, curious, skeptical]
    communication_style: academic

  llm:
    preferred: claude-sonnet
    min_context: 16000
    tools_required: false

  tools:
    - name: web_search
      protocol: native
      module: titan.tools.search

  memory:
    short_term: 10
    long_term: hive_mind

  maxTurns: 20
  timeoutMs: 300000
"""

    spec_file = dir_path / "specs" / "researcher.titan.yaml"
    if not spec_file.exists():
        spec_file.write_text(example_spec)

    console.print(f"[green]Initialized Titan project in {directory}[/green]")
    console.print("Created: specs/researcher.titan.yaml")
    console.print("\nNext steps:")
    console.print("  1. Start infrastructure: docker compose up -d redis chromadb")
    console.print("  2. Check status: titan status")
    console.print("  3. Run an agent: titan run specs/researcher.titan.yaml")


@app.command()
def topology(
    task: str = typer.Argument(..., help="Task to analyze"),
) -> None:
    """Suggest a topology for a task."""
    engine = TopologyEngine()
    suggestion = engine.suggest_topology(task)

    console.print(
        Panel(
            f"[bold]Recommended: {suggestion['recommended']}[/bold]\n\n"
            f"Reasons:\n" + "\n".join(f"  • {r}" for r in suggestion["reasons"]) + "\n\n"
            "Task Profile:\n"
            + "\n".join(f"  • {k}: {v}" for k, v in suggestion["profile"].items()),
            title="Topology Suggestion",
        )
    )


@app.command()
def runtime(
    action: str = typer.Argument("status", help="Action: status, suggest, spawn"),
    task: str = typer.Option(None, "--task", "-t", help="Task for suggestion"),
    spec: str = typer.Option(None, "--spec", "-s", help="Agent spec file for spawn"),
    runtime_type: str = typer.Option(None, "--runtime", "-r", help="Force runtime type"),
) -> None:
    """Manage runtime environments."""

    async def run_runtime() -> None:
        from runtime import RuntimeConstraints, RuntimeSelector, RuntimeType

        selector = RuntimeSelector()
        await selector.initialize()

        try:
            if action == "status":
                # Show runtime health
                health = await selector.health_check()
                console.print(
                    Panel(
                        f"[bold]Strategy:[/bold] {health['strategy']}\n"
                        f"[bold]Initialized:[/bold] {health['initialized']}\n\n"
                        f"[bold]Runtimes:[/bold]",
                        title="Runtime Status",
                    )
                )

                for rt_name, rt_health in health["runtimes"].items():
                    status = "[green]✓[/green]" if rt_health.get("initialized") else "[red]✗[/red]"
                    console.print(f"  {status} {rt_name}: {rt_health}")

            elif action == "suggest":
                # Suggest runtime for constraints
                constraints = RuntimeConstraints()
                if task:
                    # Analyze task for constraints
                    task_lower = task.lower()
                    if "gpu" in task_lower or "model" in task_lower:
                        constraints.requires_gpu = True
                    if "scale" in task_lower or "parallel" in task_lower:
                        constraints.expected_instances = 5
                        constraints.auto_scale = True
                    if "isolated" in task_lower or "sandbox" in task_lower:
                        constraints.needs_isolation = True

                suggestion = selector.suggest(constraints)
                console.print(
                    Panel(
                        f"[bold]Recommended:[/bold] {suggestion['recommended']}\n"
                        f"[bold]Score:[/bold] {suggestion['score']:.1f}\n\n"
                        f"[bold]Reasons:[/bold]\n"
                        + "\n".join(f"  • {r}" for r in suggestion["reasons"])
                        + "\n\n"
                        "[bold]Alternatives:[/bold]\n"
                        + "\n".join(
                            f"  • {a['type']}: {a['score']:.1f}" for a in suggestion["alternatives"]
                        ),
                        title="Runtime Suggestion",
                    )
                )

            elif action == "spawn":
                if not spec:
                    report_error(ORCHESTRATOR, Exception("--spec required"), "")
                    return

                # Load spec and spawn
                agent_spec = AgentSpec.from_file(spec)
                rt = RuntimeType(runtime_type) if runtime_type else None

                say(ORCHESTRATOR, f"Spawning {agent_spec.name} on {rt or 'auto-selected'} runtime")
                process = await selector.spawn(
                    agent_id=agent_spec.name,
                    agent_spec=agent_spec.to_dict(),
                    prompt=task,
                    runtime_type=rt,
                )

                console.print(
                    Panel(
                        f"[bold]Process ID:[/bold] {process.process_id}\n"
                        f"[bold]Agent:[/bold] {process.agent_id}\n"
                        f"[bold]Runtime:[/bold] {process.runtime_type.value}\n"
                        f"[bold]State:[/bold] {process.state.value}",
                        title="Agent Spawned",
                    )
                )

            else:
                console.print(f"[red]Unknown action: {action}[/red]")
                console.print("Available: status, suggest, spawn")

        finally:
            await selector.shutdown()

    asyncio.run(run_runtime())


@app.command()
def analyze(
    task: str = typer.Argument(..., help="Task description to analyze"),
    use_llm: bool = typer.Option(True, "--llm/--no-llm", help="Use LLM for analysis"),
) -> None:
    """Analyze a task with LLM and suggest topology."""

    async def run_analysis() -> None:
        from hive.analyzer import TaskAnalyzer
        from hive.events import get_event_bus
        from hive.learning import get_episodic_learner

        say(ORCHESTRATOR, f"Analyzing task: {task[:50]}...")

        # Set up components
        llm_router = None
        if use_llm:
            try:
                llm_router = get_router()
                await llm_router.initialize()
            except Exception as e:
                console.print(f"[yellow]LLM not available, using keyword analysis: {e}[/yellow]")
                llm_router = None

        analyzer = TaskAnalyzer(llm_router=llm_router, use_llm=use_llm and llm_router is not None)
        learner = get_episodic_learner()
        event_bus = get_event_bus()

        # Create topology engine with all components
        engine = TopologyEngine(
            task_analyzer=analyzer,
            episodic_learner=learner,
            event_bus=event_bus,
        )

        # Analyze
        selected, analysis = await engine.analyze_and_select(task, use_llm=use_llm)

        # Display results
        recommended = analysis.get("recommended_topology", "").upper()
        profile_lines = "\n".join(f"  • {k}: {v}" for k, v in analysis.get("profile", {}).items())
        console.print(
            Panel(
                f"[bold green]Recommended Topology: {recommended}[/bold green]\n\n"
                f"[bold]Confidence:[/bold] {analysis.get('confidence', 0.5):.0%}\n"
                f"[bold]Reasoning:[/bold] {analysis.get('reasoning', 'N/A')}\n\n"
                f"[bold]Task Profile:[/bold]\n{profile_lines}",
                title="🔬 Task Analysis",
            )
        )

        # Show learning stats if available
        stats = learner.get_statistics()
        if stats and stats["total_episodes"] > 0:
            console.print(f"\n[dim]Learning: {stats['total_episodes']} episodes recorded[/dim]")

    asyncio.run(run_analysis())


@app.command()
def learning(
    action: str = typer.Argument("stats", help="Action: stats, clear, export"),
    output: str = typer.Option(None, "--output", "-o", help="Output file for export"),
) -> None:
    """Manage episodic learning system."""
    import json

    from hive.learning import get_episodic_learner

    learner = get_episodic_learner()

    if action == "stats":
        stats = learner.get_statistics()

        # Build stats display
        content = (
            f"[bold]Total Episodes:[/bold] {stats['total_episodes']}\n"
            f"[bold]Completed:[/bold] {stats['completed_episodes']}\n"
            f"[bold]Unique Profiles:[/bold] {stats['unique_profiles']}\n"
            f"[bold]Learning Rate:[/bold] {stats['learning_rate']}\n\n"
            f"[bold]Topology Performance:[/bold]\n"
        )

        for topology, data in stats.get("topology_stats", {}).items():
            content += (
                f"  [cyan]{topology}[/cyan]: "
                f"{data['count']} uses, "
                f"avg score: {data['avg_score']:.2f}, "
                f"success: {data['success_rate']:.0%}\n"
            )

        if not stats.get("topology_stats"):
            content += "  [dim]No episodes recorded yet[/dim]\n"

        console.print(Panel(content, title="📚 Episodic Learning Statistics"))

    elif action == "clear":
        confirm = typer.confirm("Clear all learning data?")
        if confirm:
            learner._episodes.clear()
            learner._preferences.clear()
            learner._save()
            console.print("[green]Learning data cleared[/green]")

    elif action == "export":
        if not output:
            output = ".titan/learning_export.json"

        data = {
            "statistics": learner.get_statistics(),
            "episodes": [e.to_dict() for e in learner._episodes],
            "preferences": {
                k: {t.value: p.to_dict() for t, p in v.items()}
                for k, v in learner._preferences.items()
            },
        }

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(data, f, indent=2)

        console.print(f"[green]Exported learning data to {output}[/green]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: stats, clear, export")


@app.command()
def events(
    action: str = typer.Argument("history", help="Action: history, clear"),
    event_type: str = typer.Option(None, "--type", "-t", help="Filter by event type"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of events to show"),
) -> None:
    """View event history."""
    from hive.events import EventType, get_event_bus

    event_bus = get_event_bus()

    if action == "history":
        # Filter by type if specified
        filter_type = None
        if event_type:
            try:
                filter_type = EventType(event_type)
            except ValueError:
                console.print(f"[red]Unknown event type: {event_type}[/red]")
                console.print(f"Available: {', '.join(e.value for e in EventType)}")
                return

        events_list = event_bus.get_history(event_type=filter_type, limit=limit)

        if not events_list:
            console.print("[dim]No events recorded[/dim]")
            return

        table = Table(title=f"Event History (last {len(events_list)})")
        table.add_column("Time", style="dim")
        table.add_column("Type", style="cyan")
        table.add_column("Source")
        table.add_column("Payload", max_width=50)

        for event in events_list:
            payload_text = str(event.payload)
            payload_str = payload_text[:47] + "..." if len(payload_text) > 50 else payload_text
            table.add_row(
                event.timestamp.strftime("%H:%M:%S"),
                event.event_type.value,
                event.source_id or "-",
                payload_str,
            )

        console.print(table)

    elif action == "clear":
        event_bus.clear_history()
        console.print("[green]Event history cleared[/green]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: history, clear")


# ============================================================================
# Phase 4: Scale & Polish Commands
# ============================================================================


@app.command()
def stress(
    scenario: str = typer.Argument(
        "swarm",
        help="Scenario: swarm, pipeline, hierarchy, chaos, scale",
    ),
    agents: int = typer.Option(50, "--agents", "-a", help="Number of agents"),
    duration: int = typer.Option(60, "--duration", "-d", help="Duration in seconds"),
    max_concurrent: int = typer.Option(20, "--concurrent", "-c", help="Max concurrent agents"),
    failure_rate: float = typer.Option(
        0.0,
        "--failure-rate",
        "-f",
        help="Failure injection rate (0-1)",
    ),
    output: str = typer.Option(None, "--output", "-o", help="Output file for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run stress tests against the agent swarm."""
    from titan.stress import StressTestConfig, StressTestRunner
    from titan.stress.scenarios import list_scenarios

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show available scenarios
    if scenario == "list":
        table = Table(title="Available Stress Test Scenarios")
        table.add_column("Name", style="cyan")
        table.add_column("Description")

        for s in list_scenarios():
            table.add_row(s["name"], s["description"])

        console.print(table)
        return

    print_banner()
    say(ORCHESTRATOR, f"Starting stress test: {scenario}")

    async def run_stress() -> None:
        # Initialize HiveMind if available
        hive = None
        try:
            hive = HiveMind()
            await hive.initialize()
        except Exception as e:
            console.print(f"[yellow]HiveMind not available: {e}[/yellow]")

        # Configure test
        config = StressTestConfig(
            scenario_name=scenario,
            target_agents=agents,
            duration_seconds=duration,
            max_concurrent=max_concurrent,
            failure_rate=failure_rate,
            verbose=verbose,
        )

        console.print(
            Panel(
                f"[bold]Scenario:[/bold] {scenario}\n"
                f"[bold]Target Agents:[/bold] {agents}\n"
                f"[bold]Duration:[/bold] {duration}s\n"
                f"[bold]Max Concurrent:[/bold] {max_concurrent}\n"
                f"[bold]Failure Rate:[/bold] {failure_rate:.0%}",
                title="Stress Test Configuration",
            )
        )

        # Run test
        runner = StressTestRunner(config, hive_mind=hive)
        result = await runner.run()

        # Display results
        console.print("\n" + result.metrics.summary())

        if output:
            result.save(output)
            console.print(f"\n[green]Results saved to {output}[/green]")

        if hive:
            await hive.shutdown()

    try:
        asyncio.run(run_stress())
    except KeyboardInterrupt:
        say(ORCHESTRATOR, "Stress test interrupted")


@app.command()
def dashboard(
    action: str = typer.Argument("serve", help="Action: serve"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    """Start the web dashboard."""
    from dashboard.app import run_dashboard

    if action == "serve":
        print_banner()
        say(ORCHESTRATOR, f"Starting dashboard on http://{host}:{port}")
        console.print("\nEndpoints:")
        console.print(f"  Dashboard: http://{host}:{port}")
        console.print(f"  API: http://{host}:{port}/api")
        console.print(f"  WebSocket: ws://{host}:{port}/ws")
        console.print("\nPress Ctrl+C to stop\n")

        run_dashboard(host=host, port=port, reload=reload)

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: serve")


@app.command()
def metrics(
    action: str = typer.Argument("serve", help="Action: serve, show"),
    port: int = typer.Option(9100, "--port", "-p", help="Metrics port"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
) -> None:
    """Manage Prometheus metrics."""
    from titan.metrics import (
        get_metrics_text,
        start_metrics_server,
    )

    metrics_available = not get_metrics_text().startswith("# prometheus_client not installed")

    if action == "serve":
        if not metrics_available:
            console.print("[red]prometheus_client not installed[/red]")
            console.print("Install with: pip install prometheus-client")
            return
        say(ORCHESTRATOR, f"Starting metrics server on http://{host}:{port}/metrics")
        console.print("\nScrape configuration for Prometheus:")
        console.print(
            f"  - job_name: 'titan'\n    static_configs:\n      - targets: ['{host}:{port}']"
        )
        console.print("\nPress Ctrl+C to stop\n")

        try:
            start_metrics_server(port=port, host=host)
            # Keep the server running
            import time

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            say(ORCHESTRATOR, "Metrics server stopped")

    elif action == "show":
        # Show current metrics
        metrics_text = get_metrics_text()
        console.print(Panel(metrics_text[:5000], title="Current Metrics"))

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: serve, show")


@app.command()
def observe(
    action: str = typer.Argument("start", help="Action: start, stop, status"),
) -> None:
    """Start observability stack (Prometheus + Grafana)."""
    import subprocess

    deploy_dir = Path(__file__).parent.parent / "deploy"

    if action == "start":
        say(ORCHESTRATOR, "Starting observability stack...")
        try:
            result = subprocess.run(
                ["docker", "compose", "--profile", "monitoring", "up", "-d"],
                cwd=deploy_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                console.print("[green]Observability stack started![/green]")
                console.print("\nServices:")
                console.print("  Prometheus: http://localhost:9090")
                console.print("  Grafana: http://localhost:3000 (admin/titan)")
            else:
                console.print(f"[red]Failed to start: {result.stderr}[/red]")
        except FileNotFoundError:
            console.print("[red]Docker not found. Please install Docker.[/red]")

    elif action == "stop":
        say(ORCHESTRATOR, "Stopping observability stack...")
        try:
            subprocess.run(
                ["docker", "compose", "--profile", "monitoring", "down"],
                cwd=deploy_dir,
                capture_output=True,
            )
            console.print("[green]Observability stack stopped[/green]")
        except FileNotFoundError:
            console.print("[red]Docker not found[/red]")

    elif action == "status":
        try:
            result = subprocess.run(
                ["docker", "compose", "ps", "--format", "table {{.Name}}\t{{.Status}}\t{{.Ports}}"],
                cwd=deploy_dir,
                capture_output=True,
                text=True,
            )
            console.print(result.stdout)
        except FileNotFoundError:
            console.print("[red]Docker not found[/red]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: start, stop, status")


# ============================================================================
# MCP Server
# ============================================================================


@app.command()
def mcp(
    action: str = typer.Argument("run", help="Action: run, test"),
) -> None:
    """
    Run the Titan MCP Server.

    This exposes agents via the Model Context Protocol (JSON-RPC over stdio).
    Claude Code and other MCP clients can spawn and manage agents.

    Actions:
        run  - Start the MCP server on stdio
        test - Run a quick self-test
    """
    from titan_mcp.server import create_server, run_server

    if action == "run":
        say(ORCHESTRATOR, "Starting Titan MCP Server...")
        console.print("[cyan]MCP Server ready on stdio[/cyan]")
        console.print("Connect via Claude Code or other MCP clients")
        console.print("")
        console.print("Available tools:")
        console.print("  • spawn_agent - Create agents (researcher, coder, reviewer, orchestrator)")
        console.print("  • agent_status - Check agent progress")
        console.print("  • agent_result - Get completed results")
        console.print("  • list_agents - List active sessions")
        console.print("  • cancel_agent - Cancel running agents")
        console.print("")

        # Run the server
        run_server()

    elif action == "test":
        say(ORCHESTRATOR, "Testing MCP Server...")

        async def run_test() -> None:
            import json

            server = create_server()

            # Test initialize
            from titan_mcp.server import MCPRequest

            init_req = MCPRequest(
                jsonrpc="2.0",
                id=1,
                method="initialize",
                params={"protocolVersion": "2024-11-05"},
            )
            resp = await server.handle_request(init_req)
            console.print(f"[green]✓[/green] Initialize: {resp.result['serverInfo']['name']}")

            # Test tools/list
            tools_req = MCPRequest(
                jsonrpc="2.0",
                id=2,
                method="tools/list",
                params={},
            )
            resp = await server.handle_request(tools_req)
            tool_names = [t["name"] for t in resp.result["tools"]]
            console.print(f"[green]✓[/green] Tools: {', '.join(tool_names)}")

            # Test resources/list
            res_req = MCPRequest(
                jsonrpc="2.0",
                id=3,
                method="resources/list",
                params={},
            )
            resp = await server.handle_request(res_req)
            res_names = [r["name"] for r in resp.result["resources"]]
            console.print(f"[green]✓[/green] Resources: {', '.join(res_names)}")

            # Test spawn_agent (simulated, won't actually run LLM)
            spawn_req = MCPRequest(
                jsonrpc="2.0",
                id=4,
                method="tools/call",
                params={
                    "name": "spawn_agent",
                    "arguments": {
                        "agent_type": "simple",
                        "task": "Test task",
                    },
                },
            )
            resp = await server.handle_request(spawn_req)
            content = json.loads(resp.result["content"][0]["text"])
            console.print(f"[green]✓[/green] Spawn: session {content['session_id']}")

            # Test list_agents
            list_req = MCPRequest(
                jsonrpc="2.0",
                id=5,
                method="tools/call",
                params={
                    "name": "list_agents",
                    "arguments": {},
                },
            )
            resp = await server.handle_request(list_req)
            agents = json.loads(resp.result["content"][0]["text"])
            console.print(f"[green]✓[/green] List: {len(agents)} active agents")

            console.print("")
            console.print("[green]All MCP tests passed![/green]")

        asyncio.run(run_test())

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available: run, test")


# ============================================================================
# Inquiry Commands
# ============================================================================

inquiry_app = typer.Typer(
    name="inquiry",
    help="Multi-perspective inquiry commands",
)
app.add_typer(inquiry_app, name="inquiry")


@inquiry_app.command("start")
def inquiry_start(
    topic: str = typer.Argument(..., help="Topic to explore"),
    workflow: str = typer.Option(
        "expansive",
        "--workflow",
        "-w",
        help="Workflow: expansive, quick, creative",
    ),
    run: bool = typer.Option(False, "--run", "-r", help="Run immediately"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Start a new inquiry session."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from titan.workflows.inquiry_config import get_workflow
    from titan.workflows.inquiry_engine import InquiryStatus, get_inquiry_engine

    async def run_inquiry() -> None:
        wf = get_workflow(workflow)
        if not wf:
            console.print(f"[red]Unknown workflow: {workflow}[/red]")
            console.print("Available: expansive, quick, creative")
            return

        engine = get_inquiry_engine()
        session = await engine.start_inquiry(topic, wf)

        console.print(
            Panel(
                f"[bold]Session ID:[/bold] {session.id}\n"
                f"[bold]Topic:[/bold] {topic}\n"
                f"[bold]Workflow:[/bold] {wf.name}\n"
                f"[bold]Stages:[/bold] {session.total_stages}",
                title="Inquiry Session Created",
            )
        )

        if run:
            say(ORCHESTRATOR, "Running inquiry workflow...")

            async for event in engine.stream_workflow(session):
                event_type = event.get("type", "")

                if event_type == "stage_started":
                    stage_name = event.get("stage_name", "")
                    stage_idx = event.get("stage_index", 0) + 1
                    console.print(f"  [cyan]Stage {stage_idx}:[/cyan] {stage_name}...")

                elif event_type == "stage_completed":
                    result = event.get("result", {})
                    console.print(f"    [green]✓[/green] {result.get('stage_name', '')} completed")

                elif event_type == "session_completed":
                    results_count = event.get("results_count", 0)
                    console.print(f"\n[green]Inquiry completed![/green] {results_count} stages")

                elif event_type == "session_failed":
                    console.print(f"[red]Inquiry failed: {event.get('error', '')}[/red]")

            # Show summary
            if session.status == InquiryStatus.COMPLETED:
                table = Table(title="Results Summary")
                table.add_column("Stage", style="cyan")
                table.add_column("Role")
                table.add_column("Model")
                table.add_column("Duration")

                for r in session.results:
                    table.add_row(
                        r.stage_name,
                        r.role,
                        r.model_used,
                        f"{r.duration_ms}ms",
                    )

                console.print(table)
        else:
            console.print("\n[dim]Use 'titan inquiry status <session_id>' to check progress[/dim]")
            console.print("[dim]Use 'titan inquiry run <session_id>' to run the workflow[/dim]")

    asyncio.run(run_inquiry())


@inquiry_app.command("status")
def inquiry_status(
    session_id: str = typer.Argument(..., help="Session ID to check"),
) -> None:
    """Check the status of an inquiry session."""
    from titan.workflows.inquiry_engine import get_inquiry_engine

    engine = get_inquiry_engine()
    session = engine.get_session(session_id)

    if not session:
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise typer.Exit(1)

    status_color = {
        "pending": "yellow",
        "running": "blue",
        "paused": "magenta",
        "completed": "green",
        "failed": "red",
        "cancelled": "dim",
    }.get(session.status.value, "white")

    progress = f"{session.progress:.0f}% ({len(session.results)}/{session.total_stages} stages)"
    console.print(
        Panel(
            f"[bold]Session:[/bold] {session.id}\n"
            f"[bold]Topic:[/bold] {session.topic}\n"
            f"[bold]Workflow:[/bold] {session.workflow.name}\n"
            f"[bold]Status:[/bold] [{status_color}]{session.status.value}[/{status_color}]\n"
            f"[bold]Progress:[/bold] {progress}",
            title="Inquiry Session Status",
        )
    )

    if session.results:
        table = Table(title="Completed Stages")
        table.add_column("Stage", style="cyan")
        table.add_column("Model")
        table.add_column("Status")

        for r in session.results:
            status_icon = "[green]✓[/green]" if r.success else "[red]✗[/red]"
            table.add_row(r.stage_name, r.model_used, status_icon)

        console.print(table)


@inquiry_app.command("list")
def inquiry_list(
    status_filter: str = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max sessions to show"),
) -> None:
    """List inquiry sessions."""
    from titan.workflows.inquiry_engine import InquiryStatus, get_inquiry_engine

    engine = get_inquiry_engine()

    status = None
    if status_filter:
        try:
            status = InquiryStatus(status_filter)
        except ValueError:
            console.print(f"[red]Unknown status: {status_filter}[/red]")
            return

    sessions = engine.list_sessions(status)[:limit]

    if not sessions:
        console.print("[dim]No inquiry sessions found[/dim]")
        return

    table = Table(title=f"Inquiry Sessions ({len(sessions)})")
    table.add_column("ID", style="cyan")
    table.add_column("Topic", max_width=40)
    table.add_column("Workflow")
    table.add_column("Status")
    table.add_column("Progress")

    for s in sessions:
        status_color = {
            "completed": "green",
            "running": "blue",
            "failed": "red",
            "paused": "magenta",
        }.get(s.status.value, "dim")

        table.add_row(
            s.id,
            s.topic[:37] + "..." if len(s.topic) > 40 else s.topic,
            s.workflow.name,
            f"[{status_color}]{s.status.value}[/{status_color}]",
            f"{s.progress:.0f}%",
        )

    console.print(table)


@inquiry_app.command("compare")
def inquiry_compare(
    id1: str = typer.Argument(..., help="First session ID"),
    id2: str = typer.Argument(..., help="Second session ID"),
    format_output: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
) -> None:
    """Compare two inquiry sessions."""
    import json

    from titan.workflows.inquiry_engine import get_inquiry_engine

    engine = get_inquiry_engine()
    session1 = engine.get_session(id1)
    session2 = engine.get_session(id2)

    if not session1:
        console.print(f"[red]Session not found: {id1}[/red]")
        raise typer.Exit(1)

    if not session2:
        console.print(f"[red]Session not found: {id2}[/red]")
        raise typer.Exit(1)

    if format_output == "json":
        comparison = {
            "session_1": {
                "id": session1.id,
                "topic": session1.topic,
                "workflow": session1.workflow.name,
                "status": session1.status.value,
                "stages_completed": len(session1.results),
            },
            "session_2": {
                "id": session2.id,
                "topic": session2.topic,
                "workflow": session2.workflow.name,
                "status": session2.status.value,
                "stages_completed": len(session2.results),
            },
        }
        console.print(json.dumps(comparison, indent=2))
    else:
        table = Table(title="Session Comparison")
        table.add_column("Attribute", style="cyan")
        table.add_column(f"Session 1 ({id1[:8]}...)")
        table.add_column(f"Session 2 ({id2[:8]}...)")

        table.add_row("Topic", session1.topic[:30], session2.topic[:30])
        table.add_row("Workflow", session1.workflow.name, session2.workflow.name)
        table.add_row("Status", session1.status.value, session2.status.value)
        table.add_row(
            "Stages",
            f"{len(session1.results)}/{session1.total_stages}",
            f"{len(session2.results)}/{session2.total_stages}",
        )

        console.print(table)


@inquiry_app.command("export")
def inquiry_export(
    session_id: str = typer.Argument(..., help="Session ID to export"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    format_output: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: markdown, json",
    ),
) -> None:
    """Export inquiry session results."""
    import json

    from titan.workflows.inquiry_engine import get_inquiry_engine

    engine = get_inquiry_engine()
    session = engine.get_session(session_id)

    if not session:
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise typer.Exit(1)

    if format_output == "json":
        content = json.dumps(session.to_dict(), indent=2, default=str)
    else:
        # Markdown format
        lines = [
            f"# Inquiry: {session.topic}",
            "",
            f"**Session ID:** {session.id}",
            f"**Workflow:** {session.workflow.name}",
            f"**Status:** {session.status.value}",
            f"**Created:** {session.created_at.isoformat()}",
            "",
            "---",
            "",
        ]

        for result in session.results:
            lines.extend(
                [
                    f"## {result.stage_name}",
                    "",
                    f"**Role:** {result.role}",
                    f"**Model:** {result.model_used}",
                    "",
                    result.content,
                    "",
                    "---",
                    "",
                ]
            )

        content = "\n".join(lines)

    if output:
        Path(output).write_text(content)
        console.print(f"[green]Exported to {output}[/green]")
    else:
        console.print(content)


# ============================================================================
# Knowledge Commands
# ============================================================================

knowledge_app = typer.Typer(
    name="knowledge",
    help="Knowledge graph and memory commands",
)
app.add_typer(knowledge_app, name="knowledge")


@knowledge_app.command("search")
def knowledge_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    tag: str = typer.Option(None, "--tag", "-t", help="Filter by tag"),
) -> None:
    """Search the knowledge base."""
    say(ORCHESTRATOR, f"Searching: {query}")

    async def run_search() -> None:
        hive = HiveMind()
        await hive.initialize()

        try:
            # Search using HiveMind's memory
            tag_filter = [tag] if tag else None
            results = await hive.recall(query, k=limit, tags=tag_filter)

            if not results:
                console.print("[dim]No results found[/dim]")
                return

            table = Table(title=f"Search Results ({len(results)})")
            table.add_column("Memory ID", style="cyan")
            table.add_column("Preview", max_width=50)
            table.add_column("Distance/Score")

            for r in results:
                memory_id = str(r.get("id", ""))
                content = str(r.get("content", ""))
                preview = content[:47] + "..." if len(content) > 47 else content
                metric = r.get("score", r.get("distance", 0.0))
                metric_text = f"{metric:.2f}" if isinstance(metric, int | float) else "-"
                table.add_row(memory_id, preview, metric_text)

            console.print(table)

        finally:
            await hive.shutdown()

    asyncio.run(run_search())


@knowledge_app.command("stats")
def knowledge_stats() -> None:
    """Show knowledge base statistics."""

    async def show_stats() -> None:
        hive = HiveMind()
        await hive.initialize()

        try:
            health = await hive.health_check()

            console.print(
                Panel(
                    f"[bold]Redis:[/bold] {'✓' if health.get('redis') else '✗'}\n"
                    f"[bold]ChromaDB:[/bold] {'✓' if health.get('chroma') else '✗'}",
                    title="Knowledge Base Status",
                )
            )

        finally:
            await hive.shutdown()

    asyncio.run(show_stats())


@knowledge_app.command("export")
def knowledge_export(
    output: str = typer.Argument(..., help="Output file path"),
    format_output: str = typer.Option("json", "--format", "-f", help="Output format: json"),
) -> None:
    """Export knowledge base."""
    import json

    async def run_export() -> None:
        hive = HiveMind()
        await hive.initialize()

        try:
            # Get all keys (basic export)
            data = {
                "exported_at": datetime.now().isoformat(),
                "format_version": "1.0",
                "entries": [],
            }

            Path(output).write_text(json.dumps(data, indent=2))
            console.print(f"[green]Exported to {output}[/green]")

        finally:
            await hive.shutdown()

    asyncio.run(run_export())


# ============================================================================
# Workflow Commands
# ============================================================================

workflow_app = typer.Typer(
    name="workflow",
    help="Workflow management commands",
)
app.add_typer(workflow_app, name="workflow")


@workflow_app.command("list")
def workflow_list() -> None:
    """List available workflows."""
    from titan.workflows.inquiry_config import DEFAULT_WORKFLOWS

    table = Table(title="Available Workflows")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Stages")

    for name, wf in DEFAULT_WORKFLOWS.items():
        table.add_row(name, wf.description[:50] + "...", str(len(wf.stages)))

    console.print(table)


@workflow_app.command("execute")
def workflow_execute(
    name: str = typer.Argument(..., help="Workflow name"),
    topic: str = typer.Option(..., "--topic", "-t", help="Topic to explore"),
    mode: str = typer.Option(
        "staged",
        "--mode",
        "-m",
        help="Execution mode: sequential, parallel, staged",
    ),
) -> None:
    """Execute a workflow."""
    from titan.workflows.inquiry_config import get_workflow
    from titan.workflows.inquiry_dag import ExecutionMode
    from titan.workflows.inquiry_engine import get_inquiry_engine

    wf = get_workflow(name)
    if not wf:
        console.print(f"[red]Unknown workflow: {name}[/red]")
        return

    try:
        exec_mode = ExecutionMode(mode)
    except ValueError:
        console.print(f"[red]Unknown mode: {mode}[/red]")
        console.print("Available: sequential, parallel, staged")
        return

    async def run_workflow() -> None:
        engine = get_inquiry_engine()
        session = await engine.start_inquiry(topic, wf)

        say(ORCHESTRATOR, f"Running {name} workflow in {mode} mode...")

        session = await engine.run_dag_workflow(session, exec_mode)

        status_icon = "[green]✓[/green]" if session.status.value == "completed" else "[red]✗[/red]"
        console.print(f"{status_icon} Workflow {session.status.value}")
        console.print(f"Session ID: {session.id}")

    asyncio.run(run_workflow())


@workflow_app.command("visualize")
def workflow_visualize(
    name: str = typer.Argument(..., help="Workflow name"),
    format_output: str = typer.Option(
        "mermaid",
        "--format",
        "-f",
        help="Output format: mermaid, text",
    ),
) -> None:
    """Visualize a workflow structure."""
    from titan.workflows.inquiry_config import get_workflow

    wf = get_workflow(name)
    if not wf:
        console.print(f"[red]Unknown workflow: {name}[/red]")
        return

    if format_output == "mermaid":
        lines = ["```mermaid", "graph TD"]

        for i, stage in enumerate(wf.stages):
            node_id = f"S{i}"
            lines.append(f'    {node_id}["{stage.emoji} {stage.name}"]')

            # Add dependencies if present
            if stage.dependencies:
                for dep in stage.dependencies:
                    lines.append(f"    S{dep} --> {node_id}")
            elif i > 0:
                # Default sequential connection
                lines.append(f"    S{i - 1} --> {node_id}")

        lines.append("```")
        console.print("\n".join(lines))

    else:
        # Text format
        for i, stage in enumerate(wf.stages):
            prefix = "└──" if i == len(wf.stages) - 1 else "├──"
            console.print(f"  {prefix} {stage.emoji} {stage.name} ({stage.role})")


@workflow_app.command("validate")
def workflow_validate(
    file_path: str = typer.Argument(..., help="Workflow YAML file to validate"),
) -> None:
    """Validate a workflow configuration file."""
    import yaml

    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    try:
        with open(path) as f:
            data = yaml.safe_load(f)

        # Validate structure
        errors = []

        if "name" not in data:
            errors.append("Missing 'name' field")

        if "stages" not in data:
            errors.append("Missing 'stages' field")
        elif not isinstance(data["stages"], list):
            errors.append("'stages' must be a list")
        elif len(data["stages"]) == 0:
            errors.append("Workflow must have at least one stage")

        if errors:
            console.print("[red]Validation failed:[/red]")
            for e in errors:
                console.print(f"  • {e}")
            raise typer.Exit(1)

        console.print(f"[green]✓ Workflow '{data.get('name', 'unnamed')}' is valid[/green]")
        console.print(f"  Stages: {len(data.get('stages', []))}")

    except yaml.YAMLError as e:
        console.print(f"[red]Invalid YAML: {e}[/red]")
        raise typer.Exit(1)


# ============================================================================
# Entry Point
# ============================================================================


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
