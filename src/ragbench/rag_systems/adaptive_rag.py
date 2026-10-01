from __future__ import annotations

from ragbench.agents import prompts
from ragbench.agents.router import RouteDecision, decide_from_reply, describe_routes, heuristic_route
from ragbench.config.schema import SystemConfig
from ragbench.documents.schema import Document
from ragbench.models.cost import CostBreakdown
from ragbench.rag_systems.base import AnswerResult, BaseRAGSystem, IngestionResult, RetrievalResult
from ragbench.rag_systems.options import AdaptiveOptions
from ragbench.rag_systems.spec import SystemSpec
from ragbench.rag_systems.trace import Step, Tracer, activate, reconcile_steps, sum_costs
from ragbench.registry import SYSTEMS
from ragbench.utils.timing import timer

ROUTE_MAX_TOKENS = 120


@SYSTEMS.register("adaptive")
class AdaptiveRAG(BaseRAGSystem):
    """A per-question router: each question is sent to the one configured pipeline best suited to it (`routes:`).

    Every route is a complete system built from its own inline config and ingested once, so the adaptive system costs what its routes cost to
    index (embeddings are shared across routes through the run's embedding cache) plus, per question, the router and the chosen pipeline. The
    chosen route's steps are copied into this system's trace, tagged with the route, so costs and stages add up exactly. Whether routing pays
    off is visible in `routes.csv`: per route, how many questions it took and how they scored.
    """

    spec = SystemSpec(
        type="adaptive",
        title="Adaptive router",
        summary="Routes each question to the best of several configured pipelines (exact identifiers to BM25, comparisons to decomposition, calculations to a tool agent, the rest to the default)",
        best_for="Mixed workloads where no single architecture is best for every question",
        cost_profile="high",
        latency_profile="medium",
        requires_llm=True,
        agentic=False,
        options=AdaptiveOptions,
        chunker=None,
    )
    options: AdaptiveOptions

    def __init__(self, config: SystemConfig, force_mock: bool = False):
        super().__init__(config, force_mock=force_mock)
        self.systems: dict[str, BaseRAGSystem] = {}
        for route, route_config in self.options.routes.items():
            # A route inherits the adaptive system's models (embedding, generator) unless it sets its own.
            named = route_config.model_copy(update={"name": f"{self.name}/{route}", "models": {**config.models, **route_config.models}})
            self.systems[route] = SYSTEMS.get(named.type)(config=named, force_mock=force_mock)

    @classmethod
    def offered_tools(cls, config: SystemConfig) -> list[str]:
        names: list[str] = []
        for route_config in AdaptiveOptions.model_validate(config.retrieval).routes.values():
            names += [tool for tool in SYSTEMS.get(route_config.type).offered_tools(route_config) if tool not in names]
        return names

    def ingest(self, documents: list[Document]) -> IngestionResult:
        cost = CostBreakdown()
        chunks = 0
        per_route: dict[str, int] = {}
        with timer() as t:
            for route, system in self.systems.items():
                result = system.ingest(documents)
                cost, chunks, per_route[route] = cost.plus(result.cost), chunks + result.num_chunks, result.num_chunks
        return IngestionResult(
            system=self.name,
            num_documents=len(documents),
            num_chunks=chunks,
            latency_ms=t.elapsed_ms,
            cost=cost,
            metadata={"chunks_by_route": per_route, "expensive": any(system.spec is not None and system.spec.cost_profile == "high" for system in self.systems.values())},
        )

    # -- routing ---------------------------------------------------------------------------------------------------------

    def _route(self, question: str) -> RouteDecision:
        """Pick a route, recorded as one `route` step (with the LLM call's cost when `router: llm`)."""
        with self.trace.step("route", "route_question", router=self.options.router) as step:
            step.set_input(question)
            if self.options.router == "heuristic":
                decision = heuristic_route(question, self.systems)
            else:
                messages = prompts.route_question(question, describe_routes(self.systems, self.options.route_descriptions))
                result = self.llm.generate(messages, temperature=0, json_mode=True, max_tokens=ROUTE_MAX_TOKENS)
                step.set_llm(result, messages, cost=CostBreakdown(query_rewrite_cost=result.cost.total_cost))
                decision = decide_from_reply(result.text, self.systems)
            step.set_output(f"{decision.route}: {decision.reason}")
            step.set_meta(route=decision.route, reason=decision.reason, fallback=decision.fallback)
        return decision

    def fetch_context(self, question: str, top_k: int | None = None) -> RetrievalResult:
        with self._recording() as tracer, timer() as t:
            first_step = len(tracer.steps)
            decision = self._route(question)
            retrieval = self.systems[decision.route].fetch_context(question, top_k=top_k)  # records its own steps in the active tracer
            steps = tracer.steps[first_step:]
        return retrieval.model_copy(
            update={
                "latency_ms": t.elapsed_ms,
                "cost": sum_costs(step.cost for step in steps),
                "metadata": {**retrieval.metadata, "route": decision.route, "route_reason": decision.reason},
                "steps": steps,
            }
        )

    def answer_question(self, question: str, top_k: int | None = None, context_k: int | None = None) -> AnswerResult:
        tracer = Tracer()
        with activate(tracer), timer() as routing:
            decision = self._route(question)
        routing_steps = len(tracer.steps)
        # The chosen system answers on its own tracer; its steps join this question's trace, tagged with the route, so costs add up exactly.
        answer = self.systems[decision.route].answer_question(question, top_k=top_k, context_k=context_k)
        for step in answer.steps:
            tracer.add(self._tagged(step, decision.route))
        cost = sum_costs(step.cost for step in tracer.steps)
        steps = reconcile_steps(tracer.steps, cost)
        retrieval_steps = [step for step in steps if step.kind != "generate"]
        retrieval_cost = sum_costs(step.cost for step in retrieval_steps)
        route_info = {"route": decision.route, "route_reason": decision.reason, "router": self.options.router, "router_fallback": decision.fallback}
        retrieval = answer.retrieval_result.model_copy(
            update={
                "latency_ms": routing.elapsed_ms + tracer.latency_credit_ms + answer.retrieval_result.latency_ms,
                "cost": retrieval_cost,
                "metadata": {**answer.retrieval_result.metadata, **route_info},
                "steps": retrieval_steps,
            }
        )
        return answer.model_copy(
            update={
                "retrieval_result": retrieval,
                "latency_ms": routing.elapsed_ms + tracer.latency_credit_ms + answer.latency_ms,
                "cost": cost,
                "metadata": {**answer.metadata, "system_type": self.config.type, **route_info, "routed_steps": routing_steps},
                "steps": steps,
            }
        )

    @staticmethod
    def _tagged(step: Step, route: str) -> Step:
        return step.model_copy(update={"metadata": {**step.metadata, "route": route}})
