from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunTelemetryCounter:
    """Optional helper. Use it or replace it with your own observability."""
    llm_calls: int = 0
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    model_tool_calls: int = 0
    tool_call_names: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, object]] = field(default_factory=list)

    def record_llm_call(self) -> None:
        self.llm_calls += 1

    def record_tool_call(self, name: str) -> None:
        self.tool_calls += 1
        self.tool_names.append(name)

    def record_model_tool_call(self, name: str) -> None:
        self.model_tool_calls += 1
        self.tool_call_names.append(name)

    def record_diagnostic(self, event: dict[str, object]) -> None:
        # Diagnostics contain response shape and tool names only, never prompts,
        # tool arguments, request text, or model response content.
        self.diagnostics.append(event)
