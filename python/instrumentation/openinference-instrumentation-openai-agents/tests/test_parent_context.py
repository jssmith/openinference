"""Tests for parent context propagation in OpenInferenceTracingProcessor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from agents.tracing import trace as agents_trace
from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import set_span_in_context

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from openinference.instrumentation.openai_agents._processor import (
    OpenInferenceTracingProcessor,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_sdk.TracerProvider:
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return provider


@pytest.fixture
def instrument(tracer_provider: trace_sdk.TracerProvider) -> Iterator[None]:
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    OpenAIAgentsInstrumentor().uninstrument()


def test_respects_existing_otel_parent(
    instrument: None,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Verify OpenInferenceTracingProcessor respects existing parent context."""
    # Set the tracer provider as global so get_current_span() works
    trace_api.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)

    # Create a parent span and run agents trace within it
    with tracer.start_as_current_span("parent_span") as parent_span:
        parent_trace_id = parent_span.get_span_context().trace_id
        with agents_trace("child_agent_trace"):
            pass

    # Get all spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) >= 2, f"Expected at least 2 spans, got {len(spans)}"

    # Extract trace IDs
    trace_ids = {span.context.trace_id for span in spans}

    # All spans should share the same trace ID (the parent's)
    assert len(trace_ids) == 1, f"Expected all spans to share trace ID, got {trace_ids}"
    assert parent_trace_id in trace_ids, "Spans should use parent's trace ID"


def test_creates_new_trace_when_no_parent(
    instrument: None,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Verify processor creates a new trace when no parent context exists."""
    # Set the tracer provider as global
    trace_api.set_tracer_provider(tracer_provider)

    # Run agents trace without any parent
    with agents_trace("standalone_trace"):
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) >= 1, "Expected at least 1 span"

    # Verify a trace was created
    trace_ids = {span.context.trace_id for span in spans}
    assert len(trace_ids) == 1, "All spans should share the same trace ID"


def test_get_parent_context_hook_can_be_overridden(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Verify _get_parent_context can be overridden by subclasses."""
    # Set the tracer provider as global
    trace_api.set_tracer_provider(tracer_provider)
    tracer = tracer_provider.get_tracer(__name__)

    # Create a span that will be the custom parent
    custom_parent = tracer.start_span("custom_parent")
    custom_parent_ctx = set_span_in_context(custom_parent)
    custom_parent_trace_id = custom_parent.get_span_context().trace_id

    class CustomProcessor(OpenInferenceTracingProcessor):
        """Processor that always returns a custom parent context."""

        def _get_parent_context(self) -> Context | None:
            return custom_parent_ctx

    # Create and register custom processor
    from agents.tracing import add_trace_processor

    processor = CustomProcessor(tracer)
    add_trace_processor(processor)

    # Run a trace - it should use our custom parent
    # Note: We can't easily remove processors, but that's fine for this test
    with agents_trace("test_trace"):
        pass

    custom_parent.end()

    # Get all spans
    spans = in_memory_span_exporter.get_finished_spans()

    # Find spans created by our custom processor (they will have custom_parent's trace ID)
    child_spans = [
        s for s in spans
        if s.context.trace_id == custom_parent_trace_id and s.name == "test_trace"
    ]

    # The test span should have the custom parent's trace ID
    assert len(child_spans) >= 1, (
        f"Expected at least 1 span with custom parent's trace ID {custom_parent_trace_id:x}, "
        f"got spans: {[(s.name, hex(s.context.trace_id)) for s in spans]}"
    )
