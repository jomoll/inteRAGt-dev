"""Shared DSPy-driven orchestration logic for lightweight clinical agents."""

from __future__ import annotations

import json
import logging
import re
import textwrap
from abc import ABC, abstractmethod
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from .signatures import (
    GermanClinicalSummary,
    PlanPatientQuestion,
    StructuredToolPlan,
    ToolExecutionStep,
)
from .toolkit import BaseTool, ToolOutput

logger = logging.getLogger(__name__)


class DSPyAgentBase(ABC):
    """Base class that coordinates plan → tool execution → summary via DSPy signatures."""

    def __init__(self, *, tools: Optional[Iterable[BaseTool]] = None, max_tool_rounds: int = 6) -> None:
        self.tools: List[BaseTool] = list(tools or [])
        self._max_tool_rounds = max_tool_rounds
        self._event_handler: Optional[Callable[[Dict[str, Any]], None]] = None
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._history: List[ChatMessage] = []
        self._default_filters: Dict[str, Any] = {}
        self._plan_text: str = ""
        self._plan_steps: List[Dict[str, Any]] = []
        self._analysis_text: str = ""
        self._required_information: List[str] = []
        self._missing_information: List[str] = []
        self._global_stop_conditions: List[str] = []
        self._event_handler = None

    @property
    def history(self) -> Sequence[ChatMessage]:
        return tuple(self._history)

    # ------------------------------------------------------------------
    # Event streaming helpers
    # ------------------------------------------------------------------
    def _emit_event(self, event_type: str, **payload: Any) -> None:
        if not self._event_handler:
            return
        event = {
            "type": event_type,
            "payload": payload,
            "timestamp": time.time(),
        }
        try:
            self._event_handler(event)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to dispatch agent event %s", event_type)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer_with_rag(
        self,
        question: str,
        *,
        patient_context: str = "",
        report_type: Optional[str] = None,
        report_date: Optional[str] = None,
        top_k: int = 5,
        event_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> ChatMessage:
        """Plan, execute tools, and summarise a clinical answer."""

        self.reset()
        self._event_handler = event_handler

        self._default_filters = {
            "query": question.strip(),
            "top_k": top_k,
            "report_type": report_type,
            "report_date": report_date,
        }

        self._emit_event(
            "run_started",
            question=question,
            patient_context=patient_context,
            report_type=report_type,
            report_date=report_date,
            top_k=top_k,
        )

        intro_message = self._format_intro_message(question, patient_context, report_type, report_date, top_k)
        self._append_user_message(intro_message)

        allowed_tools_payload = self._allowed_tools_payload(full=True)
        allowed_tool_names_json = self._allowed_tools_payload(full=False)

        assessment_result = self._run_signature(
            PlanPatientQuestion,
            {
                "question": question,
                "patient_context": patient_context or "Keine zusätzlichen Patientendaten.",
                "default_filters": json.dumps(self._format_tool_filters(self._default_filters), ensure_ascii=False),
                "allowed_tools": allowed_tools_payload,
            },
        )

        self._analysis_text = self._normalize_text(assessment_result.get("clinical_analysis")).strip()
        self._required_information = self._parse_json_list(assessment_result.get("required_information"))
        self._missing_information = self._parse_json_list(assessment_result.get("missing_information"))
        recommended_tools = self._parse_json_list(assessment_result.get("recommended_tools"))

        tool_plan_result = self._run_signature(
            StructuredToolPlan,
            {
                "question": question,
                "patient_context": patient_context or "Keine zusätzlichen Patientendaten.",
                "default_filters": json.dumps(self._format_tool_filters(self._default_filters), ensure_ascii=False),
                "allowed_tools": allowed_tool_names_json,
                "clinical_analysis": self._analysis_text,
                "required_information": json.dumps(self._required_information, ensure_ascii=False),
                "missing_information": json.dumps(self._missing_information, ensure_ascii=False),
                "recommended_tools": json.dumps(recommended_tools, ensure_ascii=False),
            },
        )

        plan_raw = tool_plan_result.get("tool_plan")
        self._plan_text = self._normalize_text(plan_raw)
        plan_object = self._parse_plan_object(self._plan_text)
        self._plan_steps = self._extract_plan_steps(plan_object)
        self._global_stop_conditions = self._parse_json_list(plan_object.get("global_stop_conditions")) if isinstance(plan_object, dict) else []
        if not self._plan_steps:
            self._apply_fallback_structured_plan(question, top_k)
        else:
            for step in self._plan_steps:
                if not step.get("tool_name"):
                    self._apply_fallback_structured_plan(question, top_k)
                    break

        self._emit_event(
            "plan_ready",
            analysis=self._analysis_text,
            required_information=self._required_information,
            missing_information=self._missing_information,
            recommended_tools=recommended_tools,
            plan_text=self._plan_text,
            plan_steps=self._plan_steps,
            global_stop_conditions=self._global_stop_conditions,
        )

        plan_overview_message = [
            "Tool plan initialised.",
            "\nPlan (JSON):",
            self._plan_text or "{}",
        ]
        self._append_assistant_message("\n".join(plan_overview_message), signature="StructuredToolPlan")

        collected_nodes: List[Dict[str, Any]] = []
        executed_actions: List[Dict[str, Any]] = []
        previous_summaries: List[str] = [self._analysis_text] if self._analysis_text else []
        applied_filter_notes: List[str] = []

        step_index = 0
        stop_conditions_json = json.dumps(self._global_stop_conditions, ensure_ascii=False)

        for round_idx in range(self._max_tool_rounds):
            total_steps = len(self._plan_steps)
            plan_chunk_obj = self._plan_steps[step_index] if step_index < total_steps else {}
            plan_chunk = json.dumps(plan_chunk_obj, ensure_ascii=False) if plan_chunk_obj else "{}"
            plan_chunk_pretty = (
                json.dumps(plan_chunk_obj, ensure_ascii=False, indent=2) if plan_chunk_obj else "{}"
            )
            previous_text = "\n".join(previous_summaries) or "Keine Werkzeuge ausgeführt."

            self._emit_event(
                "execution_step",
                round=round_idx + 1,
                step_index=step_index,
                total_steps=total_steps,
                plan_chunk=plan_chunk_obj,
            )

            action_result = self._run_signature(
                ToolExecutionStep,
                {
                    "question": question,
                    "patient_context": patient_context or "",
                    "plan_chunk": plan_chunk,
                    "previous_results": previous_text,
                    "allowed_tools": allowed_tool_names_json,
                    "global_stop_conditions": stop_conditions_json,
                },
            )
            action_label = self._normalize_text(action_result.get("action")).strip().lower()
            tool_name = self._normalize_text(action_result.get("tool_name")).strip()
            raw_arguments = action_result.get("arguments", "")
            rationale = self._normalize_text(action_result.get("rationale")).strip()

            self._emit_event(
                "execution_decision",
                round=round_idx + 1,
                step_index=step_index,
                action=action_label,
                tool_name=tool_name,
                arguments_raw=raw_arguments,
                rationale=rationale,
            )

            logger.debug(
                "Plan progress: step %d/%d, action=%s, tool=%s, stop_conditions=%s",
                step_index + 1,
                total_steps,
                action_label or "(leer)",
                tool_name or "(keins)",
                self._global_stop_conditions,
            )

            action_summary = self._format_action_summary(
                action_label,
                tool_name,
                raw_arguments,
                rationale,
                plan_chunk_pretty,
            )
            self._append_assistant_message(action_summary, signature="ToolExecutionStep", round=round_idx + 1)

            if action_label in {"", "finish"}:
                break

            if action_label == "skip":
                step_index += 1
                continue

            if action_label != "call_tool":
                previous_summaries.append(f"Unbekannte Aktion '{action_label}'.")
                step_index += 1
                continue

            arguments = self._parse_arguments(raw_arguments)
            self._emit_event(
                "tool_started",
                round=round_idx + 1,
                step_index=step_index,
                tool_name=tool_name,
                arguments=arguments,
            )
            nodes, action_record, summary_text = self._call_tool(tool_name, arguments)
            self._emit_event(
                "tool_finished",
                round=round_idx + 1,
                step_index=step_index,
                tool_name=tool_name,
                action_record=action_record,
                nodes=nodes,
                summary=summary_text,
            )
            executed_actions.append(action_record)

            error_occurred = action_record.get("is_error", False)

            if summary_text:
                previous_summaries.append(summary_text)

            if nodes:
                collected_nodes.extend(nodes)

            if error_occurred:
                names = self._available_tool_names()
                if names:
                    previous_summaries.append("Verfügbare Werkzeuge: " + ", ".join(names))
                if executed_actions:
                    last = executed_actions[-1]
                    if last.get("tool") == tool_name and last.get("arguments") == action_record.get("arguments"):
                        logger.warning(
                            "Duplicate failed tool call detected for %s with arguments %s",
                            tool_name,
                            action_record.get("arguments"),
                        )
                step_index += 1
                continue

            filters_text = self._summarise_filters_from_action(action_record)
            if filters_text:
                applied_filter_notes.append(filters_text)

            step_index += 1

        deduped_nodes = self._deduplicate_context(collected_nodes)
        plan_execution_summary = "\n".join(previous_summaries) or "Keine Werkzeuge wurden benötigt."
        applied_filters_summary = "\n".join(applied_filter_notes) or "Standardfilter ohne Änderungen."
        context_snippets = self._format_context_snippets(deduped_nodes)

        self._emit_event(
            "context_compiled",
            executed_actions=executed_actions,
            applied_filters_summary=applied_filters_summary,
            context_nodes=deduped_nodes,
            plan_execution_summary=plan_execution_summary,
        )

        overview_header = [
            f"Analyse: {self._analysis_text or '(keine)'}",
            "Benötigte Informationen: " + (", ".join(self._required_information) if self._required_information else "(keine)"),
            "Noch fehlende Informationen: " + (", ".join(self._missing_information) if self._missing_information else "(keine)"),
            "",
            "Ausführung:",
            plan_execution_summary,
        ]
        plan_overview_text = "\n".join(overview_header)

        self._log_context_metrics(deduped_nodes)

        summary_result = self._run_signature(
            GermanClinicalSummary,
            {
                "question": question,
                "patient_context": patient_context or "Keine zusätzlichen Patientendaten.",
                "plan_overview": plan_overview_text,
                "applied_filters": applied_filters_summary,
                "context_snippets": context_snippets or "Keine Kontextausschnitte gefunden.",
                "outstanding_information": json.dumps(self._missing_information, ensure_ascii=False),
            },
        )
        clinical_brief = self._normalize_text(summary_result.get("clinical_brief")).strip()
        detailed_answer = self._normalize_text(summary_result.get("detailed_answer")).strip()
        if not clinical_brief and not detailed_answer:
            combined_answer = self._fallback_answer(question, patient_context, deduped_nodes)
        else:
            sections = []
            if clinical_brief:
                sections.append(clinical_brief)
            if detailed_answer:
                sections.append(detailed_answer)
            combined_answer = "\n\n".join(sections)

        self._emit_event(
            "summary_ready",
            clinical_brief=clinical_brief,
            detailed_answer=detailed_answer,
            combined_answer=combined_answer,
            missing_information=self._missing_information,
            required_information=self._required_information,
        )

        final_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=combined_answer,
            additional_kwargs={
                "context_nodes": deduped_nodes,
                "actions": executed_actions,
                "subqueries": [
                    record.get("arguments", {}).get("query")
                    for record in executed_actions
                    if isinstance(record.get("arguments"), dict) and record.get("arguments", {}).get("query")
                ],
                "analysis": self._analysis_text,
                "required_information": self._required_information,
                "missing_information": self._missing_information,
                "clinical_brief": clinical_brief,
                "detailed_answer": detailed_answer,
            },
        )
        self._history.append(final_message)
        self._default_filters = {}
        self._emit_event(
            "run_completed",
            answer=combined_answer,
            metadata=final_message.additional_kwargs,
        )
        return final_message

    # ------------------------------------------------------------------
    # Signature helpers
    # ------------------------------------------------------------------
    def _run_signature(self, signature_cls: Type, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_signature_prompt(signature_cls, inputs)
        self._emit_event("signature_started", signature=signature_cls.__name__, inputs=inputs)
        try:
            completion = self._complete(prompt)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Signature %s failed: %s", signature_cls.__name__, exc)
            self._emit_event("signature_failed", signature=signature_cls.__name__, error=str(exc))
            return {}
        parsed = self._parse_signature_output(completion, signature_cls)
        logger.debug("Signature %s output: %s", signature_cls.__name__, parsed)
        self._emit_event(
            "signature_completed",
            signature=signature_cls.__name__,
            outputs=parsed,
        )
        return parsed

    def _build_signature_prompt(self, signature_cls: Type, inputs: Dict[str, Any]) -> str:
        instructions = getattr(signature_cls, "instructions", "")
        input_blocks: List[str] = []
        for name, field in signature_cls.input_fields.items():  # type: ignore[attr-defined]
            value = inputs.get(name, "")
            desc = field.json_schema_extra.get("desc", "")
            prefix = field.json_schema_extra.get("prefix", f"{name}:")
            formatted_value = textwrap.indent(str(value or ""), "    ").rstrip()
            block_lines = [f"{prefix} ({desc})", formatted_value or "    (leer)"]
            input_blocks.append("\n".join(block_lines))

        output_lines = []
        for name, field in signature_cls.output_fields.items():  # type: ignore[attr-defined]
            desc = field.json_schema_extra.get("desc", "")
            output_lines.append(f'- "{name}": {desc}')

        prompt_parts = [
            instructions,
            "",
            "Eingaben:",
            "\n\n".join(input_blocks) if input_blocks else "(keine Eingaben)",
            "",
            "Erzeuge eine JSON-Antwort mit genau diesen Feldern:",
            "\n".join(output_lines),
            "",
            "JSON:",
        ]
        return "\n".join(prompt_parts)

    def _parse_signature_output(self, completion: str, signature_cls: Type) -> Dict[str, Any]:
        candidate = self._extract_json_object(completion)
        if not isinstance(candidate, dict):
            return {}
        outputs: Dict[str, Any] = {}
        for name in signature_cls.output_fields.keys():  # type: ignore[attr-defined]
            outputs[name] = candidate.get(name)
        return outputs

    def _extract_json_object(self, text: str) -> Any:
        if not text:
            return {}
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        candidate = candidate.strip()
        if not candidate:
            return {}
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            collapsed = self._collapse_string_concatenation(candidate)
            if collapsed != candidate:
                try:
                    return json.loads(collapsed)
                except json.JSONDecodeError:
                    candidate = collapsed
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if match:
                fragment = self._collapse_string_concatenation(match.group(0))
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    logger.debug("Failed to parse JSON fragment: %s", candidate)
        return {}

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
        arguments = dict(arguments or {})
        action_record: Dict[str, Any] = {
            "tool": tool_name or "",
            "arguments": arguments,
        }

        if not tool_name:
            action_record["result"] = "Kein Werkzeugname angegeben."
            action_record["is_error"] = True
            return [], action_record, action_record["result"]

        try:
            tool = self._get_tool(tool_name)
        except ValueError as exc:
            error_text = str(exc)
            action_record["result"] = error_text
            action_record["is_error"] = True
            self._append_assistant_message(
                f"Werkzeug '{tool_name}' steht nicht zur Verfügung: {error_text}",
                tool=tool_name,
                error=True,
            )
            return [], action_record, error_text

        if getattr(self, "_default_filters", None) and tool_name == "retrieve_reports":
            for key, value in self._default_filters.items():
                if value is not None and key not in arguments:
                    arguments[key] = value

        if tool_name == "retrieve_reports" and isinstance(arguments.get("report_type"), list):
            report_list = [
                rtype for rtype in arguments["report_type"] if isinstance(rtype, str) and rtype.strip()
            ]
            if report_list:
                arguments["report_type"] = report_list
            else:
                arguments.pop("report_type", None)

        invocation_text = f"Verwende Werkzeug {tool_name} mit Parametern {arguments}."
        self._append_assistant_message(invocation_text, tool=tool_name, arguments=arguments)

        output = self._safe_tool_call(tool, arguments)
        payload = getattr(output, "raw_output", None)
        if not isinstance(payload, dict):
            payload = self._parse_json_response(getattr(output, "content", None))

        tool_message_content = self._safe_json_dumps(payload) or (output.content or "")
        self._append_tool_message(tool.metadata.name, payload or {}, tool_message_content)

        nodes: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            nodes = payload.get("context_nodes") or []
            if not isinstance(nodes, list):
                nodes = []

        summary_text = ""
        if nodes:
            summary_text = self._summarise_nodes(nodes)
            action_record["result_count"] = len(nodes)
        else:
            summary_text = self._summarise_tool_response(output)
            action_record["result"] = summary_text
        action_record["is_error"] = getattr(output, "is_error", False)

        return nodes, action_record, summary_text

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _format_intro_message(
        self,
        question: str,
        patient_context: str,
        report_type: Optional[str],
        report_date: Optional[str],
        top_k: int,
    ) -> str:
        lines = [
            "Aufgabe:",
            question,
            "",
            "Patientenkontext:",
            patient_context or "Keine zusätzlichen Patientendaten.",
            "",
            f"Standardfilter: report_type={report_type or '-'}, report_date={report_date or '-'}, top_k={top_k}",
        ]
        return "\n".join(lines)

    def _format_action_summary(
        self,
        action: str,
        tool_name: str,
        arguments: Any,
        rationale: str,
        plan_chunk: str,
    ) -> str:
        action_text = action or "-"
        tool_text = tool_name or "-"
        arguments_text = self._normalize_text(arguments) or "-"
        rationale_text = rationale or "-"
        plan_text = plan_chunk or "(leer)"
        return "\n".join(
            [
                f"Planabschnitt: {plan_text}",
                f"Aktionsvorschlag: {action_text}",
                f"Werkzeug: {tool_text}",
                f"Argumente: {arguments_text}",
                f"Begründung: {rationale_text}",
            ]
        )

    def _format_context_snippets(self, nodes: List[Dict[str, Any]], limit: int = 5, max_chars: int = 220) -> str:
        if not nodes:
            return ""
        snippets: List[str] = []
        for idx, node in enumerate(nodes[:limit], start=1):
            snippet = node.get("snippet") or node.get("text", "") or ""
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 3].rstrip() + "..."
            label = node.get("section_name") or node.get("report_type") or node.get("report_id") or "Abschnitt"
            snippets.append(f"[{idx}] ({label}) {snippet}")
        return "\n".join(snippets)

    def _summarise_filters_from_action(self, action_record: Dict[str, Any]) -> str:
        arguments = action_record.get("arguments")
        if not isinstance(arguments, dict):
            return ""
        if not arguments:
            return ""
        formatted = self._format_tool_filters(arguments)
        if not formatted:
            return ""
        return f"{action_record.get('tool', '')}: {json.dumps(formatted, ensure_ascii=False)}"

    def _fallback_plan(self, question: str) -> str:
        return (
            "1. Übersetze die Fragestellung ins Deutsche und identifiziere relevante Berichtstypen.\n"
            "2. Führe mindestens eine Suche mit 'retrieve_reports' durch.\n"
            "3. Fasse die Ergebnisse zusammen und beantworte die Frage."
        )

    def _apply_fallback_structured_plan(self, question: str, top_k: int) -> None:
        self._plan_steps = [
            {
                "step_number": 1,
                "objective": "Grundlagenrecherche in Arztbriefen zur Beantwortung der Frage.",
                "tool_name": "retrieve_reports",
                "arguments": {
                    "query": question.strip() or "klinische Zusammenfassung",
                    "top_k": max(3, top_k),
                },
                "evidence_required": ["Mindestens ein relevanter Berichtsausschnitt"],
                "stop_if": "",
            }
        ]
        self._plan_text = json.dumps({"steps": self._plan_steps}, ensure_ascii=False, indent=2)
        self._global_stop_conditions = []

    def _fallback_answer(self, question: str, patient_context: str, nodes: List[Dict[str, Any]]) -> str:
        if not nodes:
            return (
                "Ich konnte keine passenden Berichtsausschnitte finden, um die Frage zu beantworten. "
                "Bitte passe die Filter an oder stelle zusätzliche Informationen bereit."
            )
        lines = [
            f"Fragestellung: {question}",
            f"Patientenkontext: {patient_context or 'Keine zusätzlichen Angaben.'}",
            "Gefundene Ausschnitte:",
        ]
        lines.append(self._format_context_snippets(nodes))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _available_tools_summary(self) -> str:
        summaries: List[str] = []
        for tool in self.tools:
            desc = getattr(tool.metadata, "description", "")
            summaries.append(f"{tool.metadata.name}: {desc}")
        return "\n".join(summaries) if summaries else "Keine Werkzeuge registriert."

    def _available_tool_names(self) -> List[str]:
        return [tool.metadata.name for tool in self.tools]

    def _allowed_tools_payload(self, full: bool) -> str:
        if not self.tools:
            return "[]"
        if full:
            data = [
                {
                    "name": tool.metadata.name,
                    "description": getattr(tool.metadata, "description", ""),
                }
                for tool in self.tools
            ]
        else:
            data = self._available_tool_names()
        try:
            return json.dumps(data, ensure_ascii=False)
        except TypeError:
            return "[]"

    def _stringify_plan(self, plan: Any) -> str:
        if plan is None:
            return ""
        if isinstance(plan, str):
            return plan
        if isinstance(plan, (list, dict)):
            try:
                return json.dumps(plan, ensure_ascii=False, indent=2)
            except TypeError:
                return str(plan)
        return str(plan)

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            parts = [self._normalize_text(item) for item in value if item is not None]
            return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            try:
                return json.dumps(value, ensure_ascii=False)
            except TypeError:
                return str(value)
        return str(value)

    def _collapse_string_concatenation(self, text: str) -> str:
        pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*\+\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
        previous = None
        while previous != text:
            previous = text
            text = pattern.sub(lambda m: '"' + m.group(1) + m.group(2) + '"', text)
        text = text.replace('"""', '"')
        return text

    def _parse_json_value(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if not value:
            return None
        if isinstance(value, str):
            stripped = self._strip_json_fence(value.strip())
            if not stripped:
                return None
            stripped = self._collapse_string_concatenation(stripped)
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return None
        return None

    def _parse_json_list(self, value: Any) -> List[str]:
        parsed = self._parse_json_value(value)
        if isinstance(parsed, list):
            result: List[str] = []
            for item in parsed:
                text = self._normalize_text(item).strip()
                if text:
                    result.append(text)
            return result
        text = self._normalize_text(value).strip()
        return [text] if text else []

    def _parse_plan_object(self, plan_text: str) -> Dict[str, Any]:
        parsed = self._parse_json_value(plan_text)
        if isinstance(parsed, dict):
            return parsed
        return {}

    def _split_plan(self, plan_text: str) -> List[Dict[str, Any]]:
        if not plan_text:
            return []
        sections: List[str] = []
        current: List[str] = []
        step_pattern = re.compile(r"^\s*(?:\*{0,2}\s*)?(?:schritt|step)\s+\d+", re.IGNORECASE)
        ordered_pattern = re.compile(r"^\s*\d+[\).\s-]")
        for line in plan_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if current and (ordered_pattern.match(stripped) or step_pattern.match(stripped)):
                sections.append(" ".join(current))
                current = [stripped]
            else:
                current.append(stripped)
        if current:
            sections.append(" ".join(current))
        return [
            {"step_number": idx + 1, "objective": section, "tool_name": "", "arguments": {}}
            for idx, section in enumerate(sections)
        ]

    def _extract_plan_steps(self, plan: Any) -> List[Dict[str, Any]]:
        if plan is None:
            return []
        if isinstance(plan, str):
            plan = plan.strip()
            if not plan:
                return []
            parsed = self._try_parse_json(plan)
            if parsed is None:
                return self._split_plan(plan)
            plan = parsed

        if isinstance(plan, dict):
            steps_data = plan.get("steps")
            if isinstance(steps_data, list):
                return [self._normalize_plan_step(entry, idx + 1) for idx, entry in enumerate(steps_data)]
            return []
        if isinstance(plan, list):
            return [self._normalize_plan_step(entry, idx + 1) for idx, entry in enumerate(plan)]
        return []

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_context_metrics(self, nodes: List[Dict[str, Any]]) -> None:
        if not nodes:
            logger.debug("Context metrics: nodes=0, chars=0, approx_tokens=0")
            return

        snippets: List[str] = []
        for node in nodes:
            snippet = node.get("snippet") or node.get("text") or ""
            snippet = re.sub(r"\s+", " ", snippet).strip()
            snippets.append(snippet)

        total_chars = sum(len(snippet) for snippet in snippets)
        approx_tokens = total_chars // 4
        logger.debug(
            "Context metrics: nodes=%d, chars=%d, approx_tokens=%d",
            len(nodes),
            total_chars,
            approx_tokens,
        )

    def _normalize_plan_step(self, entry: Any, index: int) -> Dict[str, Any]:
        if isinstance(entry, dict):
            step = dict(entry)
        elif isinstance(entry, str):
            step = {"objective": entry}
        else:
            step = {"objective": str(entry)}

        step_number = step.get("step_number") or step.get("id") or index
        step["step_number"] = step_number

        tool_name = step.get("tool_name") or step.get("tool")
        if tool_name:
            step["tool_name"] = self._normalize_text(tool_name).strip()
        else:
            step.setdefault("tool_name", "")

        arguments = step.get("arguments")
        if isinstance(arguments, str):
            parsed_args = self._parse_json_value(arguments)
            if isinstance(parsed_args, dict):
                step["arguments"] = parsed_args
            else:
                step["arguments"] = {}
        elif isinstance(arguments, dict):
            step["arguments"] = arguments
        else:
            step["arguments"] = {}

        step.setdefault("objective", step.get("description", ""))
        step.setdefault("evidence_required", step.get("evidence_required", []))
        step.setdefault("stop_if", step.get("stop_if") or "")

        return step

    def _try_parse_json(self, text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            stripped = self._strip_json_fence(text)
            if stripped and stripped != text:
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    return None
        return None

    def _parse_arguments(self, raw_arguments: Any) -> Dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if raw_arguments is None:
            return {}
        if isinstance(raw_arguments, str):
            text = raw_arguments.strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                stripped = self._strip_json_fence(text)
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    logger.debug("Argument parsing failed: %s", raw_arguments)
        return {}

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = stripped[3:-3].strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        return stripped

    def _deduplicate_context(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Dict[Any, Dict[str, Any]] = {}
        for node in nodes:
            key = node.get("section_id") or (node.get("report_id"), node.get("section_name"))
            if key and key not in seen:
                seen[key] = node
        return list(seen.values())

    def _format_tool_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in filters.items():
            if value in (None, "", [], {}, ()):
                continue
            cleaned[key] = value
        return cleaned

    def _parse_json_response(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if payload is None:
            return {}
        text = payload if isinstance(payload, str) else str(payload)
        text = text.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            stripped = self._strip_json_fence(text)
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                logger.debug("Failed to parse tool JSON payload: %s", text)
        return {}

    def _summarise_nodes(self, nodes: List[Dict[str, Any]], limit: int = 3, max_chars: int = 180) -> str:
        if not nodes:
            return "Keine passenden Berichtsausschnitte gefunden."
        summaries: List[str] = []
        for node in nodes[:limit]:
            section = node.get("section_name") or node.get("report_type") or "Abschnitt"
            snippet = node.get("snippet") or node.get("text") or ""
            snippet = re.sub(r"\s+", " ", snippet).strip()
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 3].rstrip() + "..."
            summaries.append(f"{section}: {snippet}")
        return "\n".join(summaries)

    def _summarise_tool_response(self, output: ToolOutput, max_chars: int = 200) -> str:
        if getattr(output, "is_error", False):
            return str(output.content or "Werkzeug meldete einen Fehler.")
        payload = getattr(output, "raw_output", None)
        text = ""
        if isinstance(payload, dict):
            response = payload.get("response")
            if isinstance(response, str):
                text = response
            else:
                text = payload.get("content") or ""
        else:
            text = output.content or ""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "Werkzeug lieferte keine textuelle Antwort."
        if len(text) > max_chars:
            return text[: max_chars - 3].rstrip() + "..."
        return text

    def _safe_json_dumps(self, payload: Any) -> str:
        if payload in (None, "", []):
            return ""
        try:
            return json.dumps(payload, ensure_ascii=False)
        except TypeError:
            return ""

    # ------------------------------------------------------------------
    # Tool plumbing
    # ------------------------------------------------------------------
    def _get_tool(self, name: str) -> BaseTool:
        for tool in self.tools:
            if tool.metadata.name == name:
                return tool
        raise ValueError(f"Tool '{name}' ist nicht registriert.")

    def _safe_tool_call(self, tool: BaseTool, arguments: Dict[str, Any]) -> ToolOutput:
        try:
            return tool(**arguments)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Tool %s failed: %s", tool.metadata.name, exc)
            return ToolOutput(
                tool_name=tool.metadata.name,
                content=f"Tool error: {exc}",
                raw_input={"kwargs": arguments},
                raw_output=None,
                is_error=True,
            )

    def _append_user_message(self, content: str) -> None:
        self._history.append(ChatMessage(role=MessageRole.USER, content=content))

    def _append_assistant_message(self, content: str, **kwargs: Any) -> None:
        self._history.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                additional_kwargs=kwargs if kwargs else {},
            )
        )

    def _append_tool_message(self, name: str, response: Dict[str, Any], content: str) -> None:
        self._history.append(
            ChatMessage(
                role=MessageRole.TOOL,
                content=content,
                additional_kwargs={
                    "name": name,
                    "response": response,
                },
            )
        )

    # ------------------------------------------------------------------
    # LLM bridge
    # ------------------------------------------------------------------
    @abstractmethod
    def _complete(self, prompt: str) -> str:
        """Return a completion for the given prompt."""
