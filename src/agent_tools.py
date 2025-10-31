"""
Lightweight tools that the local agents can call.

These mirror the developer tooling from the legacy project but keep the
implementations simple (and dependency-light) so they are safe to use in tests
or demos. The goal is to validate agent behaviour before wiring in the full
clinical RAG pipeline.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from rank_bm25 import BM25Okapi

from .toolkit import BaseTool, ToolOutput, ToolMetadata


__all__ = [
    "ReportsRAGTool",
    "LabValuesTool",
    "load_default_tools",
]


class ReportsRAGTool(BaseTool):
    """Lightweight RAG tool backed by the anonymised SQLite database."""

    DB_PATH = Path("src/database/anonymized.sqlite")
    DEFAULT_PATIENT_ID = "0001005847"
    DEFAULT_TOP_K = 5
    REPORT_TYPES = [
        "Arztbrief",
        "Beschluss",
        "Cytology",
        "Flow",
        "Path",
        "RAD",
    ]

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        patient_id: str = DEFAULT_PATIENT_ID,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.db_path = Path(db_path) if db_path else self.DB_PATH
        self.patient_id = patient_id
        self.top_k = top_k
        self._metadata = ToolMetadata(
            name="retrieve_reports",
            description=(
                "Retrieve clinical report sections for a patient using a BM25 keyword search "
                "over the anonymised SQLite database. Supports optional filtering by report_type "
                "and report_date. Valid report_type values: "
                + ", ".join(self.REPORT_TYPES)
                + ". Multiple report types can be provided as a list."
            ),
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(  # type: ignore[override]
        self,
        query: str | None = None,
        report_type: Union[str, Sequence[str], None] = None,
        report_date: str | None = None,
        patient_id: str | None = None,
        input: Any | None = None,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> ToolOutput:
        payload = self._coerce_payload(
            query=query,
            report_type=report_type,
            report_date=report_date,
            patient_id=patient_id,
            input_value=input,
            extra_kwargs=kwargs,
        )

        sections = self._fetch_sections(
            patient_id=payload["patient_id"],
            report_type=payload.get("report_type"),
            report_date=payload.get("report_date"),
        )

        matches = self._rank_sections(
            sections=sections,
            query=payload["query"],
            top_k=top_k or payload.get("top_k") or self.top_k,
        )

        result = {
            "response": self._format_response(matches, payload),
            "context_nodes": matches,
        }

        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(result, ensure_ascii=False),
            raw_input={"kwargs": payload},
            raw_output=result,
        )

    def _coerce_payload(
        self,
        *,
        query: str | None,
        report_type: Union[str, Sequence[str], None],
        report_date: str | None,
        patient_id: str | None,
        input_value: Any,
        extra_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "patient_id": patient_id or self.patient_id,
            "report_type": report_type,
            "report_date": report_date,
        }

        for candidate in (input_value, extra_kwargs.get("input")):
            if not candidate:
                continue
            parsed = self._parse_json(candidate)
            if isinstance(parsed, dict):
                payload.update({k: v for k, v in parsed.items() if v not in (None, "")})

        payload.update(
            {
                k: v
                for k, v in extra_kwargs.items()
                if k in {"query", "report_type", "report_date", "patient_id", "top_k"}
            }
        )

        if query and "query" not in payload:
            payload["query"] = query

        if "query" not in payload or not str(payload["query"]).strip():
            raise ValueError("missing a required argument: 'query'")

        payload["query"] = str(payload["query"]).strip()
        payload["patient_id"] = str(payload.get("patient_id") or self.patient_id)

        if payload.get("report_type") not in (None, ""):
            normalized_types = self._normalize_report_types(payload["report_type"])
            if not normalized_types:
                payload["report_type"] = None
            elif len(normalized_types) == 1:
                payload["report_type"] = normalized_types[0]
            else:
                payload["report_type"] = normalized_types
        else:
            payload["report_type"] = None

        if payload.get("report_date") in (None, ""):
            payload["report_date"] = None
        else:
            payload["report_date"] = str(payload["report_date"]).strip()

        if payload.get("top_k") in (None, ""):
            payload.pop("top_k", None)

        return payload

    def _normalize_report_types(self, value: Union[str, Sequence[str]]) -> List[str]:
        collected: List[str] = []

        def _handle(item: Any) -> None:
            if item is None:
                return

            if isinstance(item, str):
                text = item.strip()
                if not text:
                    return

                lowered = text.lower()
                if lowered in {"all", "any", "*"}:
                    return

                parsed = self._try_parse_literal(text)
                if parsed is not None and not (isinstance(parsed, str) and parsed.strip() == text):
                    _handle(parsed)
                    return

                if "," in text:
                    parts = [part.strip() for part in text.split(",")]
                    for part in parts:
                        if part:
                            _handle(part)
                    return

                if text not in self.REPORT_TYPES:
                    raise ValueError(
                        f"report_type '{text}' is not supported. Known types: {', '.join(self.REPORT_TYPES)}"
                    )
                if text not in collected:
                    collected.append(text)
                return

            if isinstance(item, (list, tuple, set)):
                for sub in item:
                    _handle(sub)
                return

            parsed = self._try_parse_literal(str(item))
            if parsed is not None and parsed is not item:
                _handle(parsed)
                return

            raise ValueError(
                f"Unsupported report_type value: {item!r}. Expected a string or sequence of strings."
            )

        _handle(value)
        return collected

    @staticmethod
    def _try_parse_literal(text: str) -> Any:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None

    def _fetch_sections(
        self,
        *,
        patient_id: str,
        report_type: Union[str, Sequence[str], None],
        report_date: Optional[str],
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        conditions = ["rs.patient_id = ?"]
        params: List[Any] = [patient_id]

        if report_type:
            if isinstance(report_type, (list, tuple, set)):
                valid_types = [rtype for rtype in report_type if rtype]
                if valid_types:
                    placeholders = ", ".join(["?"] * len(valid_types))
                    conditions.append(f"rs.report_type IN ({placeholders})")
                    params.extend(valid_types)
            else:
                conditions.append("rs.report_type = ?")
                params.append(report_type)

        if report_date:
            conditions.append("rs.report_date LIKE ?")
            params.append(f"{report_date}%")

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT
                rs.section_id,
                rs.report_id,
                rs.section_name,
                rs.section_content,
                rs.report_type,
                rs.report_date,
                rs.patient_id
            FROM report_sections rs
            WHERE {where_clause}
            ORDER BY rs.report_date DESC, rs.section_order ASC
            LIMIT ?
        """
        params.append(limit)

        cursor.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def _rank_sections(
        self,
        *,
        sections: List[Dict[str, Any]],
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not sections:
            return []

        documents = [section.get("section_content") or "" for section in sections]
        tokenized_docs = [self._tokenize(text) for text in documents]
        if not any(tokenized_docs):
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(query_tokens)
        scored_sections = list(zip(sections, scores))
        scored_sections.sort(key=lambda item: item[1], reverse=True)

        k = max(1, int(top_k) if top_k else self.top_k)
        matches = []
        for section, score in scored_sections[:k]:
            text = section.get("section_content") or ""
            snippet = self._build_snippet(text, query)
            matches.append(
                {
                    "section_id": section.get("section_id"),
                    "report_id": section.get("report_id"),
                    "report_type": section.get("report_type"),
                    "report_date": section.get("report_date"),
                    "patient_id": section.get("patient_id"),
                    "section_name": section.get("section_name"),
                    "text": text,
                    "snippet": snippet,
                    "score": float(score),
                }
            )
        return matches

    def _format_response(self, matches: List[Dict[str, Any]], payload: Dict[str, Any]) -> str:
        if not matches:
            return (
                f"No report sections found for patient {payload['patient_id']} "
                "with the provided filters."
            )

        top = matches[0]
        report_info = []
        if top.get("report_type"):
            report_info.append(top["report_type"])
        if top.get("report_date"):
            report_info.append(str(top["report_date"]))

        details = " - ".join(report_info) if report_info else ""
        snippet = top.get("snippet") or (top.get("text", "")[:240] + "...")
        response_lines = [
            f"Top section for patient {payload['patient_id']}: {details}".strip(),
            snippet,
        ]
        return "\n".join(line for line in response_lines if line)

    @staticmethod
    def _parse_json(value: Any) -> Any | None:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _build_snippet(text: str, query: str, length: int = 512) -> str:
        if not text:
            return ""
        lowered = text.lower()
        tokens = ReportsRAGTool._tokenize(query)
        if not tokens:
            return text[:length] + ("..." if len(text) > length else "")

        first = tokens[0]
        idx = lowered.find(first)
        if idx == -1:
            return text[:length] + ("..." if len(text) > length else "")

        start = max(0, idx - length // 4)
        end = min(len(text), start + length)
        snippet = text[start:end]
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{snippet}{suffix}"


class LabValuesTool(BaseTool):
    """Retrieve structured laboratory measurements from the `reports` table."""

    DB_PATH = ReportsRAGTool.DB_PATH
    DEFAULT_PATIENT_ID = ReportsRAGTool.DEFAULT_PATIENT_ID
    PANEL_ALIASES: Dict[str, List[str]] = {
        "comprehensivemetabolicpanel": [
            "Natrium",
            "Kalium",
            "Kreatinin",
            "Harnstoff-N (BUN",
            "Bilirubin, gesam",
            "Gamma-GT",
            "GOT (ASAT)",
            "GPT (ALAT)",
            "LDH",
            "Glucose (Serum)",
        ],
        "cmp": [
            "Natrium",
            "Kalium",
            "Kreatinin",
            "Harnstoff-N (BUN",
            "Bilirubin, gesam",
            "Gamma-GT",
            "GOT (ASAT)",
            "GPT (ALAT)",
            "LDH",
            "Glucose (Serum)",
        ],
        "electrolytes": ["Natrium", "Kalium", "Cl- (BG)", "Ca++ (ionisiert,"],
        "cbccbc": [
            "Leukozyten",
            "Erythrozyten",
            "Hämoglobin",
            "Hämatokrit",
            "Thrombozyten",
            "MCV",
            "MCH",
            "MCHC",
        ],
    }
    DEFAULT_SYNONYMS: Dict[str, str] = {
        "hemoglobin": "Hämoglobin",
        "hgb": "Hämoglobin",
        "hb": "Hämoglobin",
        "leukocytes": "Leukozyten",
        "wbc": "Leukozyten",
        "anc": "Neutrophile Granulozyten absolut",
        "neutrophils": "Neutrophile Granulozyten",
        "platelets": "Thrombozyten",
        "plt": "Thrombozyten",
        "creatinine": "Kreatinin",
        "gfr": "GFR",
        "egfr": "GFR",
        "alt": "GPT (ALAT)",
        "sgpt": "GPT (ALAT)",
        "ast": "GOT (ASAT)",
        "sgot": "GOT (ASAT)",
        "bilirubin": "Bilirubin, gesamt",
        "totalbilirubin": "Bilirubin, gesamt",
        "ldh": "LDH",
        "lactatedehydrogenase": "LDH",
        "gfr": "GFR (CKD-EPI)",
        "gfrckdepi": "GFR (CKD-EPI)",
        "gfrmdrd": "GFR (MDRD)",
    }

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        patient_id: str = DEFAULT_PATIENT_ID,
        catalog_path: str | Path | None = None,
        synonyms_path: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else self.DB_PATH
        self.patient_id = patient_id
        self._catalog_path = Path(catalog_path) if catalog_path else None
        self._synonyms_path = Path(synonyms_path) if synonyms_path else None

        self._metadata = ToolMetadata(
            name="retrieve_lab_values",
            description=(
                "Get structured laboratory measurements for a patient. "
                "Call without `test_name` to list all available tests. "
                "Otherwise provide the desired `test_name` (or list of names) and optional `date` (DD.MM.YYYY)."
            ),
        )
        self._lab_cache: Dict[str, Dict[str, Any]] = {}
        self._available_tests: List[str] = []
        self._synonym_map: Dict[str, str] = {}
        self._initialize_catalog()

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(  # type: ignore[override]
        self,
        test_name: Union[str, Sequence[str], None] = None,
        date: str | None = None,
        patient_id: str | None = None,
        latest: bool | None = None,
        input: Any | None = None,
        **kwargs: Any,
    ) -> ToolOutput:
        payload = self._coerce_payload(
            test_name=test_name,
            date=date,
            patient_id=patient_id,
            latest=latest,
            input_value=input,
            extra_kwargs=kwargs,
        )

        lab_data = self._get_lab_data(payload["patient_id"])
        available_tests = self._available_tests if payload["patient_id"] == self.patient_id else sorted(lab_data.keys())
        category_hint = payload.pop("_category_hint", None)
        if payload["tests"] is None and payload.get("query"):
            inferred = self._infer_tests_from_query(payload["query"], available_tests)
            if inferred:
                payload["tests"] = inferred

        if payload["tests"] is None:
            if payload.get("query"):
                message = (
                    "No laboratory tests matched the requested query. "
                    "Please specify the exact test name from the catalogue or adjust the query."
                )
                result = {
                    "response": message,
                    "available_tests": available_tests,
                    "context_nodes": [],
                }
                return ToolOutput(
                    tool_name=self._metadata.name,
                    content=json.dumps(result, ensure_ascii=False),
                    raw_input={"kwargs": payload},
                    raw_output=result,
                    is_error=True,
                )
            response = self._format_available_tests(payload["patient_id"], available_tests, category_hint)
            result = {
                "response": response,
                "available_tests": available_tests,
                "context_nodes": [],
            }
            return ToolOutput(
                tool_name=self._metadata.name,
                content=json.dumps(result, ensure_ascii=False),
                raw_input={"kwargs": payload},
                raw_output=result,
                is_error=True,
            )

        selected_tests, missing_tests = self._select_tests(payload["tests"], available_tests, category_hint)

        if not selected_tests:
            preview = ", ".join(available_tests[:50])
            suffix = "..." if len(available_tests) > 50 else ""
            message = (
                f"No matching laboratory tests for {missing_tests or payload['tests']}.")
            message += f" Available tests include: {preview}{suffix}"
            result = {
                "response": message,
                "available_tests": available_tests,
                "missing_tests": missing_tests,
                "context_nodes": [],
            }
            return ToolOutput(
                tool_name=self._metadata.name,
                content=json.dumps(result, ensure_ascii=False),
                raw_input={"kwargs": payload},
                raw_output=result,
                is_error=True,
            )

        measurements_bundle: List[Dict[str, Any]] = []
        context_nodes: List[Dict[str, Any]] = []
        for test in selected_tests:
            info = lab_data.get(test, {})
            entries = self._collect_measurements(
                test_name=test,
                info=info,
                date_filter=payload["date"],
                latest_only=payload["latest"],
            )
            record = {
                "test": test,
                "unit": info.get("unit"),
                "reference_range": info.get("range"),
                "measurements": entries,
            }
            measurements_bundle.append(record)
            for entry in entries:
                assessment_id = entry.get("assessment_id") or f"{test}:{entry.get('date')}:{entry.get('value')}"
                snippet = f"{entry.get('date') or ''} {entry.get('time') or ''}: {entry.get('value')} {info.get('unit') or ''}".strip()
                context_nodes.append(
                    {
                        "section_id": f"lab::{test}::{assessment_id}",
                        "report_type": "labor",
                        "section_name": test,
                        "snippet": snippet,
                        "test": test,
                        "value": entry.get("value"),
                        "unit": info.get("unit"),
                        "reference_range": info.get("range"),
                        "date": entry.get("date"),
                        "time": entry.get("time"),
                        "assessment_id": entry.get("assessment_id"),
                    }
                )

        response = self._format_measurement_response(
            payload["patient_id"],
            measurements_bundle,
            payload["date"],
            payload["latest"],
        )

        result = {
            "response": response,
            "available_tests": available_tests,
            "values": measurements_bundle,
            "context_nodes": context_nodes[:200],
            "missing_tests": missing_tests,
        }

        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(result, ensure_ascii=False),
            raw_input={"kwargs": payload},
            raw_output=result,
        )

    def _coerce_payload(
        self,
        *,
        test_name: Union[str, Sequence[str], None],
        date: str | None,
        patient_id: str | None,
        latest: bool | None,
        input_value: Any,
        extra_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "patient_id": patient_id or self.patient_id,
            "tests": test_name,
            "date": date,
            "latest": latest,
            "query": None,
            "_category_hint": None,
        }

        for candidate in (input_value, extra_kwargs.get("input")):
            if not candidate:
                continue
            parsed = ReportsRAGTool._parse_json(candidate)
            if isinstance(parsed, dict):
                payload.update({k: v for k, v in parsed.items() if v not in (None, "")})

        payload.update(
            {
                k: v
                for k, v in extra_kwargs.items()
                if k
                in {
                    "patient_id",
                    "test_name",
                    "test_names",
                    "tests",
                    "name",
                    "names",
                    "lab_test_name",
                    "lab_test_names",
                    "lab_names",
                    "lab_name",
                    "labs",
                    "date",
                    "latest",
                    "lab_category",
                    "category",
                    "panel",
                    "query",
                }
            }
        )

        if test_name is not None and payload.get("tests") in (None, ""):
            payload["tests"] = test_name

        payload["patient_id"] = str(payload.get("patient_id") or self.patient_id)

        for key in ("lab_category", "category", "panel"):
            if payload.get(key) not in (None, ""):
                payload["_category_hint"] = str(payload[key]).strip()
            payload.pop(key, None)

        tests_value = None
        for key in (
            "tests",
            "test_names",
            "test_name",
            "lab_names",
            "lab_name",
            "labs",
            "name",
            "names",
            "lab_test_name",
            "lab_test_names",
        ):
            if payload.get(key) not in (None, "", []):
                tests_value = payload[key]
            payload.pop(key, None)
            if tests_value is not None:
                break
        payload["tests"] = tests_value

        if payload["tests"] in (None, []) or payload["tests"] == "":
            payload["tests"] = None

        date_value = payload.get("date")
        if date_value in (None, ""):
            payload["date"] = None
        else:
            payload["date"] = str(date_value).strip()

        latest_value = payload.get("latest")
        if latest_value in (None, ""):
            payload["latest"] = False
        else:
            payload["latest"] = self._to_bool(latest_value)

        query_value = payload.get("query")
        if isinstance(query_value, str):
            query_value = query_value.strip()
            payload["query"] = query_value or None
        else:
            payload["query"] = None

        return payload

    def _load_lab_data(self, patient_id: str) -> Dict[str, Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT content
            FROM reports
            WHERE patient_id = ? AND report_type = 'labor'
            ORDER BY COALESCE(report_date, '') DESC
            LIMIT 1
            """,
            (patient_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row is None or row[0] in (None, ""):
            return {}

        try:
            data = json.loads(row[0])
        except json.JSONDecodeError:
            return {}

        lab_map: Dict[str, Dict[str, Any]] = {}
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                test_name = entry.get("test")
                if not test_name:
                    continue
                lab_map[test_name] = {
                    "range": entry.get("range"),
                    "unit": entry.get("unit"),
                    "assessments": entry.get("assessments") or {},
                }
        return lab_map

    def _get_lab_data(self, patient_id: str) -> Dict[str, Dict[str, Any]]:
        if patient_id in self._lab_cache:
            return self._lab_cache[patient_id]
        lab_map = self._load_lab_data(patient_id)
        self._lab_cache[patient_id] = lab_map
        if patient_id == self.patient_id:
            self._available_tests = sorted(lab_map.keys())
            self._write_catalog()
        return lab_map

    def _initialize_catalog(self) -> None:
        lab_map = self._load_lab_data(self.patient_id)
        self._lab_cache[self.patient_id] = lab_map
        self._available_tests = sorted(lab_map.keys())
        self._write_catalog()
        self._load_synonyms()

    def _write_catalog(self) -> None:
        if not self._catalog_path:
            return
        try:
            self._catalog_path.parent.mkdir(parents=True, exist_ok=True)
            self._catalog_path.write_text(
                json.dumps(self._available_tests, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger = logging.getLogger(__name__)
            logger.warning("Failed to write lab catalog to %s", self._catalog_path, exc_info=True)

    def _load_synonyms(self) -> None:
        synonym_map: Dict[str, str] = {}
        for canonical in self._available_tests:
            norm = self._normalize_label(canonical)
            if norm:
                synonym_map.setdefault(norm, canonical)
        if self._synonyms_path and self._synonyms_path.exists():
            try:
                data = json.loads(self._synonyms_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for canonical, aliases in data.items():
                        canonical_norm = self._normalize_label(canonical)
                        resolved = None
                        if canonical_norm in synonym_map:
                            resolved = synonym_map[canonical_norm]
                        else:
                            match = self._fuzzy_match(canonical_norm, synonym_map)
                            if match:
                                resolved = match
                        if not resolved:
                            continue
                        for alias in aliases if isinstance(aliases, (list, tuple)) else [aliases]:
                            norm_alias = self._normalize_label(str(alias))
                            if norm_alias:
                                synonym_map[norm_alias] = resolved
                elif isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        canonical = item.get("canonical")
                        aliases = item.get("aliases", [])
                        if not canonical:
                            continue
                        canonical_norm = self._normalize_label(canonical)
                        resolved = synonym_map.get(canonical_norm)
                        if not resolved:
                            resolved = self._fuzzy_match(canonical_norm, synonym_map)
                        if not resolved:
                            continue
                        for alias in aliases:
                            norm_alias = self._normalize_label(str(alias))
                            if norm_alias:
                                synonym_map[norm_alias] = resolved
            except Exception:
                logger = logging.getLogger(__name__)
                logger.warning("Failed to load lab synonyms from %s", self._synonyms_path, exc_info=True)
        self._synonym_map = synonym_map

    def _build_synonym_lookup(self, available_norm_map: Dict[str, str]) -> Dict[str, str]:
        lookup: Dict[str, str] = dict(available_norm_map)

        for alias, canonical in self.DEFAULT_SYNONYMS.items():
            norm_alias = self._normalize_label(alias)
            canonical_norm = self._normalize_label(canonical)
            resolved = available_norm_map.get(canonical_norm)
            if not resolved and available_norm_map:
                resolved = self._fuzzy_match(canonical_norm, available_norm_map)
            if norm_alias and resolved:
                lookup[norm_alias] = resolved

        for alias_norm, canonical in self._synonym_map.items():
            canonical_norm = self._normalize_label(canonical)
            resolved = available_norm_map.get(canonical_norm)
            if not resolved and available_norm_map:
                resolved = self._fuzzy_match(canonical_norm, available_norm_map)
            if alias_norm and resolved:
                lookup[alias_norm] = resolved

        return lookup

    def _resolve_label_from_lookup(
        self,
        label: str,
        synonym_lookup: Dict[str, str],
        available_norm_map: Dict[str, str],
    ) -> Optional[str]:
        norm_label = self._normalize_label(label)
        if not norm_label:
            return None
        if norm_label in synonym_lookup:
            return synonym_lookup[norm_label]
        return self._fuzzy_match(norm_label, available_norm_map)

    def _infer_tests_from_query(self, query: str, available: Sequence[str]) -> List[str]:
        if not query or not available:
            return []

        available_norm_map = {self._normalize_label(name): name for name in available if name}
        synonym_lookup = self._build_synonym_lookup(available_norm_map)
        norm_query = self._normalize_label(query)

        inferred: List[str] = []

        def register(candidate: Optional[str]) -> None:
            if candidate and candidate not in inferred:
                inferred.append(candidate)

        for alias, tests in self.PANEL_ALIASES.items():
            if alias in norm_query:
                for test in tests:
                    register(self._resolve_label_from_lookup(test, synonym_lookup, available_norm_map))

        raw_segments = re.split(r"[,;/\n]+", query)
        if len(raw_segments) == 1:
            raw_segments = re.split(r"\\band\\b|\\bor\\b", raw_segments[0])

        for segment in raw_segments:
            norm_segment = self._normalize_label(segment)
            if not norm_segment:
                continue
            candidate = synonym_lookup.get(norm_segment)
            if not candidate:
                candidate = self._fuzzy_match(norm_segment, synonym_lookup)
            if not candidate and len(norm_segment) > 3:
                candidate = synonym_lookup.get(norm_segment.rstrip("s"))
            if not candidate:
                candidate = self._fuzzy_match(norm_segment, available_norm_map)
            register(candidate)

        for token in set(re.findall(r"[a-z0-9]+", norm_query)):
            candidate = synonym_lookup.get(token)
            if not candidate and len(token) > 3:
                candidate = self._fuzzy_match(token, synonym_lookup)
            if not candidate:
                candidate = self._fuzzy_match(token, available_norm_map)
            register(candidate)

        if "gfr" in norm_query:
            for preferred in ("GFR (CKD-EPI)", "GFR (MDRD)"):
                register(self._resolve_label_from_lookup(preferred, synonym_lookup, available_norm_map))

        return inferred

    def _select_tests(
        self,
        raw: Union[str, Sequence[str]],
        available: Sequence[str],
        category_hint: Optional[str],
    ) -> Tuple[List[str], List[str]]:
        if isinstance(raw, (list, tuple, set)):
            inputs = list(raw)
        elif isinstance(raw, str):
            literal = self._try_parse_literal(raw)
            if isinstance(literal, (list, tuple, set)):
                inputs = list(literal)
            else:
                inputs = [item.strip() for item in raw.split(",") if item.strip()]
        else:
            inputs = [str(raw)]

        if not inputs:
            return [], []

        available_norm_map = {self._normalize_label(name): name for name in available if name}
        synonym_lookup = self._build_synonym_lookup(available_norm_map)
        resolved: List[str] = []
        missing: List[str] = []

        if category_hint:
            panel_key = self._normalize_label(category_hint)
            for alias, tests in self.PANEL_ALIASES.items():
                if alias == panel_key:
                    for test in tests:
                        match = self._resolve_label_from_lookup(test, synonym_lookup, available_norm_map)
                        if match:
                            resolved.append(match)

        for item in inputs:
            label = str(item).strip()
            if not label:
                continue
            match = self._resolve_label_from_lookup(label, synonym_lookup, available_norm_map)
            if match:
                resolved.append(match)
            else:
                missing.append(label)

        unique_resolved = list(dict.fromkeys(resolved))
        return unique_resolved, missing

    def _collect_measurements(
        self,
        *,
        test_name: str,
        info: Dict[str, Any],
        date_filter: str | None,
        latest_only: bool,
    ) -> List[Dict[str, Any]]:
        assessments = info.get("assessments") or {}
        records: List[Dict[str, Any]] = []

        for assessment_id, measurement in assessments.items():
            if not isinstance(measurement, dict):
                continue
            date_str = measurement.get("date")
            time_str = measurement.get("time")
            normalized_date = self._normalize_date(date_str)
            sort_key = self._compose_sort_key(date_str, time_str)
            record = {
                "assessment_id": assessment_id,
                "date": date_str,
                "time": time_str,
                "value": measurement.get("value"),
                "normalized_date": normalized_date,
                "_sort_key": sort_key,
            }
            records.append(record)

        if date_filter:
            target_date = self._normalize_date(date_filter)
            if target_date:
                records = [rec for rec in records if rec.get("normalized_date") == target_date]
            else:
                records = [rec for rec in records if (rec.get("date") or "").strip() == date_filter.strip()]

        records.sort(key=lambda rec: rec.get("_sort_key"), reverse=True)

        if latest_only and records:
            records = [records[0]]

        for rec in records:
            rec.pop("_sort_key", None)
            rec.pop("normalized_date", None)

        return records

    def _format_available_tests(self, patient_id: str, tests: Sequence[str], category_hint: str | None) -> str:
        if not tests:
            return f"No laboratory data found for patient {patient_id}."
        header = f"Laboratory tests available for patient {patient_id}"
        if category_hint:
            header += f" (no category '{category_hint}' recognised)"
        preview = ", ".join(tests[:50])
        suffix = "..." if len(tests) > 50 else ""
        return f"{header}: {preview}{suffix}"

    def _format_measurement_response(
        self,
        patient_id: str,
        bundles: Sequence[Dict[str, Any]],
        date_filter: str | None,
        latest_only: bool,
    ) -> str:
        if not bundles or all(not bundle.get("measurements") for bundle in bundles):
            postfix = f" on {date_filter}" if date_filter else ""
            return f"No laboratory measurements found for patient {patient_id}{postfix}."

        lines: List[str] = []
        if date_filter:
            lines.append(f"Filtered by date {date_filter}.")
        if latest_only:
            lines.append("Only the most recent measurement per test is shown.")

        for bundle in bundles:
            test = bundle.get("test")
            unit = bundle.get("unit")
            ref = bundle.get("reference_range")
            measurements = bundle.get("measurements") or []
            if not measurements:
                lines.append(f"{test}: keine Messwerte gefunden.")
                continue

            fragments = []
            for entry in measurements[:3]:
                value = entry.get("value")
                date = entry.get("date")
                time = entry.get("time")
                fragment = f"{date} {time or ''}: {value} {unit or ''}".strip()
                fragments.append(fragment)
            summary = "; ".join(fragments)
            suffix = f" (Ref: {ref})" if ref else ""
            if len(measurements) > 3:
                summary += f" (+{len(measurements) - 3} weitere)"
            lines.append(f"{test}: {summary}{suffix}")

        return "\n".join(lines)

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "ja"}:
            return True
        if text in {"0", "false", "no", "n", "nein"}:
            return False
        raise ValueError(f"Cannot interpret value '{value}' as boolean.")

    @staticmethod
    def _try_parse_literal(text: str) -> Any:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None

    def _normalize_label(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        normalized = "".join(ch for ch in normalized if ch.isalnum())
        return normalized.lower()

    def _fuzzy_match(self, label: str, candidates: Dict[str, str]) -> Optional[str]:
        if not candidates:
            return None
        if label in candidates:
            return candidates[label]
        best_score = 0
        best_match = None
        for cand_key, original in candidates.items():
            if not cand_key:
                continue
            common_prefix = self._common_prefix(label, cand_key)
            score = common_prefix / max(len(label), len(cand_key))
            if score > best_score and score >= 0.6:
                best_score = score
                best_match = original
        return best_match

    @staticmethod
    def _common_prefix(a: str, b: str) -> int:
        count = 0
        for ch_a, ch_b in zip(a, b):
            if ch_a != ch_b:
                break
            count += 1
        return count

    @staticmethod
    def _normalize_date(value: str | None) -> str | None:
        if not value:
            return None
        value = value.strip()
        for pattern in ("%d.%m.%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                dt = datetime.strptime(value, pattern)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    @staticmethod
    def _compose_sort_key(date_str: str | None, time_str: str | None) -> str:
        normalized_date = LabValuesTool._normalize_date(date_str)
        if normalized_date:
            time_component = (time_str or "00:00").strip() or "00:00"
            for pattern in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(f"{normalized_date} {time_component}", pattern)
                    return dt.strftime("%Y-%m-%dT%H:%M")
                except ValueError:
                    continue
            try:
                dt = datetime.strptime(normalized_date, "%Y-%m-%d")
                return dt.strftime("%Y-%m-%dT00:00")
            except ValueError:
                pass
        return f"{(date_str or '').strip()}T{(time_str or '').strip()}"

def load_default_tools() -> List[BaseTool]:
    """Return the default tool bundle for the agents."""
    return [ReportsRAGTool(), LabValuesTool()]
