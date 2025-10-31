"""DSPy signature definitions for the lightweight agents."""

from __future__ import annotations

import dspy


class PlanPatientQuestion(dspy.Signature):
    """Führe eine medizinische Einschätzung durch und identifiziere die Informationslücken."""

    question = dspy.InputField(desc="Die vom Nutzer gestellte klinische Frage.")
    patient_context = dspy.InputField(desc="Bekannter Patientenkontext (frei gelassen, wenn nicht vorhanden).")
    default_filters = dspy.InputField(desc="Standardfilter (z. B. report_type, top_k), die berücksichtigt werden sollen.")
    allowed_tools = dspy.InputField(desc="Liste der verfügbaren Werkzeuge samt kurzer Beschreibung.")

    clinical_analysis = dspy.OutputField(
        desc=(
            "Kurze medizinische Einschätzung (max. 3 Sätze) mit relevanter Leitlinienkenntnis. "
            "Keine Wiederholung der Frage, nur neue Einsichten."
        )
    )
    required_information = dspy.OutputField(
        desc=(
            "JSON-Liste der klinisch notwendigen Informationspunkte (z. B. Diagnosen, Scores, Laborgruppen). "
            "Format: [\"...\", ...]."
        )
    )
    missing_information = dspy.OutputField(
        desc=(
            "JSON-Liste der Informationen, die aktuell fehlen und ggf. beim Nutzer nachgefragt werden müssen." 
            "Format: [\"...\", ...]."
        )
    )
    recommended_tools = dspy.OutputField(
        desc=(
            "JSON-Liste der Werkzeugnamen aus allowed_tools, sortiert nach Priorität, die benötigt werden, "
            "um die Informationslücken zu schließen. Format: [\"retrieve_reports\", ...]."
        )
    )


class StructuredToolPlan(dspy.Signature):
    """Leite aus der Analyse einen deterministischen Werkzeugplan ab."""

    question = dspy.InputField(desc="Die klinische Fragestellung.")
    patient_context = dspy.InputField(desc="Bekannter Patientenkontext.")
    default_filters = dspy.InputField(desc="Standardfilter (report_type, top_k, etc.).")
    allowed_tools = dspy.InputField(desc="Liste der verfügbaren Werkzeuge (Namen).")
    clinical_analysis = dspy.InputField(desc="Ergebnis aus PlanPatientQuestion.clinical_analysis.")
    required_information = dspy.InputField(desc="JSON-Liste der benötigten Informationen.")
    missing_information = dspy.InputField(desc="JSON-Liste der fehlenden Informationen.")
    recommended_tools = dspy.InputField(desc="JSON-Liste der vorgesehenen Werkzeuge.")

    tool_plan = dspy.OutputField(
        desc=(
            "JSON-Objekt mit deterministischem Plan. Struktur: {\"steps\": ["
            "{\"step_number\": 1, \"objective\": \"...\", \"tool_name\": \"...\", "
            "\"arguments\": { ... }, \"evidence_required\": [\"...\"], "
            "\"stop_if\": \"Bedingung\"}], \"global_stop_conditions\": [\"...\"]}. "
            "Keine erneute klinische Analyse oder Zusammenfassung ausgeben."
        )
    )


class ToolExecutionStep(dspy.Signature):
    """Wähle den nächsten Toolschritt oder beende frühzeitig."""

    question = dspy.InputField(desc="Originale Nutzerfrage.")
    patient_context = dspy.InputField(desc="Patientenkontext, falls verfügbar.")
    plan_chunk = dspy.InputField(desc="JSON-Objekt des nächsten Planschritts.")
    previous_results = dspy.InputField(desc="Zusammenfassung der bisherigen Werkzeugausgaben und Befunde.")
    allowed_tools = dspy.InputField(desc="Liste der zulässigen Werkzeugnamen.")
    global_stop_conditions = dspy.InputField(desc="JSON-Liste von Bedingungen, bei deren Erfüllung beendet werden darf.")

    action = dspy.OutputField(desc="'call_tool', 'skip' oder 'finish'.")
    tool_name = dspy.OutputField(desc="Name des aufzurufenden Werkzeugs bei 'call_tool'.")
    arguments = dspy.OutputField(desc="JSON-String mit den Argumenten für den Werkzeugaufruf.")
    rationale = dspy.OutputField(
        desc=(
            "Begründung, warum der Schritt ausgeführt, übersprungen oder beendet wird. "
            "Muss explizit prüfen, ob bereits ausreichend Evidenz für eine Antwort vorliegt."
        )
    )



class GermanClinicalSummary(dspy.Signature):
    """Verfasse eine patientenbezogene Zusammenfassung mit klaren Empfehlungen."""

    question = dspy.InputField(desc="Die zu beantwortende Frage.")
    patient_context = dspy.InputField(desc="Patientenkontext (falls vorhanden).")
    plan_overview = dspy.InputField(desc="Kurzfassung der ausgeführten Schritte und Analyse.")
    applied_filters = dspy.InputField(desc="Beschreibung der eingesetzten Filter und Tools.")
    context_snippets = dspy.InputField(desc="Nummerierte Ausschnitte aus den Quellen, inklusive Zitations-IDs.")
    outstanding_information = dspy.InputField(desc="JSON-Liste der weiterhin fehlenden Informationen.")

    clinical_brief = dspy.OutputField(
        desc=(
            "Knappes Fazit (1-3 Sätze) für ärztliche Kolleg*innen mit eindeutiger Empfehlung oder nächstem Schritt. "
            "Muss patientenbezogen sein und mindestens eine Quelle zitieren [n]."
        )
    )
    detailed_answer = dspy.OutputField(
        desc=(
            "Ausführliche Begründung mit Unterüberschriften (Situation, Bewertung, Empfehlungen, fehlende Daten). "
            "Alle Aussagen mit [n] belegen; fehlende Informationen klar nennen."
        )
    )
