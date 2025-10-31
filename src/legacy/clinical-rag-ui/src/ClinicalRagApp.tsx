import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Loader2, Send, Trash2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"

const API_BASE = import.meta.env.VITE_AGENT_API ?? "/api"
const PATIENTS_ENDPOINT = `${API_BASE}/patients`
const CHAT_ENDPOINT = `${API_BASE}/chat`

interface PatientSummary {
  id: string
  name: string
  description?: string
  details?: string
}

interface AgentContextNode {
  report_type?: string
  report_date?: string
  section_name?: string
  snippet?: string
}

interface AgentMetadata {
  clinical_brief?: string
  detailed_answer?: string
  context_nodes?: AgentContextNode[]
  missing_information?: string[]
  analysis?: string
  required_information?: string[]
  actions?: Array<Record<string, unknown>>
}

type MessageRole = "user" | "assistant" | "system"

interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  createdAt: string
  metadata?: AgentMetadata
}

type AgentEventPayload = Record<string, unknown>

interface AgentEvent {
  id: string
  type: string
  timestamp: number
  payload: AgentEventPayload
}

function createId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return Math.random().toString(36).slice(2)
}

function normaliseEvent(envelope: Record<string, unknown> | null | undefined): AgentEvent | null {
  if (!envelope || typeof envelope !== "object") {
    return null
  }
  const idRaw = envelope.id
  const id = typeof idRaw === "string" && idRaw ? idRaw : createId()
  const typeRaw = envelope.type
  const type = typeof typeRaw === "string" && typeRaw ? typeRaw : "unknown"
  const timestampRaw = envelope.timestamp
  let timestamp: number
  if (typeof timestampRaw === "number" && Number.isFinite(timestampRaw)) {
    timestamp = timestampRaw
  } else if (typeof timestampRaw === "string") {
    const parsed = Number(timestampRaw)
    timestamp = Number.isFinite(parsed) ? parsed : Date.now() / 1000
  } else {
    timestamp = Date.now() / 1000
  }
  const payloadRaw = envelope.payload
  const payload: AgentEventPayload =
    payloadRaw && typeof payloadRaw === "object" ? (payloadRaw as AgentEventPayload) : {}
  return { id, type, timestamp, payload }
}

export default function ClinicalRagApp() {
  const [patients, setPatients] = useState<PatientSummary[]>([])
  const [patientsLoading, setPatientsLoading] = useState(false)
  const [patientsError, setPatientsError] = useState<string | null>(null)

  const [selectedPatientId, setSelectedPatientId] = useState<string>("")
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputText, setInputText] = useState("")
  const [isSending, setIsSending] = useState(false)
  const [chatError, setChatError] = useState<string | null>(null)
  const [stepEvents, setStepEvents] = useState<AgentEvent[]>([])
  const [runId, setRunId] = useState<string | null>(null)
  const [monitoringLoading, setMonitoringLoading] = useState(false)
  const [monitoringError, setMonitoringError] = useState<string | null>(null)
  const runIdRef = useRef<string | null>(null)

  const selectedPatient = useMemo(
    () => patients.find((patient) => patient.id === selectedPatientId) ?? null,
    [patients, selectedPatientId]
  )

  useEffect(() => {
    setPatientsLoading(true)
    setPatientsError(null)
    fetch(PATIENTS_ENDPOINT)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(await response.text())
        }
        return response.json()
      })
      .then((data: PatientSummary[] | { patients: PatientSummary[] }) => {
        const list = Array.isArray(data) ? data : data.patients ?? []
        setPatients(list)
        if (list.length && !selectedPatientId) {
          setSelectedPatientId(list[0].id)
        }
      })
      .catch((error: Error) => {
        console.error("Failed to load patients", error)
        setPatientsError("Patientenliste konnte nicht geladen werden.")
      })
      .finally(() => setPatientsLoading(false))
  }, [selectedPatientId])

  useEffect(() => {
    setMessages([])
    setChatError(null)
    setStepEvents([])
    setRunId(null)
    runIdRef.current = null
    setMonitoringError(null)
    setMonitoringLoading(false)
  }, [selectedPatientId])

  const handleSend = useCallback(async () => {
    const trimmed = inputText.trim()
    if (!trimmed || isSending) {
      return
    }

    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: trimmed,
      createdAt: new Date().toISOString(),
    }

    setMessages([userMessage])
    setInputText("")
    setIsSending(true)
    setChatError(null)
    setStepEvents([])
    setMonitoringError(null)
    setRunId(null)
    runIdRef.current = null

    const streamUrl = `${CHAT_ENDPOINT}/stream`

    try {
      const response = await fetch(streamUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: selectedPatientId || null,
          question: trimmed,
        }),
      })

      if (!response.ok) {
        throw new Error(await response.text())
      }

      if (!response.body) {
        throw new Error("Streaming response did not include a body.")
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let streamFailed = false

      while (true) {
        const { value, done } = await reader.read()
        if (value) {
          buffer += decoder.decode(value, { stream: !done })
        } else if (done) {
          buffer += decoder.decode(new Uint8Array(), { stream: false })
        }

        let newlineIndex = buffer.indexOf("\n")
        while (newlineIndex !== -1) {
          const line = buffer.slice(0, newlineIndex).trim()
          buffer = buffer.slice(newlineIndex + 1)

          if (line) {
            let parsed: Record<string, unknown> | null = null
            try {
              parsed = JSON.parse(line)
            } catch (parseError) {
              console.warn("Failed to parse stream line", parseError, line)
            }

            if (parsed) {
              if (typeof parsed.run_id === "string") {
                runIdRef.current = parsed.run_id
                setRunId(parsed.run_id)
              }

              const eventRecord = normaliseEvent(parsed)
              if (eventRecord) {
                setStepEvents((prev) => [...prev, eventRecord])

                if (eventRecord.type === "run_completed") {
                  const payload = eventRecord.payload
                  const metadata =
                    payload["metadata"] && typeof payload["metadata"] === "object"
                      ? (payload["metadata"] as AgentMetadata)
                      : ({} as AgentMetadata)
                  const answerFromPayload =
                    typeof payload["answer"] === "string" ? (payload["answer"] as string) : ""
                  const answer = answerFromPayload || metadata.detailed_answer || metadata.clinical_brief || ""
                  const assistantMessage: ChatMessage = {
                    id: eventRecord.id,
                    role: "assistant",
                    content: answer,
                    createdAt: new Date().toISOString(),
                    metadata,
                  }
                  setMessages((prev) => {
                    const userOnly = prev.filter((msg) => msg.role === "user")
                    return [...userOnly, assistantMessage]
                  })
                }

                if (eventRecord.type === "run_failed") {
                  const failureMessage =
                    typeof eventRecord.payload["message"] === "string"
                      ? (eventRecord.payload["message"] as string)
                      : "An error occurred while running the agent. Please check the server logs."
                  setChatError(failureMessage)
                  setMessages((prev) => [
                    ...prev,
                    {
                      id: createId(),
                      role: "assistant",
                      content:
                        "Sorry, I couldn’t generate a response. Please try again or contact support.",
                      createdAt: new Date().toISOString(),
                    },
                  ])
                  streamFailed = true
                }
              }
            }
          }

          newlineIndex = buffer.indexOf("\n")
        }

        if (done || streamFailed) {
          if (streamFailed) {
            try {
              await reader.cancel()
            } catch {
              // ignore cancellation errors
            }
          }
          break
        }
      }
    } catch (error) {
      console.error("Chat stream failed", error)
      setChatError(
        "An error occurred while fetching the answer. Please retry or check the server logs."
      )
      setMessages((prev) => [
        ...prev,
        {
          id: createId(),
          role: "assistant",
          content:
            "Sorry, I couldn’t generate a response. Please try again or contact support.",
          createdAt: new Date().toISOString(),
        },
      ])
    } finally {
      setIsSending(false)
      if (runIdRef.current) {
        setRunId(runIdRef.current)
      }
    }
  }, [inputText, isSending, selectedPatientId])

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault()
        handleSend().catch((error) => console.error("Send failed", error))
      }
    },
    [handleSend]
  )

  const handleClearChat = useCallback(() => {
    setMessages([])
    setChatError(null)
    setStepEvents([])
    setRunId(null)
    runIdRef.current = null
    setMonitoringError(null)
    setMonitoringLoading(false)
  }, [])

  const refreshMonitoring = useCallback(async () => {
    const targetRunId = runIdRef.current ?? runId
    if (!targetRunId) {
      return
    }

    setMonitoringLoading(true)
    setMonitoringError(null)
    try {
      const response = await fetch(`${API_BASE}/monitor/runs/${targetRunId}`)
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const data = (await response.json()) as { events?: Array<Record<string, unknown>> }
      const eventsArray = Array.isArray(data.events) ? data.events : []
      const normalisedEvents = eventsArray
        .map((event) => normaliseEvent(event))
        .filter((event): event is AgentEvent => Boolean(event))
      setStepEvents(normalisedEvents)
      setRunId(targetRunId)
      runIdRef.current = targetRunId
    } catch (error) {
      console.error("Monitoring fetch failed", error)
      setMonitoringError(
        error instanceof Error
          ? error.message
          : "Unable to refresh monitoring."
      )
    } finally {
      setMonitoringLoading(false)
    }
  }, [runId])

  const patientDetails = selectedPatient?.details ?? selectedPatient?.description

  return (
    <div className="min-h-screen bg-muted/40">
      <header className="border-b bg-background">
        <div className="mx-auto flex h-16 w-full max-w-6xl items-center justify-between px-4">
          <div>
            <h1 className="text-lg font-semibold">Clinical RAG Assistant</h1>
            <p className="text-sm text-muted-foreground">
              Chat support for tumor board preparation
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" disabled={isSending} onClick={handleClearChat}>
              <Trash2 className="mr-2 h-4 w-4" /> Clear session
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-6xl gap-4 px-4 py-6 md:grid-cols-[260px_1fr]">
        <aside className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Patient</CardTitle>
              <CardDescription>Select a patient case to begin.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Select value={selectedPatientId} onValueChange={setSelectedPatientId} disabled={patientsLoading}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Choose patient" />
                </SelectTrigger>
                <SelectContent>
                  {patients.map((patient) => (
                    <SelectItem key={patient.id} value={patient.id}>
                      {patient.name || patient.id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {patientsLoading && (
                <p className="flex items-center text-sm text-muted-foreground">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading patients…
                </p>
              )}
              {patientsError && <p className="text-sm text-destructive">{patientsError}</p>}
            </CardContent>
          </Card>
          {patientDetails && (
            <Card>
              <CardHeader>
                <CardTitle>Patient Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground whitespace-pre-line">{patientDetails}</p>
              </CardContent>
            </Card>
          )}
          {chatError && (
            <Card className="border-destructive/50">
              <CardHeader>
                <CardTitle className="text-destructive">Error</CardTitle>
            </CardHeader>
              <CardContent>
                <p className="text-sm text-destructive/80">{chatError}</p>
              </CardContent>
            </Card>
          )}
        </aside>

        <section className="flex flex-col space-y-4">
          <Card className="flex h-[70vh] flex-col">
            <CardHeader>
              <CardTitle>Conversation</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-1 flex-col overflow-hidden">
              <ScrollArea className="flex-1 rounded-md border bg-background">
                <div className="space-y-4 p-4">
                  {messages.length === 0 && (
                    <p className="text-sm text-muted-foreground">
                      Ask a question about the selected patient to get started.
                    </p>
                  )}
                  {messages.map((message) => (
                    <MessageBubble key={message.id} message={message} />
                  ))}
                </div>
              </ScrollArea>
              <div className="mt-4 grid gap-3">
                <Textarea
                  rows={4}
                  value={inputText}
                  onChange={(event) => setInputText(event.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your question. Press Ctrl/⌘ + Enter to send (e.g. Does the patient meet the CAR-T eligibility criteria?)."
                  disabled={isSending}
                />
                <div className="flex justify-end gap-2">
                  {isSending && (
                    <p className="flex items-center text-sm text-muted-foreground">
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Assistant is thinking…
                    </p>
                  )}
                  <Button onClick={() => handleSend().catch(console.error)} disabled={isSending || !inputText.trim()}>
                    Send <Send className="ml-2 h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="flex h-[70vh] flex-col">
            <CardHeader>
              <CardTitle>Agent Monitoring</CardTitle>
              <CardDescription>Live steps and audit trail for the latest run.</CardDescription>
            </CardHeader>
            <CardContent className="flex-1 overflow-hidden">
              <ScrollArea className="h-full rounded-md border bg-background p-3">
                <AgentEventTimeline events={stepEvents} />
              </ScrollArea>
              {monitoringError && <p className="mt-3 text-sm text-destructive">{monitoringError}</p>}
              {runId && (
                <p className="mt-3 text-xs text-muted-foreground">
                  Run ID: <span className="font-mono">{runId}</span>
                </p>
              )}
            </CardContent>
            <CardFooter className="flex items-center justify-between gap-3">
              <span className="text-xs text-muted-foreground">
                {isSending
                  ? "Live stream active…"
                  : stepEvents.length
                    ? "Latest agent steps"
                    : "No events yet"}
              </span>
              <div className="flex items-center gap-2">
                {monitoringLoading && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => refreshMonitoring().catch(console.error)}
                  disabled={!runId || monitoringLoading}
                >
                  Refresh monitoring
                </Button>
              </div>
            </CardFooter>
          </Card>
        </section>
      </main>
    </div>
  )
}

function AgentEventTimeline({ events }: { events: AgentEvent[] }) {
  if (events.length === 0) {
    return <p className="text-sm text-muted-foreground">Noch keine Ereignisse.</p>
  }

  return (
    <ul className="space-y-3">
      {events.map((event) => {
        const summary = describeEvent(event)
        return (
          <li key={event.id} className="rounded-md border bg-muted/30 p-3 text-xs">
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium text-foreground">{formatEventLabel(event.type)}</span>
              <time className="text-[10px] uppercase text-muted-foreground">
                {formatTimestamp(event.timestamp)}
              </time>
            </div>
            {summary && <p className="mt-1 text-muted-foreground">{summary}</p>}
            <details className="mt-2">
              <summary className="cursor-pointer text-[10px] text-muted-foreground">Show details</summary>
              <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-background p-2 font-mono text-[11px] leading-snug">
                {JSON.stringify(event.payload, null, 2)}
              </pre>
            </details>
          </li>
        )
      })}
    </ul>
  )
}

function formatEventLabel(type: string): string {
  return type
    .split("_")
    .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : ""))
    .join(" ")
}

function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp * 1000)
  if (Number.isNaN(date.getTime())) {
    return ""
  }
  return date.toLocaleTimeString()
}

function safeString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null
}

function describeEvent(event: AgentEvent): string | null {
  const payload = event.payload
  switch (event.type) {
    case "run_started": {
      const question = safeString(payload["question"])
      return question ? `Question: ${question}` : "Run started."
    }
    case "plan_ready":
      return "Plan prepared."
    case "execution_step": {
      const stepIndex = typeof payload["step_index"] === "number" ? (payload["step_index"] as number) : null
      return stepIndex !== null ? `Executing plan step ${stepIndex + 1}.` : "Executing plan step."
    }
    case "tool_started": {
      const toolName = safeString(payload["tool_name"]) ?? safeString(payload["tool"])
      return toolName ? `Started tool ${toolName}.` : "Tool execution started."
    }
    case "tool_finished": {
      const toolName = safeString(payload["tool_name"]) ?? safeString(payload["tool"])
      const summary = safeString(payload["summary"])
      if (summary) {
        return toolName ? `Finished tool ${toolName}: ${summary}` : `Finished tool: ${summary}`
      }
      return toolName ? `Finished tool ${toolName}.` : "Tool execution finished."
    }
    case "run_failed": {
      const message = safeString(payload["message"])
      return message ? `Error: ${message}` : "Run failed."
    }
    case "run_completed":
      return "Assistant response generated."
    default:
      return null
  }
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isAssistant = message.role !== "user"
  const metadata = message.metadata
  const summary = isAssistant ? metadata?.clinical_brief?.trim() ?? null : null
  let bodyContent: string | null = typeof message.content === "string" ? message.content : null

  if (summary && bodyContent) {
    const leadingTrimmed = bodyContent.trimStart()
    if (leadingTrimmed.startsWith(summary)) {
      bodyContent = leadingTrimmed.slice(summary.length)
      bodyContent = bodyContent.replace(/^\s+/, "")
    }
    if (bodyContent.trim().length === 0) {
      bodyContent = null
    }
  }

  return (
    <div className={cn("flex flex-col gap-2", isAssistant ? "items-start" : "items-end")}
    >
      <div
        className={cn(
          "w-full max-w-2xl rounded-lg border p-4 text-sm shadow-sm",
          isAssistant ? "bg-card" : "bg-primary text-primary-foreground"
        )}
      >
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>{isAssistant ? "Assistant" : "You"}</span>
          <time dateTime={message.createdAt}>{new Date(message.createdAt).toLocaleTimeString()}</time>
        </div>
        {summary && (
          <div className="mt-2 rounded-md bg-muted p-3 text-sm">
            <strong className="block text-xs uppercase text-muted-foreground">Summary</strong>
            <p>{summary}</p>
          </div>
        )}
        {bodyContent ? (
          <div className="mt-3 whitespace-pre-wrap text-sm leading-relaxed">{bodyContent}</div>
        ) : null}
        {metadata?.missing_information?.length && isAssistant ? (
          <div className="mt-3 space-y-1 rounded-md border border-dashed p-3 text-sm">
            <strong className="block text-xs uppercase text-muted-foreground">Missing information</strong>
            <ul className="list-disc pl-4">
              {metadata.missing_information.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ) : null}
        {metadata?.context_nodes?.length ? (
          <details className="mt-3">
            <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
              Context snippets ({metadata.context_nodes.length})
            </summary>
            <ul className="mt-2 space-y-2 text-xs text-muted-foreground">
              {metadata.context_nodes.slice(0, 5).map((node, index) => (
                <li key={`${node.section_name ?? node.report_type}-${index}`}>
                  <span className="font-medium">{node.report_type ?? node.section_name ?? "Section"}:</span>
                  <span className="ml-1">{node.snippet ?? "(no snippet provided)"}</span>
                </li>
              ))}
              {metadata.context_nodes.length > 5 && <li>… additional snippets available on the server.</li>}
            </ul>
          </details>
        ) : null}
      </div>
    </div>
  )
}
