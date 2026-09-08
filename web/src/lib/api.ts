const BASE = "/api";

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { signal });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

async function postResponse<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let message = `API error: ${res.status}`;
    try {
      const payload = await res.json();
      if (payload?.error) message = payload.error;
    } catch {
      // Keep the status-based message.
    }
    throw new Error(message);
  }
  return res.json();
}

/* Types */

export interface Attempt {
  commit_hash: string;
  agent_id: string;
  title: string;
  score: number | null;
  status: string;
  parent_hash: string | null;
  timestamp: string;
  feedback: string;
}

export interface DagNode {
  id: string;
  parent: string | null;
  is_root: boolean;
  agent_id: string;
  score: number | null;
  status: string;
  title: string;
  timestamp: string;
  is_best: boolean;
  user_best?: boolean;
}

export interface DagResponse {
  nodes: DagNode[];
  edges: { from: string; to: string }[];
}

export interface Note {
  audit?: {
    version: number;
    note_sha256: string;
    evidence_sha256: Record<string, string>;
    checks: Record<string, {
      status: "passed" | "failed" | "unchecked";
      code: string;
      message: string;
    }>;
    actual_delta?: number;
    improved?: boolean;
  };
  date: string;
  title: string;
  body: string;
  creator?: string;
  filename?: string;
  relative_path?: string;
  category?: string;
  index: number;
  // Structured-trace fields (optional; present when the note carries the schema).
  type?: string;
  status?: string;
  confidence?: ConfidenceLevel | number | string | null;
  based_on?: string;
  touched?: string[] | string;
  // Provenance fields — present on raw/ source captures (category "raw").
  source_url?: string;
  source_type?: string;
  also_confirmed_by?: string[] | string;
  // Complete frontmatter passthrough — every field the author wrote, in order.
  frontmatter?: Record<string, unknown>;
}

export type ConfidenceLevel = "low" | "medium" | "high";

export interface NoteGraphNode {
  id: string;
  title: string;
  type: string;
  status?: string | null;
  // Schema is the enum; the `number` branch is for legacy notes written
  // under the older float-based schema and is migrated by the renderer.
  confidence?: ConfidenceLevel | number | string | null;
  creator: string;
  island_id?: string | null;
  date: string;
  based_on?: string | null;
}

export interface NoteGraphEdge {
  from: string;
  to: string;
  kind: "supersedes" | "refutes" | "references";
}

export interface NotesGraphResponse {
  nodes: NoteGraphNode[];
  edges: NoteGraphEdge[];
}

export interface Skill {
  name: string;
  description: string;
  creator: string;
  created: string;
  path: string;
}

export interface SkillDetail {
  content: string;
  metadata: Record<string, string>;
  body: string;
  files: string[];
}

export interface AgentStatus {
  agent_id: string;
  status: "active" | "idle" | "stopped";
  sessions: number;
  last_activity: number;
  attempts: number;
  best_score: number | null;
}

export interface RunStatus {
  manager_alive: boolean;
  manager_pid: number | null;
  eval_count: number;
  total_attempts: number;
  scored_attempts: number;
  crashed_attempts: number;
  best_score: number | null;
  best_title: string | null;
  agents: AgentStatus[];
}

export interface SteeringAction {
  id: string;
  kind: "continue_from" | "mark_best";
  hash: string;
  instruction?: string;
  created_at: string;
  applied_at: string | null;
}

export interface SteeringResponse {
  actions: SteeringAction[];
  pending_count: number;
}

export interface LogEntry {
  type:
    | "thinking" | "tool_call" | "tool_result" | "text" | "system" | "error"
    | "coral_prompt" | "subagent_start" | "subagent_progress" | "subagent_done"
    | "compact" | "result";
  content: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  details: Record<string, any>;
  timestamp: string;
}

export interface LogTurn {
  index: number;
  entries: LogEntry[];
  usage: {
    input_tokens?: number;
    output_tokens?: number;
    cache_creation?: number;
    cache_read?: number;
  };
  timestamp: string;
}

export interface SessionMeta {
  total_cost_usd?: number;
  duration_ms?: number;
  duration_api_ms?: number;
  num_turns?: number;
  stop_reason?: string;
  session_id?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  usage?: Record<string, any>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  model_usage?: Record<string, any>;
}

export interface LogSession {
  session_index: number;
  turns: LogTurn[];
  meta?: SessionMeta;
}

export interface LogData {
  agent_id: string;
  log_files: Array<{
    path: string;
    index: number;
    size_bytes: number;
    modified: number;
  }>;
  turns: LogTurn[];
  sessions?: LogSession[];
  agent_meta?: {
    total_cost_usd?: number;
    duration_ms?: number;
    duration_api_ms?: number;
    num_turns?: number;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    usage?: Record<string, any>;
  };
}

export interface RunInfo {
  timestamp: string;
  status: "running" | "stopped";
  attempts: number;
  is_latest: boolean;
}

export interface TaskRuns {
  slug: string;
  runs: RunInfo[];
}

export interface RunRoot {
  id: string;
  label: string;
  tasks: TaskRuns[];
}

export interface RunsResponse {
  current: { root?: string; task: string; run: string };
  tasks: TaskRuns[];
  roots?: RunRoot[];
}

export interface TaskConfig {
  task: {
    name: string;
    description: string;
    files?: string[];
    tips?: string[];
  };
  grader: {
    type: string;
    timeout?: number;
    direction?: "maximize" | "minimize";
  };
  agents: {
    count: number;
    model: string;
    max_turns: number;
    reflect_every: number;
  };
}

/* API functions */

export const api = {
  config: () => get<TaskConfig>("/config"),
  attempts: () => get<Attempt[]>("/attempts"),
  dag: () => get<DagResponse>("/dag"),
  steering: () => get<SteeringResponse>("/steer"),
  steer: (body: { kind: "continue_from"; hash: string; instruction?: string } | { kind: "mark_best"; hash: string }) =>
    postResponse<{ action: SteeringAction | { kind: "mark_best"; hash: string; applied: boolean } }>("/steer", body),
  leaderboard: (top = 20) => get<Attempt[]>(`/leaderboard?top=${top}`),
  attempt: (hash: string) => get<Attempt>(`/attempts/${hash}`),
  agentAttempts: (id: string) => get<Attempt[]>(`/attempts/agent/${id}`),
  notes: () => get<Note[]>("/notes"),
  notesGraph: () => get<NotesGraphResponse>("/notes/graph"),
  skills: () => get<Skill[]>("/skills"),
  skill: (name: string) => get<SkillDetail>(`/skills/${name}`),
  logs: (agentId: string, signal?: AbortSignal) => get<LogData>(`/logs/${agentId}`, signal),
  logsList: () => get<Record<string, Array<{ path: string; index: number; size_bytes: number; modified: number }>>>("/logs"),
  status: () => get<RunStatus>("/status"),
  runs: () => get<RunsResponse>("/runs"),
  switchRun: (root: string, task: string, run: string) =>
    post<{ ok: boolean; root: string; task: string; run: string }>("/runs/switch", { root, task, run }),
};
