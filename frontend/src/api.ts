export interface ChallengeSummary {
  id: string
  title: string
  shortDescription: string
  difficulty: string
  category: string
  tags: string[]
  estimatedTime: string
  testCount: number | null
  order: number
}

export interface HintItem {
  title: string
  html: string
}

export interface ChallengeDetail extends ChallengeSummary {
  template: string
  testFile: string
  readmeHtml: string
  learningHtml: string
  hintsHtml: string
  hints: HintItem[]
}

export interface RunResult {
  ok: boolean
  passed: number
  total: number
  failed: number
  durationMs: number
  output: string
}

export interface ScoreRow {
  username: string
  passed: number
  total: number
}

export interface LeaderUser {
  username: string
  solved: number
  challenges: Record<string, boolean>
}

export interface Leaderboard {
  totalChallenges: number
  users: LeaderUser[]
}

export interface SaveResult {
  success: boolean
  filePath: string
  gitCommands: string[]
}

export interface Submission {
  code: string
  exists: boolean
}

export interface GitUsername {
  username: string
  source: string
  success: boolean
}

export interface UserStats {
  attempted: number
  completed: number
  avgScore: number
  rank: number
  totalChallenges: number
  success: boolean
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
  return (await res.json()) as T
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`)
  return (await res.json()) as T
}

export const api = {
  challenges: () => getJSON<ChallengeSummary[]>("/api/challenges"),
  challenge: (id: string) => getJSON<ChallengeDetail>(`/api/challenges/${id}`),
  scoreboard: (id: string) => getJSON<ScoreRow[]>(`/api/scoreboard/${id}`),
  leaderboard: () => getJSON<Leaderboard>("/api/leaderboard"),
  gitUsername: () => getJSON<GitUsername>("/api/git-username"),
  userStats: (username: string) => getJSON<UserStats>(`/api/user-stats/${encodeURIComponent(username)}`),
  submission: (challengeId: string, username: string) =>
    getJSON<Submission>(`/api/submission/${challengeId}/${encodeURIComponent(username)}`),
  run: (challengeId: string, code: string) =>
    postJSON<RunResult>("/api/run", { challengeId, code }),
  save: (challengeId: string, username: string, code: string) =>
    postJSON<SaveResult>("/api/save", { challengeId, username, code }),
}
