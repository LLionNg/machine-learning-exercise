import { useEffect, useRef, useState } from "react"
import {
  ArrowLeft,
  BookOpen,
  Check,
  Code2,
  Copy,
  FlaskConical,
  Lightbulb,
  ListChecks,
  Loader2,
  Maximize2,
  Minimize2,
  Play,
  RotateCcw,
  Save,
  Send,
  Trophy,
  X,
} from "lucide-react"
import {
  api,
  type ChallengeDetail,
  type RunResult,
  type SaveResult,
  type ScoreRow,
} from "@/api"
import { navigate } from "@/App"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { CodeEditor } from "@/components/CodeEditor"

type Tab = "solution" | "tests" | "results" | "scoreboard" | "hints" | "learning"

const TABS: { key: Tab; label: string; icon: typeof Code2 }[] = [
  { key: "solution", label: "Solution", icon: Code2 },
  { key: "tests", label: "Tests", icon: FlaskConical },
  { key: "results", label: "Results", icon: ListChecks },
  { key: "scoreboard", label: "Scoreboard", icon: Trophy },
  { key: "hints", label: "Hints", icon: Lightbulb },
  { key: "learning", label: "Learning", icon: BookOpen },
]

const REPO_NAME = "machine-learning-exercise"

// Where a per-challenge draft of the editor's code is auto-saved in the browser,
// so work survives a reload or restart (same idea as go-interview).
const draftKey = (id: string) => `challenge_${id}_code`

export default function Challenge({
  id,
  username,
}: {
  id: string
  username: string
}) {
  const [detail, setDetail] = useState<ChallengeDetail | null>(null)
  const [scores, setScores] = useState<ScoreRow[]>([])
  const [notFound, setNotFound] = useState(false)
  const [tab, setTab] = useState<Tab>("solution")
  const [code, setCode] = useState("")
  const [focus, setFocus] = useState(false)
  const [confirmReset, setConfirmReset] = useState(false)
  const [revealed, setRevealed] = useState(0)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<RunResult | null>(null)
  const [submitMode, setSubmitMode] = useState(false)
  const [saved, setSaved] = useState<SaveResult | null>(null)
  const [copied, setCopied] = useState(false)
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved">("idle")

  // `dirty` = the user has edited this challenge's code (so we don't clobber
  // their work with a loaded submission). `usernameRef` lets the load effect
  // read the latest username without re-running when it resolves.
  const dirtyRef = useRef(false)
  const usernameRef = useRef(username)
  usernameRef.current = username

  // Load the challenge, then restore code with priority:
  // saved filesystem submission > local draft > template.
  useEffect(() => {
    setDetail(null)
    setNotFound(false)
    setResult(null)
    setSaved(null)
    setSubmitMode(false)
    setTab("solution")
    setConfirmReset(false)
    setRevealed(0)
    setSaveState("idle")
    dirtyRef.current = false
    let cancelled = false
    ;(async () => {
      let d: ChallengeDetail
      try {
        d = await api.challenge(id)
      } catch {
        if (!cancelled) setNotFound(true)
        return
      }
      if (cancelled) return
      setDetail(d)
      let initial = localStorage.getItem(draftKey(id)) ?? d.template
      const user = usernameRef.current.trim()
      if (user) {
        try {
          const s = await api.submission(id, user)
          if (s.exists) initial = s.code
        } catch {
          // ignore -- fall back to draft/template
        }
      }
      if (!cancelled && !dirtyRef.current) setCode(initial)
    })()
    api.scoreboard(id).then((s) => !cancelled && setScores(s)).catch(() => setScores([]))
    return () => {
      cancelled = true
    }
  }, [id])

  // If the username resolves after the challenge loaded, pull in a saved
  // submission (unless the user has already started editing).
  useEffect(() => {
    const user = username.trim()
    if (!user || dirtyRef.current) return
    let cancelled = false
    api
      .submission(id, user)
      .then((s) => {
        if (!cancelled && s.exists && !dirtyRef.current) setCode(s.code)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [id, username])

  // Auto-save the user's edits to a local draft (debounced).
  useEffect(() => {
    if (!dirtyRef.current) return
    const t = setTimeout(() => {
      localStorage.setItem(draftKey(id), code)
      setSaveState("saved")
    }, 800)
    return () => clearTimeout(t)
  }, [code, id])

  const editCode = (v: string) => {
    dirtyRef.current = true
    setSaveState("saving")
    setCode(v)
  }

  const resetToTemplate = () => {
    if (!detail) return
    localStorage.removeItem(draftKey(id))
    dirtyRef.current = false
    setSaveState("idle")
    setCode(detail.template)
    setConfirmReset(false)
  }

  const grade = async (asSubmit: boolean) => {
    setRunning(true)
    setResult(null)
    setSaved(null)
    setSubmitMode(asSubmit)
    setTab("results")
    try {
      const r = await api.run(id, code)
      setResult(r)
      // On a passing submit, save the solution to the repo automatically so the
      // pull-request / scoreboard steps are ready (like go-interview's submit).
      if (asSubmit && r.ok && username.trim()) {
        try {
          setSaved(await api.save(id, username.trim(), code))
          // The save just regraded and rewrote this challenge's scoreboard --
          // pull the fresh rows so the Scoreboard tab reflects it immediately.
          api.scoreboard(id).then(setScores).catch(() => {})
        } catch {
          // leave `saved` null -- the manual "Save to filesystem" button remains
        }
      }
    } catch {
      setResult({ ok: false, passed: 0, total: 0, failed: 0, durationMs: 0, output: "Failed to reach the grader." })
    } finally {
      setRunning(false)
    }
  }

  const saveToFilesystem = async () => {
    if (!username.trim()) return
    try {
      setSaved(await api.save(id, username.trim(), code))
      api.scoreboard(id).then(setScores).catch(() => {})
    } catch {
      setSaved(null)
    }
  }

  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // ignore
    }
  }

  if (notFound) {
    return (
      <div className="py-16 text-center">
        <p className="text-sm text-muted-foreground">Challenge not found.</p>
        <Button variant="outline" className="mt-4" onClick={() => navigate("#/")}>
          <ArrowLeft size={16} /> Back to challenges
        </Button>
      </div>
    )
  }

  if (!detail) {
    return <p className="py-16 text-center text-sm text-muted-foreground">Loading...</p>
  }

  const editorHeight = focus ? "calc(100vh - 15rem)" : "560px"
  const paneH = focus ? "h-[calc(100vh-15rem)]" : "h-[560px]"

  return (
    <div className="mx-auto max-w-[104rem] space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="ghost" size="sm" onClick={() => navigate("#/")}>
          <ArrowLeft size={16} /> Challenges
        </Button>
        <h1 className="text-xl font-semibold">{detail.title}</h1>
        <Badge variant="outline">{detail.difficulty}</Badge>
        {detail.estimatedTime && (
          <span className="text-xs text-muted-foreground">{detail.estimatedTime}</span>
        )}
      </div>

      <div className={focus ? "" : "grid items-start gap-6 lg:grid-cols-12"}>
        {/* Left: problem description (persistent) */}
        {!focus && (
          <Card className="lg:col-span-5">
            <div className="border-b border-border/60 px-5 py-2.5 text-sm font-medium text-muted-foreground">
              Problem
            </div>
            <CardContent className="max-h-[640px] overflow-auto p-5">
              <div className="prose max-w-none" dangerouslySetInnerHTML={{ __html: detail.readmeHtml }} />
            </CardContent>
          </Card>
        )}

        {/* Right: tabbed workspace */}
        <div className={focus ? "" : "lg:col-span-7"}>
          <Card className="flex flex-col overflow-hidden">
            {/* Tab bar */}
            <div className="flex gap-1 overflow-x-auto border-b border-border/60 p-2">
              {TABS.map((t) => {
                const Icon = t.icon
                return (
                  <button
                    key={t.key}
                    onClick={() => setTab(t.key)}
                    className={
                      "inline-flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors " +
                      (tab === t.key
                        ? "bg-secondary text-secondary-foreground"
                        : "text-muted-foreground hover:bg-muted/60")
                    }
                  >
                    <Icon size={14} />
                    {t.label}
                  </button>
                )
              })}
            </div>

            {tab === "solution" && (
              <div>
                <div className="flex items-center gap-2 border-b border-border/60 px-3 py-1.5">
                  <span className="font-mono text-xs text-muted-foreground">solution.py</span>
                  {saveState !== "idle" && (
                    <span className="text-xs text-muted-foreground">
                      {saveState === "saving" ? "Saving..." : "Saved locally"}
                    </span>
                  )}
                  <div className="ml-auto flex items-center gap-1.5">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setFocus((f) => !f)}
                      title={focus ? "Exit focus mode" : "Focus on the editor"}
                    >
                      {focus ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
                      {focus ? "Exit focus" : "Focus"}
                    </Button>
                    {confirmReset ? (
                      <span className="flex items-center gap-1 text-xs text-muted-foreground">
                        Reset to template?
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 px-2 text-destructive"
                          onClick={resetToTemplate}
                        >
                          <Check size={14} /> Yes
                        </Button>
                        <Button variant="ghost" size="sm" className="h-7 px-2" onClick={() => setConfirmReset(false)}>
                          <X size={14} /> No
                        </Button>
                      </span>
                    ) : (
                      <Button variant="ghost" size="sm" onClick={() => setConfirmReset(true)}>
                        <RotateCcw size={14} /> Reset
                      </Button>
                    )}
                  </div>
                </div>
                <CodeEditor name="solution-editor" value={code} onChange={editCode} height={editorHeight} />
              </div>
            )}

            {tab === "tests" && (
              <div>
                <div className="flex items-center gap-2 border-b border-border/60 px-3 py-1.5">
                  <span className="font-mono text-xs text-muted-foreground">test_solution.py</span>
                  <span className="ml-auto text-xs text-muted-foreground">read-only - the canonical tests</span>
                </div>
                <CodeEditor
                  name="tests-editor"
                  value={detail.testFile}
                  readOnly
                  showStatus={false}
                  height={editorHeight}
                />
              </div>
            )}

            {tab === "results" && (
              <div className={`${paneH} overflow-auto p-5`}>
                {running ? (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 size={16} className="animate-spin" /> Running tests...
                  </div>
                ) : result ? (
                  <div className="space-y-4">
                    <div className="flex flex-wrap items-center gap-3">
                      <Badge variant={result.ok ? "success" : "destructive"}>
                        {result.ok ? "All tests passed" : "Tests failed"}
                      </Badge>
                      <span className="text-sm tabular-nums">
                        <strong>{result.passed}</strong>/{result.total} passed
                      </span>
                      {result.failed > 0 && (
                        <span className="text-sm text-muted-foreground">{result.failed} failed</span>
                      )}
                      <span className="ml-auto text-xs text-muted-foreground">
                        {(result.durationMs / 1000).toFixed(1)}s
                      </span>
                    </div>
                    {result.output && (
                      <pre className="overflow-auto rounded-lg bg-muted p-3 font-mono text-xs leading-relaxed">
                        {result.output}
                      </pre>
                    )}

                    {submitMode && result.ok && (
                      <div className="rounded-lg border border-border bg-muted/40 p-4">
                        <p className="font-medium text-success">Solution submitted - all tests passed.</p>
                        <p className="mt-1 text-sm text-muted-foreground">
                          To appear on the leaderboard, save your solution to the repo and open a pull request.
                        </p>
                        {!saved ? (
                          <Button className="mt-3" onClick={saveToFilesystem}>
                            <Save size={14} /> Save to filesystem
                          </Button>
                        ) : (
                          <div className="mt-4 space-y-4">
                            <p className="text-sm">
                              Saved to{" "}
                              <code className="rounded bg-muted px-1.5 py-0.5 text-xs">{saved.filePath}</code>
                            </p>
                            <div>
                              <div className="mb-1 flex items-center justify-between">
                                <span className="text-xs font-medium text-muted-foreground">
                                  1. Commit &amp; push to your fork
                                </span>
                                <Button variant="ghost" size="sm" onClick={() => copy(saved.gitCommands.join("\n"))}>
                                  {copied ? <Check size={14} /> : <Copy size={14} />} Copy
                                </Button>
                              </div>
                              <pre className="overflow-x-auto rounded-lg bg-muted p-3 font-mono text-xs">
                                {saved.gitCommands.join("\n")}
                              </pre>
                            </div>
                            <div>
                              <span className="text-xs font-medium text-muted-foreground">
                                2. Create a pull request
                              </span>
                              <ol className="mt-1 list-decimal space-y-0.5 pl-5 text-sm text-muted-foreground">
                                <li>
                                  Go to{" "}
                                  <a
                                    className="text-foreground underline"
                                    target="_blank"
                                    rel="noreferrer"
                                    href={`https://github.com/${username}/${REPO_NAME}`}
                                  >
                                    your fork on GitHub
                                  </a>
                                </li>
                                <li>Open a pull request against the upstream repo</li>
                                <li>Title it "Add {detail.title} solution" and submit</li>
                              </ol>
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {submitMode && !result.ok && (
                      <p className="text-sm text-muted-foreground">
                        Not all tests passed yet - fix the failures above, then submit again.
                      </p>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">Run your code to see test results.</p>
                )}
              </div>
            )}

            {tab === "scoreboard" && (
              <div className={`${paneH} overflow-auto p-5`}>
                {scores.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No submissions yet.</p>
                ) : (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-xs text-muted-foreground">
                        <th className="pb-1 font-medium">User</th>
                        <th className="pb-1 text-right font-medium">Passed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scores.map((r) => (
                        <tr key={r.username} className="border-t border-border/50">
                          <td className="py-1.5">{r.username}</td>
                          <td className="py-1.5 text-right tabular-nums">
                            <span className={r.passed === r.total ? "text-success" : "text-muted-foreground"}>
                              {r.passed}/{r.total}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}

            {tab === "hints" && (
              <div className={`${paneH} overflow-auto p-5`}>
                {detail.hints.length === 0 ? (
                  <div className="prose max-w-none" dangerouslySetInnerHTML={{ __html: detail.hintsHtml }} />
                ) : (
                  <div className="mx-auto max-w-2xl">
                    <div className="mb-5 text-center">
                      <Lightbulb size={28} className="mx-auto mb-2 text-amber-500" />
                      <h3 className="font-semibold">Progressive Hints</h3>
                      <p className="text-sm text-muted-foreground">
                        Reveal one hint at a time - try each before moving on.
                      </p>
                    </div>

                    <div className="space-y-3">
                      {detail.hints.slice(0, revealed).map((h, i) => (
                        <div
                          key={i}
                          className="hint-in rounded-lg border border-amber-500/25 bg-amber-500/5 p-4"
                        >
                          <div className="mb-2 flex items-center gap-2">
                            <span className="inline-flex items-center rounded-full bg-amber-500/15 px-2 py-0.5 text-xs font-semibold text-amber-600 dark:text-amber-400">
                              Hint {i + 1}
                            </span>
                            {h.title && <span className="text-sm font-medium">{h.title}</span>}
                          </div>
                          <div className="prose max-w-none" dangerouslySetInnerHTML={{ __html: h.html }} />
                        </div>
                      ))}
                    </div>

                    <div className="mt-5 flex flex-col items-center gap-2">
                      <div className="flex gap-2">
                        {revealed < detail.hints.length && (
                          <Button variant="outline" onClick={() => setRevealed((r) => r + 1)}>
                            <Lightbulb size={14} /> Show next hint
                          </Button>
                        )}
                        {revealed > 0 && (
                          <Button variant="ghost" onClick={() => setRevealed(0)}>
                            <RotateCcw size={14} /> Reset
                          </Button>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {revealed} of {detail.hints.length} hints revealed
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {tab === "learning" && (
              <div className={`${paneH} overflow-auto p-5`}>
                <div className="prose max-w-none" dangerouslySetInnerHTML={{ __html: detail.learningHtml }} />
              </div>
            )}

            {/* Bottom action bar */}
            <div className="flex flex-wrap items-center gap-2 border-t border-border/60 px-3 py-2.5">
              <Button onClick={() => grade(false)} disabled={running}>
                {running ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
                Run tests
              </Button>
              <Button
                variant="outline"
                className="ml-auto"
                onClick={() => grade(true)}
                disabled={running || !username.trim()}
                title={
                  username
                    ? `Submit as ${username}`
                    : "GitHub username not detected (set git config user.name or your fork's origin remote)"
                }
              >
                <Send size={14} /> Submit
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
