import { useEffect, useRef, useState } from "react"
import { ArrowRight, Check, ChevronDown, Clock, Trophy } from "lucide-react"
import { api, type ChallengeSummary, type Leaderboard } from "@/api"
import { navigate } from "@/App"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

function difficultyVariant(difficulty: string): "success" | "default" | "destructive" | "outline" {
  const d = difficulty.toLowerCase()
  if (d.startsWith("beginner") && !d.includes("intermediate")) return "success"
  if (d.includes("advanced")) return "destructive"
  if (d.includes("intermediate")) return "default"
  return "outline"
}

// Rank difficulties easiest-first (like go-interview's beginner/intermediate/
// advanced = 1/2/3), with our combined "Beginner-Intermediate" tier in between.
function difficultyRank(difficulty: string): number {
  const d = difficulty.toLowerCase()
  if (d.includes("advanced")) return 4
  if (d.includes("beginner") && d.includes("intermediate")) return 2
  if (d.includes("intermediate")) return 3
  if (d.includes("beginner")) return 1
  return 5
}

type SortKey = "difficulty" | "number-asc" | "number-desc"

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "difficulty", label: "Difficulty" },
  { key: "number-asc", label: "Number (asc)" },
  { key: "number-desc", label: "Number (desc)" },
]

// A custom dropdown (rather than a native <select>) so the option list is
// themed by our tokens and stays correct in dark mode.
function SortDropdown({ value, onChange }: { value: SortKey; onChange: (k: SortKey) => void }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener("mousedown", onDown)
    return () => document.removeEventListener("mousedown", onDown)
  }, [open])

  const current = SORT_OPTIONS.find((o) => o.key === value) ?? SORT_OPTIONS[0]

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex h-8 items-center gap-1.5 rounded-md border border-input bg-background px-2.5 text-sm text-foreground transition-colors hover:bg-muted"
      >
        {current.label}
        <ChevronDown size={14} className="text-muted-foreground" />
      </button>
      {open && (
        <div className="absolute right-0 z-20 mt-1 w-40 rounded-md border border-border bg-background p-1 shadow-lg">
          {SORT_OPTIONS.map((o) => (
            <button
              key={o.key}
              onClick={() => {
                onChange(o.key)
                setOpen(false)
              }}
              className={
                "flex w-full items-center justify-between rounded-md px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-muted " +
                (o.key === value ? "text-foreground" : "text-muted-foreground")
              }
            >
              {o.label}
              {o.key === value && <Check size={14} />}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Home() {
  const [challenges, setChallenges] = useState<ChallengeSummary[]>([])
  const [board, setBoard] = useState<Leaderboard | null>(null)
  const [loading, setLoading] = useState(true)
  const [sort, setSort] = useState<SortKey>("difficulty")

  useEffect(() => {
    Promise.all([api.challenges(), api.leaderboard()])
      .then(([c, b]) => {
        setChallenges(c)
        setBoard(b)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return <p className="py-16 text-center text-sm text-muted-foreground">Loading...</p>
  }

  const sortedChallenges = [...challenges].sort((a, b) => {
    if (sort === "number-asc") return a.order - b.order
    if (sort === "number-desc") return b.order - a.order
    return difficultyRank(a.difficulty) - difficultyRank(b.difficulty) || a.order - b.order
  })

  return (
    <div className="mx-auto grid max-w-6xl gap-6 lg:grid-cols-3">
      {/* Challenges */}
      <section className="space-y-3 lg:col-span-2">
        <div className="flex items-center justify-between gap-3">
          <h1 className="text-xl font-semibold">Challenges</h1>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Sort</span>
            <SortDropdown value={sort} onChange={setSort} />
            <span className="text-sm text-muted-foreground">{challenges.length} available</span>
          </div>
        </div>
        {sortedChallenges.map((c) => (
          <button
            key={c.id}
            onClick={() => navigate(`#/challenge/${c.id}`)}
            className="group block w-full text-left"
          >
            <Card className="transition-colors hover:border-foreground/20 hover:bg-muted/40">
              <CardContent className="p-5">
                <div className="flex items-start justify-between gap-3">
                  <h2 className="font-semibold leading-tight">{c.title}</h2>
                  <ArrowRight
                    size={16}
                    className="mt-0.5 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5"
                  />
                </div>
                <p className="mt-1.5 text-sm text-muted-foreground">{c.shortDescription}</p>
                <div className="mt-3 flex flex-wrap items-center gap-1.5">
                  <Badge variant="default">Challenge #{c.order}</Badge>
                  <Badge variant={difficultyVariant(c.difficulty)}>{c.difficulty}</Badge>
                  {c.estimatedTime && (
                    <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                      <Clock size={12} /> {c.estimatedTime}
                    </span>
                  )}
                  {c.testCount != null && (
                    <span className="text-xs text-muted-foreground">| {c.testCount} tests</span>
                  )}
                  <span className="ml-auto flex flex-wrap gap-1">
                    {c.tags.slice(0, 3).map((t) => (
                      <Badge key={t} variant="outline" className="font-normal text-muted-foreground">
                        {t}
                      </Badge>
                    ))}
                  </span>
                </div>
              </CardContent>
            </Card>
          </button>
        ))}
      </section>

      {/* Leaderboard */}
      <aside id="leaderboard" className="lg:col-span-1">
        <Card className="lg:sticky lg:top-20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Trophy size={16} /> Leaderboard
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {!board || board.users.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No submissions yet. Solve a challenge to appear here.
              </p>
            ) : (
              board.users.map((u, i) => (
                <div key={u.username} className="flex items-center gap-3 rounded-md px-1 py-1.5">
                  <span className="w-6 shrink-0 text-center text-sm">
                    <span className={i < 3 ? "font-semibold text-foreground" : "text-muted-foreground"}>
                      {i + 1}
                    </span>
                  </span>
                  <span className="flex-1 truncate text-sm font-medium">{u.username}</span>
                  <span className="flex shrink-0 gap-1">
                    {challenges.map((c) => (
                      <span
                        key={c.id}
                        title={c.title}
                        className={
                          "h-2.5 w-2.5 rounded-full " +
                          (u.challenges[c.id] ? "bg-success" : "bg-muted-foreground/25")
                        }
                      />
                    ))}
                  </span>
                  <span className="w-10 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
                    {u.solved}/{board.totalChallenges}
                  </span>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </aside>
    </div>
  )
}
