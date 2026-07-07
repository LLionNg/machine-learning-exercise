import { useEffect, useRef, useState } from "react"
import {
  ChevronDown,
  Github,
  Moon,
  Pencil,
  RotateCcw,
  Sun,
  Terminal,
  Trophy,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { api, type UserStats } from "@/api"
import Home from "@/views/Home"
import Challenge from "@/views/Challenge"

interface Identity {
  username: string
  source: string
}

function useDarkMode() {
  const [dark, setDark] = useState<boolean>(() => {
    const stored = localStorage.getItem("theme")
    if (stored) return stored === "dark"
    return window.matchMedia("(prefers-color-scheme: dark)").matches
  })
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
    localStorage.setItem("theme", dark ? "dark" : "light")
  }, [dark])
  return [dark, setDark] as const
}

// The GitHub username: a manually-set value (persisted) wins; otherwise it is
// auto-detected from git on load (fork origin remote -> git config), exactly
// like go-interview. "Change Username" in the profile lets the user override.
function useIdentity(): [Identity, (name: string) => void] {
  const [identity, setState] = useState<Identity>(() => {
    const stored = localStorage.getItem("gh-username")
    return stored && stored !== "undefined"
      ? { username: stored, source: "localStorage" }
      : { username: "", source: "" }
  })

  useEffect(() => {
    localStorage.removeItem("ml-username")
    localStorage.removeItem("gh-identity")
    if (localStorage.getItem("gh-username")) return // manual override wins
    api
      .gitUsername()
      .then((r) => {
        if (r.success && r.username && r.username.trim()) {
          setState({ username: r.username.trim(), source: r.source })
        }
      })
      .catch(() => {})
  }, [])

  const setUsername = (name: string) => {
    const trimmed = name.trim()
    if (trimmed) {
      localStorage.setItem("gh-username", trimmed)
      setState({ username: trimmed, source: "manual" })
    } else {
      localStorage.removeItem("gh-username")
      setState({ username: "", source: "" })
    }
  }

  return [identity, setUsername]
}

const SOURCE_LABELS: Record<string, string> = {
  "remote-origin": "Auto-detected from git remote",
  "git-config": "Auto-detected from git config",
  manual: "Manually entered",
  localStorage: "Saved locally",
}

function sourceLabel(source: string) {
  if (source.startsWith("remote-")) return "Auto-detected from git remote"
  return SOURCE_LABELS[source] || "GitHub username"
}

function isHandle(username: string) {
  return /^[a-zA-Z0-9][a-zA-Z0-9-]*$/.test(username)
}

function rankColor(rank: number) {
  if (rank === 1) return "text-amber-500"
  if (rank <= 3) return "text-sky-500"
  if (rank <= 10) return "text-success"
  return "text-primary"
}

function ProfileDropdown({ identity, onChange }: { identity: Identity; onChange: (name: string) => void }) {
  const { username, source } = identity
  const [open, setOpen] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState("")
  const [stats, setStats] = useState<UserStats | null>(null)
  const [imgFailed, setImgFailed] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const handle = isHandle(username)

  const loadStats = () => {
    if (!username) return
    setStats(null)
    api.userStats(username).then(setStats).catch(() => setStats(null))
  }

  useEffect(() => {
    setImgFailed(false)
    loadStats()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [username])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
        setEditing(false)
      }
    }
    document.addEventListener("mousedown", onDown)
    return () => document.removeEventListener("mousedown", onDown)
  }, [open])

  const commit = () => {
    onChange(draft)
    setEditing(false)
    setOpen(false)
  }

  const startEdit = () => {
    setDraft(handle ? username : "")
    setEditing(true)
  }

  const viewScoreboard = () => {
    setOpen(false)
    navigate("#/")
    setTimeout(() => document.getElementById("leaderboard")?.scrollIntoView({ behavior: "smooth" }), 50)
  }

  const avatar =
    handle && !imgFailed ? (
      <img
        src={`https://github.com/${username}.png?size=64`}
        alt=""
        className="h-6 w-6 rounded-full bg-muted object-cover"
        onError={() => setImgFailed(true)}
      />
    ) : (
      <Github size={18} />
    )

  // No username yet: a plain button that opens the editor to set one.
  if (!username && !editing) {
    return (
      <button
        onClick={() => {
          setDraft("")
          setEditing(true)
          setOpen(true)
        }}
        className="flex items-center gap-2 rounded-md px-2 py-1 text-sm text-muted-foreground transition-colors hover:bg-muted"
      >
        <Github size={18} /> Set username
      </button>
    )
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 rounded-md px-2 py-1 text-sm transition-colors hover:bg-muted"
      >
        {avatar}
        <span className="max-w-[160px] truncate">{username || "Set username"}</span>
        <ChevronDown size={14} className="text-muted-foreground" />
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-72 rounded-lg border border-border bg-background p-1 shadow-lg">
          {editing ? (
            <div className="p-2">
              <label className="mb-1 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <Github size={13} /> GitHub username
              </label>
              <input
                autoFocus
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commit()
                  else if (e.key === "Escape") setEditing(false)
                }}
                placeholder="your GitHub username"
                className="h-8 w-full rounded-md border border-input bg-transparent px-2 text-sm outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
              <div className="mt-2 flex justify-end gap-1">
                <Button variant="ghost" size="sm" onClick={() => setEditing(false)}>
                  Cancel
                </Button>
                <Button size="sm" onClick={commit}>
                  Save
                </Button>
              </div>
            </div>
          ) : (
            <>
              <div className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-muted-foreground">
                <Github size={13} /> {sourceLabel(source)}
              </div>
              <div className="border-t border-border/60" />

              <div className="px-3 py-2 text-xs font-medium text-muted-foreground">Your Progress</div>
              <div className="grid grid-cols-3 gap-2 px-3 pb-2 text-center">
                <div>
                  <div className="text-[11px] text-muted-foreground">Attempted</div>
                  <div className="text-lg font-bold text-primary">{stats ? stats.attempted : "-"}</div>
                </div>
                <div>
                  <div className="text-[11px] text-muted-foreground">Completed</div>
                  <div className="text-lg font-bold text-success">{stats ? stats.completed : "-"}</div>
                </div>
                <div>
                  <div className="text-[11px] text-muted-foreground">Avg Score</div>
                  <div className="text-lg font-bold text-amber-500">{stats ? `${stats.avgScore}%` : "-"}</div>
                </div>
              </div>
              <div className="px-3 pb-3 text-center">
                <div className="text-[11px] text-muted-foreground">Main Scoreboard Rank</div>
                <div className={"font-bold " + (stats && stats.rank > 0 ? rankColor(stats.rank) : "text-muted-foreground")}>
                  {stats ? (stats.rank > 0 ? `#${stats.rank}` : "Unranked") : "-"}
                </div>
              </div>
              <div className="border-t border-border/60" />

              {handle && (
                <a
                  href={`https://github.com/${username}`}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors hover:bg-muted"
                >
                  <Github size={15} /> View GitHub Profile
                </a>
              )}
              <button
                onClick={viewScoreboard}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors hover:bg-muted"
              >
                <Trophy size={15} /> View Scoreboard
              </button>
              <button
                onClick={loadStats}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors hover:bg-muted"
              >
                <RotateCcw size={15} /> Refresh Progress
              </button>
              <button
                onClick={startEdit}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors hover:bg-muted"
              >
                <Pencil size={15} /> Change Username
              </button>
            </>
          )}
        </div>
      )}
    </div>
  )
}

type Route = { name: "home" } | { name: "challenge"; id: string }

function parseHash(): Route {
  const m = window.location.hash.match(/^#\/challenge\/(.+)$/)
  return m ? { name: "challenge", id: decodeURIComponent(m[1]) } : { name: "home" }
}

export function navigate(to: string) {
  window.location.hash = to
}

export default function App() {
  const [dark, setDark] = useDarkMode()
  const [identity, setUsername] = useIdentity()
  const [route, setRoute] = useState<Route>(parseHash)

  useEffect(() => {
    const onChange = () => setRoute(parseHash())
    window.addEventListener("hashchange", onChange)
    return () => window.removeEventListener("hashchange", onChange)
  }, [])

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-200">
      <header className="sticky top-0 z-10 border-b border-border/60 bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-[104rem] items-center gap-2 px-4 py-3">
          <button
            onClick={() => navigate("#/")}
            className="flex items-center gap-2 font-semibold"
          >
            <Terminal size={18} />
            ML Practice
          </button>
          <div className="ml-auto flex items-center gap-1">
            <ProfileDropdown identity={identity} onChange={setUsername} />
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setDark(!dark)}
              aria-label="Toggle theme"
            >
              {dark ? <Sun size={18} /> : <Moon size={18} />}
            </Button>
          </div>
        </div>
      </header>

      <main className="px-4 py-6">
        {route.name === "home" ? (
          <Home />
        ) : (
          <Challenge id={route.id} username={identity.username} />
        )}
      </main>
    </div>
  )
}
