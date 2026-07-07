import { useEffect, useRef, useState } from "react"
import AceEditor from "react-ace"
import type { Ace } from "ace-builds"
import "ace-builds/src-noconflict/mode-python"
import "ace-builds/src-noconflict/theme-chrome"
import "ace-builds/src-noconflict/theme-one_dark"
import "ace-builds/src-noconflict/ext-searchbox" // Ctrl+F find/replace box

interface CodeEditorProps {
  /** Unique DOM id for this Ace instance. */
  name: string
  value: string
  onChange?: (value: string) => void
  /** CSS height (e.g. "560px" or "calc(100vh - 16rem)"). */
  height?: string
  readOnly?: boolean
  showStatus?: boolean
}

/** Tracks whether the `dark` class is on <html> so Ace can switch themes. */
function useIsDark() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"))
  useEffect(() => {
    const observer = new MutationObserver(() =>
      setDark(document.documentElement.classList.contains("dark")),
    )
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })
    return () => observer.disconnect()
  }, [])
  return dark
}

/**
 * Ace-based code editor (same engine the go-interview project uses): line
 * numbers, code folding (fold widgets for functions/imports/classes), syntax
 * highlighting, a compact line-height, and an Ln/Col status bar.
 */
export function CodeEditor({
  name,
  value,
  onChange,
  height = "560px",
  readOnly = false,
  showStatus = true,
}: CodeEditorProps) {
  const dark = useIsDark()
  const [pos, setPos] = useState({ ln: 1, col: 1 })
  const editorRef = useRef<Ace.Editor | null>(null)

  const onLoad = (editor: Ace.Editor) => {
    editorRef.current = editor
    editor.selection.on("changeCursor", () => {
      const c = editor.getCursorPosition()
      setPos({ ln: c.row + 1, col: c.column + 1 })
    })
  }

  // Keep Ace sized correctly when the height (focus mode) changes.
  useEffect(() => {
    editorRef.current?.resize()
  }, [height])

  return (
    <div className="flex flex-col">
      <AceEditor
        name={name}
        onLoad={onLoad}
        mode="python"
        theme={dark ? "one_dark" : "chrome"}
        value={value}
        onChange={onChange}
        readOnly={readOnly}
        width="100%"
        height={height}
        fontSize={14}
        showPrintMargin={false}
        highlightActiveLine={!readOnly}
        setOptions={{
          useWorker: false,
          showFoldWidgets: true,
          fadeFoldWidgets: false,
          showLineNumbers: true,
          tabSize: 4,
          useSoftTabs: true,
          wrap: false,
        }}
        editorProps={{ $blockScrolling: true }}
      />
      {showStatus && (
        <div className="flex justify-end border-t border-border/60 px-3 py-1 font-mono text-[11px] text-muted-foreground">
          Ln {pos.ln}, Col {pos.col}
        </div>
      )}
    </div>
  )
}
