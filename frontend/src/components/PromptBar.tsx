import { useState } from 'react'

const EXAMPLES = [
  'The Eiffel Tower is in',
  'When John and Mary went to the store, John gave a drink to',
  'The capital of France is',
]

export function PromptBar({
  onRun,
  loading,
}: {
  onRun: (p: string) => void
  loading: boolean
}) {
  const [text, setText] = useState(EXAMPLES[0])
  const run = () => {
    if (text.trim()) onRun(text)
  }
  return (
    <div className="panel">
      <h2>Prompt</h2>
      <div className="prompt-row">
        <textarea
          className="prompt-input"
          rows={2}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run()
          }}
          placeholder="Enter a prompt…"
        />
        <button className="btn" onClick={run} disabled={loading}>
          {loading ? 'Running…' : 'Run ▸'}
        </button>
      </div>
      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            className="btn-ghost"
            onClick={() => {
              setText(ex)
              onRun(ex)
            }}
          >
            {ex.length > 44 ? ex.slice(0, 44) + '…' : ex}
          </button>
        ))}
      </div>
    </div>
  )
}
