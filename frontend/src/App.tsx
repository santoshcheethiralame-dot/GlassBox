import { useEffect, useState } from 'react'
import { PromptBar } from './components/PromptBar'
import { Tokens } from './components/Tokens'
import { Predictions } from './components/Predictions'
import { AttentionView } from './components/AttentionView'
import { PatchingView } from './components/PatchingView'
import { getHealth, runForward } from './api'
import type { ForwardResponse, HealthResponse } from './types'

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [data, setData] = useState<ForwardResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  const run = async (prompt: string) => {
    setLoading(true)
    setError(null)
    try {
      setData(await runForward(prompt))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>GlassBox</h1>
        <span className="subtitle">peering inside GPT-2 small</span>
        {health ? (
          <span className="badge">
            {health.model} · {health.n_layers}L × {health.n_heads}H · d{health.d_model}
          </span>
        ) : (
          <span className="badge offline">backend offline</span>
        )}
      </header>

      <PromptBar onRun={run} loading={loading} />

      {error && <div className="panel error">⚠ {error}</div>}

      {data ? (
        <>
          <Tokens tokens={data.tokens} />
          <Predictions preds={data.top_predictions} />
          <AttentionView data={data} />
        </>
      ) : (
        !error && (
          <div className="panel placeholder">
            Enter a prompt and hit <b>Run</b> to see GPT-2's tokenization, next-token predictions,
            and per-head attention patterns.
          </div>
        )
      )}

      <PatchingView />
    </div>
  )
}
