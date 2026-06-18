import { useEffect, useState } from 'react'
import { Tokens } from './components/Tokens'
import { AttentionView } from './components/AttentionView'
import { ResidualView } from './components/ResidualView'
import { ProbeView } from './components/ProbeView'
import { PatchingView } from './components/PatchingView'
import { AttributionView } from './components/AttributionView'
import { NeuronView } from './components/NeuronView'
import { SaeView } from './components/SaeView'
import { getHealth, getModels, runForward, runTrajectory } from './api'
import type { ForwardResponse, HealthResponse, ModelInfo, TrajectoryResponse } from './types'

const DEFAULT_PROMPT = 'When Mary and John went to the store, John gave a drink to'

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [model, setModel] = useState('gpt2')
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT)
  const [applied, setApplied] = useState(DEFAULT_PROMPT)
  const [fwd, setFwd] = useState<ForwardResponse | null>(null)
  const [traj, setTraj] = useState<TrajectoryResponse | null>(null)
  const [focus, setFocus] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async (p: string) => {
    const q = p.trim()
    if (!q) return
    setApplied(q)
    setLoading(true)
    setError(null)
    setFocus(null)
    setTraj(null)
    try {
      const [f, t] = await Promise.all([runForward(q), runTrajectory(q)])
      setFwd(f)
      setTraj(t)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
    getModels()
      .then(setModels)
      .catch(() => setModels([]))
    run(DEFAULT_PROMPT)
  }, [])

  return (
    <div className="shell">
      <header className="masthead">
        <div className="brand">
          <span className="mark">
            <i />
            <i />
            <i />
            <i />
          </span>
          <span className="nm">GLASSBOX</span>
        </div>
        <div className="prompt-zone">
          <span className="lbl tag">prompt /</span>
          <input
            className="prompt-in"
            value={prompt}
            spellCheck={false}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') run(prompt)
            }}
          />
        </div>
        <button className="run" onClick={() => run(prompt)} disabled={loading}>
          {loading ? 'RUNNING…' : 'RUN ▶'}
        </button>
        <div className="mdl">
          <span className="lbl">model</span>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {(models.length ? models : [{ key: 'gpt2', name: 'gpt2', device: 'cpu' }]).map((m) => (
              <option key={m.key} value={m.key}>
                {m.key}
              </option>
            ))}
          </select>
        </div>
        <div className="status">
          {health ? (
            <>
              <span className="live">
                <span className="dotlive" />
                LIVE
              </span>
              <span className="meta">
                {health.model.toUpperCase()} · {health.n_layers}L · {health.n_heads}H · D
                {health.d_model}
              </span>
            </>
          ) : (
            <>
              <span className="live off">
                <span className="dotlive" />
                OFFLINE
              </span>
              <span className="meta">BACKEND :8000</span>
            </>
          )}
        </div>
      </header>

      <main className="page">
        <span className="crop tl" />
        <span className="crop tr" />
        <span className="crop bl" />
        <span className="crop br" />

        <section className="hero">
          <div className="hero-top">
            <div className="hero-tt">
              <span className="bq" />
              <h1>
                INTERPRETABILITY ENGINE
                <small>MECHANISTIC ANALYSIS OF GPT-2 SMALL</small>
              </h1>
            </div>
            <div className="echo">
              <div className="lbl">active prompt · {fwd ? fwd.tokens.length : '—'} tokens</div>
              <div className="q">"{prompt}"</div>
            </div>
          </div>
          <div className="stat-row">
            <div className="stat">
              <div className="v">{health ? health.n_layers : 12}</div>
              <div className="k lbl">layers</div>
            </div>
            <div className="stat">
              <div className="v">{health ? health.n_heads : 12}</div>
              <div className="k lbl">heads / layer</div>
            </div>
            <div className="stat accent">
              <div className="v">{health ? health.d_model : 768}</div>
              <div className="k lbl">d_model</div>
            </div>
            <div className="stat">
              <div className="v">3072</div>
              <div className="k lbl">d_mlp</div>
            </div>
            <div className="stat">
              <div className="v">50K</div>
              <div className="k lbl">vocab</div>
            </div>
          </div>
        </section>

        {error && (
          <div className="mod">
            <div className="mod-body err">⚠ {error}</div>
          </div>
        )}

        {fwd && (
          <Tokens tokens={fwd.tokens} preds={fwd.top_predictions} focus={focus} onFocus={setFocus} />
        )}
        {fwd && <AttentionView data={fwd} focus={focus} onFocus={setFocus} />}

        <SaeView prompt={applied} model={model} />

        <div className="split">
          <ResidualView traj={traj} focus={focus} />
          <ProbeView />
        </div>

        <PatchingView />
        <NeuronView />
        <AttributionView model={model} />

        <footer className="colophon">
          <span>GLASSBOX // GPT-2 INTERPRETABILITY ENGINE</span>
          <span>TRANSFORMERLENS · FASTAPI</span>
        </footer>
      </main>
    </div>
  )
}
