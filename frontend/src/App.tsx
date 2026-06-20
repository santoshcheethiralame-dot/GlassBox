import { useEffect, useRef, useState } from 'react'
import { Tokens } from './components/Tokens'
import { AttentionView } from './components/AttentionView'
import { ResidualView } from './components/ResidualView'
import { ProbeView } from './components/ProbeView'
import { PatchingView } from './components/PatchingView'
import { AttributionView } from './components/AttributionView'
import { ExperimentView } from './components/ExperimentView'
import { NeuronView } from './components/NeuronView'
import { SaeView } from './components/SaeView'
import { InterveneView, type InterveneTarget } from './components/InterveneView'
import { ACTS, SideNav, ActHeader } from './components/Acts'
import { Figure } from './components/Figure'
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
  const [target, setTarget] = useState<InterveneTarget | null>(null)
  const [activeAct, setActiveAct] = useState('observe')
  const [progress, setProgress] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runId = useRef(0)
  const didModelMount = useRef(false)

  const selectFeature = (t: InterveneTarget) => {
    setTarget(t)
    // bring INTERVENE into view so the cross-panel handoff is visible
    requestAnimationFrame(() => {
      const el = document.getElementById('intervene')
      if (!el) return
      const r = el.getBoundingClientRect()
      if (r.top > window.innerHeight - 120 || r.bottom < 120) {
        const smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches
        el.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' })
      }
    })
  }

  const run = async (p: string) => {
    const q = p.trim()
    if (!q) return
    const id = ++runId.current
    setApplied(q)
    setLoading(true)
    setError(null)
    setFocus(null)
    setTraj(null)
    setTarget(null)
    try {
      const [f, t] = await Promise.all([runForward(q, 10, model), runTrajectory(q, model)])
      if (id !== runId.current) return
      setFwd(f)
      setTraj(t)
    } catch (e) {
      if (id === runId.current) {
        setError(e instanceof Error ? e.message : String(e))
        setFwd(null)
        setTraj(null)
      }
    } finally {
      if (id === runId.current) setLoading(false)
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    setTarget(null)
    // re-run the active prompt against the newly selected model (skip the first mount,
    // which the initial-load effect already covers)
    if (!didModelMount.current) {
      didModelMount.current = true
      return
    }
    run(applied || prompt || DEFAULT_PROMPT)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model])

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    let raf = 0
    const onMove = (e: MouseEvent) => {
      if (raf) return
      raf = requestAnimationFrame(() => {
        raf = 0
        const root = document.documentElement.style
        root.setProperty('--mx', ((e.clientX / window.innerWidth) * 100).toFixed(1) + '%')
        root.setProperty('--my', ((e.clientY / window.innerHeight) * 100).toFixed(1) + '%')
      })
    }
    window.addEventListener('mousemove', onMove)
    return () => {
      window.removeEventListener('mousemove', onMove)
      if (raf) cancelAnimationFrame(raf)
    }
  }, [])

  useEffect(() => {
    let raf = 0
    const onScroll = () => {
      if (raf) return
      raf = requestAnimationFrame(() => {
        raf = 0
        const doc = document.documentElement
        const max = doc.scrollHeight - doc.clientHeight
        setProgress(max > 0 ? Math.min(1, Math.max(0, doc.scrollTop / max)) : 0)
        let cur = ACTS[0].id
        for (const a of ACTS) {
          const el = document.getElementById(a.id)
          if (el && el.getBoundingClientRect().top <= 160) cur = a.id
        }
        if (doc.scrollTop + doc.clientHeight >= doc.scrollHeight - 4) cur = ACTS[ACTS.length - 1].id
        setActiveAct(cur)
      })
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    onScroll()
    return () => {
      window.removeEventListener('scroll', onScroll)
      if (raf) cancelAnimationFrame(raf)
    }
  }, [])

  return (
    <div className="shell">
      <a href="#observe" className="skip">Skip to content</a>
      <header className="masthead">
        <div className="brand">
          <span className="mark">
            <svg viewBox="0 0 30 30" fill="none" aria-hidden="true">
              <rect x="3.5" y="3.5" width="10.5" height="10.5" rx="3" fill="rgba(255, 162, 77, 0.32)" stroke="var(--acc)" strokeWidth="1.5" />
              <rect x="16" y="3.5" width="10.5" height="10.5" rx="3" fill="rgba(255, 162, 77, 0.07)" stroke="var(--acc)" strokeWidth="1.5" />
              <rect x="3.5" y="16" width="10.5" height="10.5" rx="3" fill="rgba(255, 162, 77, 0.07)" stroke="var(--acc)" strokeWidth="1.5" />
              <rect x="16" y="16" width="10.5" height="10.5" rx="3" fill="rgba(255, 162, 77, 0.18)" stroke="var(--acc)" strokeWidth="1.5" />
              <path d="M5.5 6.5 L9 5" stroke="#ffffff" strokeWidth="1" strokeLinecap="round" opacity="0.55" />
            </svg>
          </span>
          <span className="nm">GLASSBOX</span>
          <span className="ver">v4</span>
        </div>
        <div className="prompt-zone">
          <span className="lbl tag">prompt /</span>
          <input
            className="prompt-in"
            aria-label="prompt"
            value={prompt}
            spellCheck={false}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') run(prompt)
            }}
          />
        </div>
        <button className="run" onClick={() => run(prompt)} disabled={loading || !prompt.trim()}>
          {loading ? 'RUNNING…' : 'RUN ▶'}
        </button>
        <div className="mdl">
          <span className="lbl">model</span>
          <select aria-label="model" value={model} onChange={(e) => setModel(e.target.value)}>
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

      <div className="body">
        <SideNav
          active={activeAct}
          progress={progress}
          tokens={fwd ? fwd.tokens.length : 0}
          model={model}
          prompt={applied}
          layers={health ? health.n_layers : 12}
          heads={health ? health.n_heads : 12}
          dmodel={health ? health.d_model : 768}
        />

        <main className="page">

          <section className="hero">
            <div className="hero-top">
              <div className="hero-tt">
                <span className="bq" />
                <h1>
                  INTERPRETABILITY ENGINE
                  <small>FEATURES · INTERVENTIONS · CAUSAL ATTRIBUTION</small>
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
              <div className="mod-body err-bar">
                <span>⚠ {error}</span>
                <button className="retry-btn" onClick={() => run(applied || prompt)} disabled={loading}>
                  {loading ? 'RETRYING…' : 'RETRY ▶'}
                </button>
              </div>
            </div>
          )}

          <ActHeader act={ACTS[0]} />
          {fwd && (
            <Figure name="01-tokenization">
              <Tokens
                tokens={fwd.tokens}
                preds={fwd.top_predictions}
                focus={focus}
                onFocus={setFocus}
              />
            </Figure>
          )}
          {fwd && (
            <Figure name="02-attention">
              <AttentionView data={fwd} focus={focus} onFocus={setFocus} />
            </Figure>
          )}
          <div className="split">
            <Figure name="03-residual">
              <ResidualView traj={traj} focus={focus} error={error} />
            </Figure>
            <Figure name="04-probes">
              <ProbeView model={model} />
            </Figure>
          </div>

          <ActHeader act={ACTS[1]} />
          <Figure name="05-sae-features">
            <SaeView
              prompt={applied}
              model={model}
              selectedFeature={target?.feature ?? null}
              onSelect={selectFeature}
            />
          </Figure>
          <Figure name="06-neurons">
            <NeuronView model={model} nLayers={health ? health.n_layers : 12} />
          </Figure>

          <ActHeader act={ACTS[2]} />
          <Figure name="07-intervene">
            <InterveneView target={target} prompt={applied} model={model} />
          </Figure>

          <ActHeader act={ACTS[3]} />
          <Figure name="08-patching">
            <PatchingView model={model} />
          </Figure>
          <Figure name="09-attribution">
            <AttributionView model={model} />
          </Figure>

          <ActHeader act={ACTS[4]} />
          <Figure name="10-hallucination-lab">
            <ExperimentView model={model} />
          </Figure>

          <footer className="colophon">
            <span>GLASSBOX // INTERPRETABILITY ENGINE</span>
            <span>TRANSFORMERLENS · FASTAPI</span>
          </footer>
        </main>
      </div>
    </div>
  )
}
