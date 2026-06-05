import { useState } from 'react'
import { neuronDetail, scanLayer } from '../api'
import { cleanToken } from '../util'
import type { NeuronContext, NeuronDetail, NeuronScanResponse } from '../types'

function ContextLine({ ctx, compact }: { ctx: NeuronContext; compact?: boolean }) {
  const max = Math.max(ctx.max_act, 1e-6)
  return (
    <div className={`ctx-line ${compact ? 'compact' : ''}`}>
      {!compact && <span className="ctx-act">{ctx.max_act.toFixed(2)}</span>}
      <span className="ctx-tokens">
        {ctx.tokens.map((t, i) => {
          const a = Math.max(0, ctx.acts[i]) / max
          return (
            <span
              key={i}
              className={`ctx-tok ${i === ctx.max_pos ? 'mx' : ''}`}
              style={{ background: `rgba(163, 230, 53, ${(a * 0.8).toFixed(3)})` }}
            >
              {cleanToken(t)}
            </span>
          )
        })}
      </span>
    </div>
  )
}

export function NeuronView() {
  const [layer, setLayer] = useState(0)
  const [scan, setScan] = useState<NeuronScanResponse | null>(null)
  const [detail, setDetail] = useState<NeuronDetail | null>(null)
  const [manual, setManual] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const doScan = async () => {
    setLoading(true)
    setError(null)
    setDetail(null)
    try {
      setScan(await scanLayer(layer))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const doDetail = async (index: number) => {
    setLoading(true)
    setError(null)
    try {
      setDetail(await neuronDetail(layer, index))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="panel">
      <h2>
        Neuron analysis <span className="dim">— what makes each MLP neuron fire?</span>
      </h2>
      <p className="hint">
        Each MLP layer has 3072 neurons. A corpus of short sentences is run through the model and, for
        a chosen neuron, the token contexts where its post-GELU activation is highest are collected —
        its <b>maximally-activating examples</b>. Scan a layer to surface its most selective neurons,
        or inspect any neuron by index. Brighter green = stronger activation.
      </p>

      <div className="selector">
        <span className="sel-label">LAYER</span>
        {Array.from({ length: 12 }, (_, l) => l).map((l) => (
          <button
            key={l}
            className={`chip ${l === layer ? 'active' : ''}`}
            onClick={() => setLayer(l)}
          >
            {l}
          </button>
        ))}
        <button className="btn" onClick={doScan} disabled={loading}>
          {loading ? 'Scanning…' : 'Scan layer ▸'}
        </button>
      </div>

      <div className="selector">
        <span className="sel-label">NEURON</span>
        <input
          className="neuron-input"
          value={manual}
          onChange={(e) => setManual(e.target.value)}
          placeholder="0–3071"
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              const i = parseInt(manual, 10)
              if (!Number.isNaN(i)) doDetail(i)
            }
          }}
        />
        <button
          className="btn-ghost"
          onClick={() => {
            const i = parseInt(manual, 10)
            if (!Number.isNaN(i)) doDetail(i)
          }}
        >
          Inspect ▸
        </button>
      </div>

      {error && <div className="error">⚠ {error}</div>}

      {scan && !detail && (
        <div className="neuron-grid">
          {scan.neurons.map((n) => (
            <button key={n.index} className="neuron-card" onClick={() => doDetail(n.index)}>
              <div className="neuron-card-head">
                L{scan.layer}·N{n.index} <span className="dim">max {n.max_act.toFixed(2)}</span>
              </div>
              <ContextLine ctx={n.contexts[0]} compact />
            </button>
          ))}
        </div>
      )}

      {detail && (
        <div>
          <div className="neuron-detail-head">
            <b>
              L{detail.layer}·N{detail.index}
            </b>
            <span className="dim">
              max activation {detail.max_act.toFixed(2)} · fires on{' '}
              {(detail.density * 100).toFixed(1)}% of tokens
            </span>
            <button className="btn-ghost back" onClick={() => setDetail(null)}>
              ← back
            </button>
          </div>
          <div className="contexts">
            {detail.contexts.map((ctx, i) => (
              <ContextLine key={i} ctx={ctx} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
