import { useState } from 'react'
import type { ChangeEvent } from 'react'
import { interpolateRdBu } from 'd3-scale-chromatic'
import { Heatmap } from './Heatmap'
import { cleanToken, clamp } from '../util'
import { runPatching } from '../api'
import type { PatchResponse } from '../types'

const DEFAULTS = {
  clean_prompt: 'The capital of France is',
  corrupted_prompt: 'The capital of England is',
  answer: ' Paris',
  corrupted_answer: ' London',
}

export function PatchingView() {
  const [form, setForm] = useState(DEFAULTS)
  const [res, setRes] = useState<PatchResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const set =
    (k: keyof typeof form) =>
    (e: ChangeEvent<HTMLInputElement>) =>
      setForm({ ...form, [k]: e.target.value })

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      setRes(await runPatching(form))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const flat = res ? res.scores.flat() : []
  const bound = flat.length ? Math.max(0.001, ...flat.map((v) => Math.abs(v))) : 1
  const colLabels = res ? res.tokens.map(cleanToken) : []
  const rowLabels = res ? res.scores.map((_, l) => `L${l}`) : []

  return (
    <div className="panel">
      <h2>
        Activation patching <span className="dim">— causal tracing</span>
      </h2>
      <p className="hint">
        Patch each (layer, position) of the residual stream from the <b>clean</b> run into the{' '}
        <b>corrupted</b> run, and measure how much it restores the clean answer.{' '}
        <b style={{ color: '#d6604d' }}>1.0</b> = full restoration, <b>0</b> = no effect. The bright
        cells are the causal sites — where & when the model holds the answer.
      </p>

      <div className="patch-form">
        <label className="fld">
          <span>Clean prompt</span>
          <input value={form.clean_prompt} onChange={set('clean_prompt')} />
        </label>
        <label className="fld narrow">
          <span>predicts</span>
          <input value={form.answer} onChange={set('answer')} />
        </label>
        <label className="fld">
          <span>Corrupted prompt</span>
          <input value={form.corrupted_prompt} onChange={set('corrupted_prompt')} />
        </label>
        <label className="fld narrow">
          <span>predicts</span>
          <input value={form.corrupted_answer} onChange={set('corrupted_answer')} />
        </label>
        <button className="btn" onClick={run} disabled={loading}>
          {loading ? 'Patching…' : 'Run patch ▸'}
        </button>
      </div>

      {error && <div className="error">⚠ {error}</div>}

      {res && (
        <>
          <div className="patch-stats">
            logit diff = <b>{cleanToken(res.answer)}</b> − <b>{cleanToken(res.corrupted_answer)}</b>
            &nbsp;·&nbsp; clean <b>{res.logit_diff_clean.toFixed(2)}</b> &nbsp;·&nbsp; corrupted{' '}
            <b>{res.logit_diff_corrupted.toFixed(2)}</b>
          </div>
          <div className="heatmap-wrap">
            <Heatmap
              matrix={res.scores}
              rowLabels={rowLabels}
              colLabels={colLabels}
              cell={clamp(Math.floor(540 / res.seq), 16, 42)}
              interpolator={interpolateRdBu}
              domain={[bound, -bound]}
              tooltipLabel={(r, c, v) =>
                `L${r} · pos ${c} (${cleanToken(res.tokens[c])})\nrestoration ${v.toFixed(3)}`
              }
            />
          </div>
          <div className="legend">
            <span className="swatch" style={{ background: interpolateRdBu(0.04) }} /> restores clean
            <span className="swatch" style={{ background: interpolateRdBu(0.5) }} /> no effect
            <span className="swatch" style={{ background: interpolateRdBu(0.96) }} /> reinforces
            corrupted
          </div>
        </>
      )}
    </div>
  )
}
