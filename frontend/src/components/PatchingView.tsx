import { useEffect, useState } from 'react'
import type { ChangeEvent } from 'react'
import { runPatching } from '../api'
import type { PatchInput } from '../api'
import type { PatchResponse } from '../types'
import { accMix, negMix, short } from '../util'

const DEF: PatchInput = {
  clean_prompt: 'The capital of France is',
  corrupted_prompt: 'The capital of England is',
  answer: ' Paris',
  corrupted_answer: ' London',
}

export function PatchingView() {
  const [form, setForm] = useState(DEF)
  const [res, setRes] = useState<PatchResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const run = (input: PatchInput) => {
    setLoading(true)
    setError(null)
    runPatching(input)
      .then((r) => {
        setRes(r)
        setLoading(false)
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e))
        setLoading(false)
      })
  }

  useEffect(() => {
    run(DEF)
  }, [])

  const set = (k: keyof PatchInput) => (e: ChangeEvent<HTMLInputElement>) =>
    setForm({ ...form, [k]: e.target.value })

  let heat = null
  let metrics = null
  if (res) {
    const sc = res.scores
    const nl = sc.length
    const np = sc[0].length
    const toks = res.tokens.map((t) => short(t, 7))
    let amax = 1e-6
    let best = -1e9
    let bl = 0
    let bp = 0
    sc.forEach((row, l) =>
      row.forEach((v, q) => {
        if (Math.abs(v) > amax) amax = Math.abs(v)
        if (v > best) {
          best = v
          bl = l
          bp = q
        }
      }),
    )
    const cell = 34
    const lab = 40
    const top = 104
    const W = lab + np * cell + 4
    const H = top + nl * cell + 4
    heat = (
      <div className="hmwrap">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet" style={{ maxHeight: 520 }}>
          {toks.map((t, q) => {
            const x = lab + q * cell + cell * 0.66
            return (
              <text key={`c${q}`} className="axt" fontSize={12} x={x} y={top - 6} transform={`rotate(-90 ${x} ${top - 6})`} textAnchor="start">
                {t}
              </text>
            )
          })}
          {Array.from({ length: nl }, (_, l) => l).map((l) => (
            <text key={`l${l}`} className="axt" fontSize={12} x={lab - 7} y={top + l * cell + cell * 0.66} textAnchor="end">
              L{l}
            </text>
          ))}
          {sc.map((row, l) =>
            row.map((v, q) => (
              <rect
                key={`${l}-${q}`}
                className="gl"
                x={lab + q * cell}
                y={top + l * cell}
                width={cell}
                height={cell}
                fill={v >= 0 ? accMix(v / amax) : negMix(-v / amax)}
              />
            )),
          )}
          <rect x={lab + bp * cell} y={top + bl * cell} width={cell} height={cell} fill="none" stroke="var(--strong)" strokeWidth={2.6} />
        </svg>
      </div>
    )
    metrics = (
      <div className="metrics">
        <div className="mcell">
          <div className="mk lbl">logit diff · clean</div>
          <div className="mv">
            {res.logit_diff_clean >= 0 ? '+' : ''}
            {res.logit_diff_clean.toFixed(2)}
          </div>
        </div>
        <div className="mcell">
          <div className="mk lbl">logit diff · corrupted</div>
          <div className="mv">
            {res.logit_diff_corrupted >= 0 ? '+' : ''}
            {res.logit_diff_corrupted.toFixed(2)}
          </div>
        </div>
        <div className="mcell hero-metric">
          <div className="mk lbl">max restoration</div>
          <div className="mv">{best.toFixed(2)}</div>
          <div className="lbl" style={{ marginTop: 10 }}>
            site → L{bl} · POS {bp} · "{res.tokens[bp]}"
          </div>
        </div>
      </div>
    )
  }

  return (
    <section className="mod">
      <div className="mod-head">
        <div className="idx">05</div>
        <div className="ttl">ACTIVATION PATCHING</div>
        <div className="hmeta">
          <span className="lbl">
            resid_pre · <b>causal sweep</b>
          </span>
        </div>
      </div>
      <div className="mod-body">
        <div className="patch-form">
          <label className="fld">
            <span className="lbl">clean prompt</span>
            <input value={form.clean_prompt} onChange={set('clean_prompt')} />
            <input style={{ maxWidth: 120 }} value={form.answer} onChange={set('answer')} />
          </label>
          <label className="fld">
            <span className="lbl">corrupted</span>
            <input value={form.corrupted_prompt} onChange={set('corrupted_prompt')} />
            <input style={{ maxWidth: 120 }} value={form.corrupted_answer} onChange={set('corrupted_answer')} />
          </label>
          <div className="fld">
            <span className="lbl" />
            <button className="run" style={{ height: 34, padding: '0 22px' }} onClick={() => run(form)} disabled={loading}>
              {loading ? 'PATCHING…' : 'RUN PATCH ▶'}
            </button>
          </div>
        </div>
        {error && <div className="err">⚠ {error}</div>}
        {!res && !error && <div className="busy">[ SWEEPING ]</div>}
        {res && (
          <div className="patch-grid">
            <div>{heat}</div>
            {metrics}
          </div>
        )}
      </div>
    </section>
  )
}
