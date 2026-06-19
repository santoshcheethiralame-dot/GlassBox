import { useState } from 'react'
import { runExperiment } from '../api'
import type { ExperimentResponse } from '../types'
import { cleanToken } from '../util'

export function ExperimentView({ model }: { model: string }) {
  const [data, setData] = useState<ExperimentResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = () => {
    setLoading(true)
    setError(null)
    runExperiment(model, 7)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }

  const maxLd = data ? Math.max(...data.rows.map((r) => Math.abs(r.logit_diff)), 1) : 1

  return (
    <section className="mod adv">
      <div className="mod-head">
        <div className="idx">10</div>
        <div className="ttl">HALLUCINATION LAB</div>
        <div className="hmeta">
          <span className="lbl">
            knowledge conflict · <b>ground vs confabulate</b>
          </span>
        </div>
      </div>
      <div className="mod-body">
        <div className="note">
          Each context states a <b>false</b> fact. Does the model follow the context (<b>grounded</b>)
          or fall back on what it memorised (<b>confabulated</b>)? And can <b>ablating one feature</b>{' '}
          suppress a confabulation? — Phases 1–3 combined.
        </div>

        <button
          className="run"
          style={{ height: 36, padding: '0 24px' }}
          onClick={run}
          disabled={loading}
        >
          {loading ? 'RUNNING BATCH…' : 'RUN EXPERIMENT ▶'}
        </button>

        {error && <div className="err">⚠ {error}</div>}
        {loading && !data && <div className="busy">[ RUNNING THE BATCH ]</div>}
        {!data && !loading && !error && (
          <div className="interv-empty">◦ idle — press RUN to score a batch of conflict prompts (~10s)</div>
        )}

        {data && (
          <>
            <div className="exp-summary">
              <span className="exp-stat g">
                <b>{data.n_grounded}</b> grounded
              </span>
              <span className="exp-stat c">
                <b>{data.n_confabulated}</b> confabulated
              </span>
              <span className="lbl">
                of {data.n_total} · {data.model} · layer {data.layer}
              </span>
            </div>

            <div className="exp-rows">
              {data.rows.map((r, i) => {
                const w = (Math.abs(r.logit_diff) / maxLd) * 50
                const pos = r.logit_diff >= 0
                return (
                  <div key={i} className="exp-row">
                    <span className="exp-subj">
                      {r.subject} <b>{cleanToken(r.predicted)}</b>
                    </span>
                    <span className={`exp-badge ${r.label}`}>{r.label}</span>
                    <span className="exp-bar">
                      <span className="exp-axis" />
                      <span
                        className={`exp-fill ${pos ? 'g' : 'c'}`}
                        style={{ width: `${w}%`, left: pos ? '50%' : `${50 - w}%` }}
                      />
                    </span>
                    <span className={`exp-ld ${pos ? 'g' : 'c'}`}>
                      {pos ? '+' : ''}
                      {r.logit_diff.toFixed(2)}
                    </span>
                  </div>
                )
              })}
            </div>
            <div className="lbl" style={{ marginTop: 8, textTransform: 'none', letterSpacing: '0.3px' }}>
              bar = logit-diff (grounded − parametric): <span style={{ color: 'var(--acc)' }}>right/amber</span>{' '}
              follows context · <span style={{ color: 'var(--cool)' }}>left/slate</span> confabulates
            </div>

            {data.flip && (
              <div className="flip-card">
                <div className="lbl" style={{ marginBottom: 10 }}>
                  the money figure · feature ablation suppresses a confabulation
                </div>
                <div className="flip-head">
                  <b>{data.flip.subject}</b> — the model says "{cleanToken(data.flip.parametric)}"
                  (memorised) over "{cleanToken(data.flip.grounded)}" (context).
                </div>
                <div className="flip-action">
                  ablate <b>f/{data.flip.feature}</b>
                  {data.flip.feature_label ? ` · ${data.flip.feature_label}` : ''}
                </div>
                <div className="flip-bars">
                  <span className="lbl">grounded − parametric</span>
                  <span className="flip-val c">
                    {data.flip.ld_before >= 0 ? '+' : ''}
                    {data.flip.ld_before.toFixed(2)}
                  </span>
                  <span className="flip-arrow">▶</span>
                  <span className={`flip-val ${data.flip.ld_after >= 0 ? 'g' : 'c'}`}>
                    {data.flip.ld_after >= 0 ? '+' : ''}
                    {data.flip.ld_after.toFixed(2)}
                  </span>
                  <span className="flip-shift">
                    {data.flip.shift >= 0 ? '+' : ''}
                    {data.flip.shift.toFixed(2)} toward grounded
                  </span>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  )
}
