import { useEffect, useState } from 'react'
import { runIntervene } from '../api'
import type { InterveneResponse } from '../types'
import { cleanToken } from '../util'

export type InterveneTarget = { layer: number; feature: number; label: string | null }

export function InterveneView({
  target,
  prompt,
  model,
}: {
  target: InterveneTarget | null
  prompt: string
  model: string
}) {
  const [mode, setMode] = useState<'ablate' | 'steer'>('ablate')
  const [coeff, setCoeff] = useState(10)
  const [interv, setInterv] = useState<InterveneResponse | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setInterv(null)
  }, [target?.feature, target?.layer])

  useEffect(() => {
    if (!target) return
    const q = prompt.trim()
    if (!q) return
    let live = true
    setLoading(true)
    const t = setTimeout(() => {
      runIntervene({ prompt: q, layer: target.layer, feature: target.feature, mode, coeff, model })
        .then((r) => live && setInterv(r))
        .catch(() => live && setInterv(null))
        .finally(() => live && setLoading(false))
    }, 220)
    return () => {
      live = false
      clearTimeout(t)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target?.feature, target?.layer, mode, coeff, prompt, model])

  const maxDelta = interv ? interv.deltas.reduce((m, d) => Math.max(m, Math.abs(d.delta)), 0) || 1 : 1

  return (
    <section className="mod">
      <div className="mod-head">
        <div className="idx">07</div>
        <div className="ttl">INTERVENE</div>
        <div className="hmeta">
          <span className="lbl">ablate · steer · live logits</span>
        </div>
      </div>
      <div className="mod-body">
        <div className="note">
          Remove a feature from the residual stream (<b>ablate</b>) or amplify it along its decoder
          direction (<b>steer</b>), and watch the next-token distribution move. Click a feature in{' '}
          <b>DECODE</b> above to load it here.
        </div>

        {!target ? (
          <div className="interv-empty">
            ◦ no feature selected — pick one from <b>SAE FEATURES</b> in DECODE
          </div>
        ) : (
          <>
            <div className="interv-head">
              <span className="lbl">
                <b>f/{target.feature}</b>
                {target.label ? ` · ${target.label}` : ''} · L
                {String(target.layer).padStart(2, '0')}
              </span>
              <span className="seg">
                <button className={mode === 'ablate' ? 'on' : ''} onClick={() => setMode('ablate')}>
                  ABLATE
                </button>
                <button className={mode === 'steer' ? 'on' : ''} onClick={() => setMode('steer')}>
                  STEER
                </button>
              </span>
              {mode === 'steer' && (
                <span className="coeffctl">
                  <input
                    type="range"
                    aria-label="steer coefficient"
                    min={-20}
                    max={40}
                    step={1}
                    value={coeff}
                    onChange={(e) => setCoeff(Number(e.target.value))}
                  />
                  <span className="cval">
                    {coeff > 0 ? '+' : ''}
                    {coeff}
                  </span>
                </span>
              )}
            </div>

            {interv ? (
              <div className="interv-body">
                <div className="shifts">
                  <div className="lbl" style={{ marginBottom: 6 }}>
                    top logit shifts{loading ? ' · …' : ''}
                  </div>
                  {interv.deltas.slice(0, 8).map((d, i) => (
                    <div key={i} className="shift">
                      <span className="st">{cleanToken(d.token)}</span>
                      <span className="sbarwrap">
                        <span
                          className={`sbar ${d.delta >= 0 ? 'up' : 'down'}`}
                          style={{ width: `${(Math.abs(d.delta) / maxDelta) * 100}%` }}
                        />
                      </span>
                      <span className={`sd ${d.delta >= 0 ? 'up' : 'down'}`}>
                        {d.delta >= 0 ? '+' : ''}
                        {d.delta.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="cols">
                  <div className="col">
                    <div className="lbl">baseline</div>
                    {interv.baseline.slice(0, 6).map((p, i) => (
                      <div key={i} className="prow">
                        <span className="pt">{cleanToken(p.token)}</span>
                        <span className="pl">{p.logit.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                  <div className="col">
                    <div className="lbl">{mode === 'ablate' ? 'ablated' : 'steered'}</div>
                    {interv.intervened.slice(0, 6).map((p, i) => (
                      <div key={i} className="prow">
                        <span className="pt">{cleanToken(p.token)}</span>
                        <span className="pl">{p.logit.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="busy">[ RUNNING ]</div>
            )}
          </>
        )}
      </div>
    </section>
  )
}
