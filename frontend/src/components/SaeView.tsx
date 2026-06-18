import { useEffect, useState } from 'react'
import { getSaeInfo, getSaeLabels, runIntervene, runSaeFeatures } from '../api'
import type { InterveneResponse, SaeFeaturesResponse, SaeInfo } from '../types'
import { cleanToken } from '../util'

export function SaeView({ prompt, model }: { prompt: string; model: string }) {
  const [info, setInfo] = useState<SaeInfo | null>(null)
  const [layer, setLayer] = useState(7)
  const [data, setData] = useState<SaeFeaturesResponse | null>(null)
  const [sel, setSel] = useState<number | null>(null)
  const [labels, setLabels] = useState<Record<string, string | null>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [selFeat, setSelFeat] = useState<number | null>(null)
  const [mode, setMode] = useState<'ablate' | 'steer'>('ablate')
  const [coeff, setCoeff] = useState(10)
  const [interv, setInterv] = useState<InterveneResponse | null>(null)
  const [intervLoading, setIntervLoading] = useState(false)

  useEffect(() => {
    getSaeInfo(model)
      .then((r) => {
        setInfo(r)
        if (r.n_layers) setLayer((l) => Math.min(l, r.n_layers! - 1))
      })
      .catch(() => setInfo({ available: false }))
  }, [model])

  useEffect(() => {
    if (!info?.available) return
    const q = prompt.trim()
    if (!q) return
    setLoading(true)
    setError(null)
    runSaeFeatures(q, layer, model)
      .then((r) => {
        setData(r)
        setSel(r.tokens.length - 1)
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [prompt, model, layer, info?.available])

  useEffect(() => {
    if (!data || sel == null) return
    const idxs = (data.features[sel] || []).map((f) => f.index)
    if (!idxs.length) return
    getSaeLabels(model, data.layer, idxs)
      .then((m) => setLabels((p) => ({ ...p, ...m })))
      .catch(() => undefined)
  }, [data, sel, model])

  useEffect(() => {
    setSelFeat(null)
    setInterv(null)
  }, [prompt, layer, model])

  useEffect(() => {
    if (selFeat == null || !info?.available) return
    const q = prompt.trim()
    if (!q) return
    let live = true
    setIntervLoading(true)
    const t = setTimeout(() => {
      runIntervene({ prompt: q, layer, feature: selFeat, mode, coeff, model })
        .then((r) => live && setInterv(r))
        .catch(() => live && setInterv(null))
        .finally(() => live && setIntervLoading(false))
    }, 220)
    return () => {
      live = false
      clearTimeout(t)
    }
  }, [selFeat, mode, coeff, prompt, layer, model, info?.available])

  if (!info) return null

  const nLayers = info.n_layers || 12
  const feats = data && sel != null ? data.features[sel] || [] : []
  const maxAct = feats.reduce((m, f) => Math.max(m, f.act), 0) || 1
  const maxDelta = interv ? interv.deltas.reduce((m, d) => Math.max(m, Math.abs(d.delta)), 0) || 1 : 1

  return (
    <section className="mod">
      <div className="mod-head">
        <div className="idx">03</div>
        <div className="ttl">SAE FEATURES</div>
        <div className="hmeta">
          <span className="lbl">
            {data ? (
              <>
                hook <b>{data.hook}</b> · {data.d_sae} feats
              </>
            ) : (
              'sparse autoencoder'
            )}
          </span>
        </div>
      </div>
      <div className="mod-body">
        {!info.available ? (
          <div className="note">
            No SAE is configured for <b>{model}</b>.
          </div>
        ) : (
          <>
            <div className="ctlbar" style={{ marginBottom: 16 }}>
              <span className="lbl">layer</span>
              <span className="stepper">
                <button onClick={() => setLayer((layer - 1 + nLayers) % nLayers)}>◀</button>
                <span className="v">L{String(layer).padStart(2, '0')}</span>
                <button onClick={() => setLayer((layer + 1) % nLayers)}>▶</button>
              </span>
              <span className="lbl" style={{ marginLeft: 'auto' }}>
                click a token, then a feature → ablate / steer it
              </span>
            </div>

            {info.pt_it_caveat && (
              <div className="caveat">
                ⚠ Gemma Scope SAEs were trained on the <b>base (pt)</b> model; this runs the{' '}
                <b>instruct (-it)</b> model. Labels are best-effort and layer-dependent — pick a
                layer where they read coherently.
              </div>
            )}

            {error && <div className="err">⚠ {error}</div>}
            {loading && !data && <div className="busy">[ ENCODING ]</div>}

            {data && (
              <>
                <div className="tokens">
                  {data.tokens.map((t, i) => (
                    <div key={i} className={`tk ${i === sel ? 'sel' : ''}`} onClick={() => setSel(i)}>
                      <span className="t">{cleanToken(t)}</span>
                      <span className="i">{(data.features[i]?.[0]?.act ?? 0).toFixed(1)}</span>
                    </div>
                  ))}
                </div>

                <div className="featlist">
                  <div className="lbl" style={{ marginBottom: 4 }}>
                    top features · token {sel != null ? `"${cleanToken(data.tokens[sel])}"` : '—'}
                  </div>
                  {feats.length === 0 && <div className="dim">no active features here</div>}
                  {feats.map((f) => (
                    <div
                      key={f.index}
                      className={`featrow click ${selFeat === f.index ? 'on' : ''}`}
                      onClick={() => setSelFeat(f.index)}
                    >
                      <span className="fid">f/{f.index}</span>
                      <span className="fbarwrap">
                        <span className="fbar" style={{ width: `${(f.act / maxAct) * 100}%` }} />
                      </span>
                      <span className="fact">{f.act.toFixed(2)}</span>
                      <span className="flabel">{labels[String(f.index)] ?? '…'}</span>
                    </div>
                  ))}
                </div>

                {selFeat != null && (
                  <div className="interv">
                    <div className="interv-head">
                      <span className="lbl">
                        intervene · <b>f/{selFeat}</b>
                        {labels[String(selFeat)] ? ` · ${labels[String(selFeat)]}` : ''}
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
                      <button className="interv-x" onClick={() => setSelFeat(null)} title="close">
                        ✕
                      </button>
                    </div>

                    {interv ? (
                      <div className="interv-body">
                        <div className="shifts">
                          <div className="lbl" style={{ marginBottom: 6 }}>
                            top logit shifts{intervLoading ? ' · …' : ''}
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
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>
    </section>
  )
}
