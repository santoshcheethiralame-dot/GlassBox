import { useEffect, useState } from 'react'
import { getSaeInfo, getSaeLabels, runSaeFeatures } from '../api'
import type { SaeFeaturesResponse, SaeInfo } from '../types'
import { cleanToken } from '../util'
import type { InterveneTarget } from './InterveneView'

export function SaeView({
  prompt,
  model,
  selectedFeature,
  onSelect,
}: {
  prompt: string
  model: string
  selectedFeature: number | null
  onSelect: (t: InterveneTarget) => void
}) {
  const [info, setInfo] = useState<SaeInfo | null>(null)
  const [layer, setLayer] = useState(7)
  const [data, setData] = useState<SaeFeaturesResponse | null>(null)
  const [sel, setSel] = useState<number | null>(null)
  const [labels, setLabels] = useState<Record<string, string | null>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prompt, model, layer, info?.available])

  useEffect(() => {
    if (!data || sel == null) return
    const idxs = (data.features[sel] || []).map((f) => f.index)
    if (!idxs.length) return
    getSaeLabels(model, data.layer, idxs)
      .then((m) => setLabels((p) => ({ ...p, ...m })))
      .catch(() => undefined)
  }, [data, sel, model])

  if (!info) return null

  const nLayers = info.n_layers || 12
  const feats = data && sel != null ? data.features[sel] || [] : []
  const maxAct = feats.reduce((m, f) => Math.max(m, f.act), 0) || 1

  return (
    <section className="mod adv">
      <div className="mod-head">
        <div className="idx">05</div>
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
                click a token, then a feature → send to INTERVENE
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
                      className={`featrow click ${selectedFeature === f.index ? 'on' : ''}`}
                      onClick={() =>
                        onSelect({
                          layer: data.layer,
                          feature: f.index,
                          label: labels[String(f.index)] ?? null,
                        })
                      }
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
              </>
            )}
          </>
        )}
      </div>
    </section>
  )
}
