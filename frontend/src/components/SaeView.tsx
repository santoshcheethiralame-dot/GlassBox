import { useEffect, useState } from 'react'
import { getSaeInfo, getSaeLabels, runSaeFeatures, runSaeTrack } from '../api'
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
  const [track, setTrack] = useState<number[] | null>(null)
  const [trackLabel, setTrackLabel] = useState<string | null>(null)
  const [search, setSearch] = useState('')

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

  useEffect(() => {
    if (selectedFeature == null || !data) {
      setTrack(null)
      setTrackLabel(null)
      return
    }
    let live = true
    runSaeTrack(prompt, data.layer, selectedFeature, model)
      .then((r) => {
        if (!live) return
        setTrack(r.acts)
        setTrackLabel(r.label)
      })
      .catch(() => {
        if (live) {
          setTrack(null)
          setTrackLabel(null)
        }
      })
    return () => {
      live = false
    }
  }, [selectedFeature, data, prompt, model])

  if (!info) return null

  const nLayers = info.n_layers || 12
  const feats = data && sel != null ? data.features[sel] || [] : []
  const maxAct = feats.reduce((m, f) => Math.max(m, f.act), 0) || 1
  const trackMax = track ? track.reduce((m, a) => Math.max(m, a), 0) || 1 : 1
  const tintable = !!(data && track && track.length === data.tokens.length && trackMax > 0)

  const npUrl = (idx: number) =>
    data?.np_model && data?.np_source
      ? `https://www.neuronpedia.org/${data.np_model}/${data.np_source}/${idx}`
      : null

  const inspect = () => {
    const idx = parseInt(search, 10)
    if (!Number.isNaN(idx) && data && idx >= 0 && idx < data.d_sae) {
      onSelect({ layer: data.layer, feature: idx, label: labels[String(idx)] ?? null })
    }
  }

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
              <span className="feat-search">
                <input
                  type="text"
                  inputMode="numeric"
                  spellCheck={false}
                  placeholder="feature #"
                  value={search}
                  onChange={(e) => setSearch(e.target.value.replace(/[^0-9]/g, ''))}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') inspect()
                  }}
                />
                <button onClick={inspect}>INSPECT</button>
              </span>
              <span className="lbl" style={{ marginLeft: 'auto' }}>
                click a token, then a feature → INTERVENE
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
                    <div
                      key={i}
                      className={`tk ${i === sel ? 'sel' : ''}`}
                      onClick={() => setSel(i)}
                      style={
                        tintable
                          ? {
                              boxShadow: `inset 0 -5px 0 0 rgba(46, 230, 166, ${(
                                track![i] / trackMax
                              ).toFixed(3)})`,
                            }
                          : undefined
                      }
                    >
                      <span className="t">{cleanToken(t)}</span>
                      <span className="i">{(data.features[i]?.[0]?.act ?? 0).toFixed(1)}</span>
                    </div>
                  ))}
                </div>

                {tintable && selectedFeature != null && (
                  <div
                    className="lbl"
                    style={{ marginTop: 8, textTransform: 'none', letterSpacing: '0.3px' }}
                  >
                    strip tinted by <b style={{ color: 'var(--acc)' }}>f/{selectedFeature}</b> activity
                    {trackLabel ? ` · ${trackLabel}` : ''}
                  </div>
                )}

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
                      {npUrl(f.index) && (
                        <a
                          className="np-link"
                          href={npUrl(f.index)!}
                          target="_blank"
                          rel="noreferrer"
                          onClick={(e) => e.stopPropagation()}
                          title="open this feature on Neuronpedia"
                        >
                          ↗
                        </a>
                      )}
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
