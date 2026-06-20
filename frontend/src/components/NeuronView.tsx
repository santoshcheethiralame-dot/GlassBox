import { useEffect, useState } from 'react'
import { scanLayer } from '../api'
import type { NeuronScanResponse } from '../types'
import { cleanToken, clickable } from '../util'

export function NeuronView({ model, nLayers }: { model: string; nLayers: number }) {
  const [layer, setLayer] = useState(0)
  const [res, setRes] = useState<NeuronScanResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [sel, setSel] = useState<number | null>(null)

  const scan = (l: number) => {
    setLoading(true)
    setError(null)
    setSel(null)
    scanLayer(l, model)
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
    setLayer(0)
    scan(0)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model])

  const step = (d: number) => {
    const l = Math.max(0, Math.min(nLayers - 1, layer + d))
    if (l === layer) return
    setLayer(l)
    scan(l)
  }

  return (
    <section className="mod">
      <div className="mod-head">
        <div className="idx">06</div>
        <div className="ttl">NEURONS</div>
        <div className="hmeta">
          <span className="lbl">
            mlp_post · <b>{res ? `${res.n_sentences} seq · ${res.d_mlp} / layer` : '…'}</b>
          </span>
          <span className="stepper">
            <button aria-label="previous layer" disabled={layer === 0} onClick={() => step(-1)}>◀</button>
            <span className="v">L{String(layer).padStart(2, '0')}</span>
            <button aria-label="next layer" disabled={layer === nLayers - 1} onClick={() => step(1)}>▶</button>
          </span>
        </div>
      </div>
      <div className="mod-body">
        {error && <div className="err">⚠ {error}</div>}
        {loading && <div className="busy">[ SCANNING L{String(layer).padStart(2, '0')} ]</div>}
        {res && !loading && (
          <div className="ngrid">
            {res.neurons.map((nn, i) => (
              <div
                key={nn.index}
                className={`ncard ${sel === i ? 'sel' : ''}`}
                {...clickable(() => setSel(sel === i ? null : i))}
              >
                <div className="nch">
                  <span className="nid">
                    L{res.layer}.N{nn.index}
                  </span>
                  <span className="nact">{nn.max_act.toFixed(2)}</span>
                </div>
                <div className="ndesc lbl">density {(nn.density * 100).toFixed(1)}%</div>
                <div className="nctx">
                  {nn.contexts.map((c, k) => (
                    <div key={k} className="snip">
                      <span className="sv">{c.max_act.toFixed(2)}</span>
                      {c.tokens.map((t, j) =>
                        j === c.max_pos ? (
                          <span key={j} className="mxt">
                            {cleanToken(t)}
                          </span>
                        ) : (
                          <span key={j}>{cleanToken(t)}</span>
                        ),
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
