import { useMemo, useState } from 'react'
import { cleanToken, short } from '../util'
import type { ForwardResponse } from '../types'

export function AttentionView({
  data,
  focus,
  onFocus,
}: {
  data: ForwardResponse
  focus: number | null
  onFocus: (i: number | null) => void
}) {
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)
  const [hover, setHover] = useState<{ r: number; c: number; x: number; y: number } | null>(null)

  const n = data.tokens.length
  const labels = data.tokens.map((t) => short(t, 7))
  const cell = 30
  const lab = 104
  const W = lab + n * cell + 4

  const heads = useMemo(
    () => (
      <div className="heads">
        {data.attention[layer].map((hp, h) => (
          <div
            key={h}
            className={`head ${h === head ? 'sel' : ''}`}
            onClick={() => setHead(h)}
            title={`L${layer} H${h}`}
          >
            <svg viewBox={`0 0 ${n} ${n}`} width="100%">
              {hp.map((row, r) =>
                row.map((v, c) =>
                  c <= r ? (
                    <rect key={`${r}-${c}`} x={c} y={r} width={1} height={1} fill="var(--acc)" fillOpacity={Math.max(0.05, v ** 0.7)} />
                  ) : null,
                ),
              )}
            </svg>
            <span className="hl">H{String(h).padStart(2, '0')}</span>
          </div>
        ))}
      </div>
    ),
    [data, layer, head, n],
  )

  const m = data.attention[layer][head]

  return (
    <section className="mod" onMouseLeave={() => setHover(null)}>
      <div className="mod-head">
        <div className="idx">02</div>
        <div className="ttl">ATTENTION</div>
        <div className="hmeta">
          <span className="lbl">
            site <b>blocks.{layer}.attn.hook_pattern</b>
          </span>
        </div>
      </div>
      <div className="mod-body">
        <div className="attn-wrap">
          <div className="attn-main">
            <div className="ctlbar">
              <span className="lbl">layer</span>
              <span className="stepper">
                <button onClick={() => setLayer((layer - 1 + data.n_layers) % data.n_layers)}>◀</button>
                <span className="v">L{String(layer).padStart(2, '0')}</span>
                <button onClick={() => setLayer((layer + 1) % data.n_layers)}>▶</button>
              </span>
              <span className="lbl">head</span>
              <span className="stepper">
                <button onClick={() => setHead((head - 1 + data.n_heads) % data.n_heads)}>◀</button>
                <span className="v">H{String(head).padStart(2, '0')}</span>
                <button onClick={() => setHead((head + 1) % data.n_heads)}>▶</button>
              </span>
              <span className="lbl" style={{ marginLeft: 'auto' }}>
                rows attend to cols · causal
              </span>
            </div>
            <div className="hmwrap">
              <svg
                viewBox={`0 0 ${W} ${W}`}
                width="100%"
                preserveAspectRatio="xMidYMid meet"
                style={{ maxHeight: 560 }}
              >
                {labels.map((t, c) => {
                  const x = lab + c * cell + cell * 0.66
                  return (
                    <text
                      key={`c${c}`}
                      className="axt"
                      fontSize={12}
                      x={x}
                      y={lab - 6}
                      transform={`rotate(-90 ${x} ${lab - 6})`}
                      textAnchor="start"
                    >
                      {t}
                    </text>
                  )
                })}
                {labels.map((t, r) => (
                  <text
                    key={`r${r}`}
                    className="axt"
                    fontSize={12}
                    x={lab - 7}
                    y={lab + r * cell + cell * 0.66}
                    textAnchor="end"
                  >
                    {t}
                  </text>
                ))}
                {m.map((row, r) =>
                  row.map((v, c) => (
                    <rect
                      key={`${r}-${c}`}
                      className="gl"
                      x={lab + c * cell}
                      y={lab + r * cell}
                      width={cell}
                      height={cell}
                      fill={c > r ? '#000000' : 'var(--acc)'}
                      fillOpacity={c > r ? 0.22 : Math.max(0.04, v ** 0.7)}
                      onMouseMove={(e) => setHover({ r, c, x: e.clientX, y: e.clientY })}
                      onClick={() => onFocus(focus === c ? null : c)}
                    />
                  )),
                )}
                {focus != null && focus < n && (
                  <>
                    <rect x={lab} y={lab + focus * cell} width={n * cell} height={cell} fill="none" stroke="var(--acc)" strokeWidth={2.5} />
                    <rect x={lab + focus * cell} y={lab} width={cell} height={n * cell} fill="none" stroke="var(--acc)" strokeWidth={2.5} />
                  </>
                )}
              </svg>
            </div>
          </div>
          <div className="attn-side">
            <div className="lbl" style={{ marginBottom: 10 }}>
              all heads · layer {String(layer).padStart(2, '0')}
            </div>
            {heads}
          </div>
        </div>
      </div>
      {hover && (
        <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {`${cleanToken(data.tokens[hover.r])} → ${cleanToken(data.tokens[hover.c])}\n${m[hover.r][hover.c].toFixed(3)}`}
        </div>
      )}
    </section>
  )
}
