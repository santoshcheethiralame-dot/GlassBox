import { useState } from 'react'
import { cleanToken, short } from '../util'
import type { TrajectoryResponse } from '../types'

export function ResidualView({ traj, focus }: { traj: TrajectoryResponse | null; focus: number | null }) {
  const [hover, setHover] = useState<{ t: number; p: number; x: number; y: number } | null>(null)

  let body
  if (!traj) {
    body = <div className="busy">[ COMPUTING ]</div>
  } else {
    const tr = traj.trajectories
    let mnx = 1e9
    let mxx = -1e9
    let mny = 1e9
    let mxy = -1e9
    tr.forEach((p) =>
      p.forEach(([x, y]) => {
        if (x < mnx) mnx = x
        if (x > mxx) mxx = x
        if (y < mny) mny = y
        if (y > mxy) mxy = y
      }),
    )
    const W = 420
    const H = 300
    const pad = 24
    const sx = (x: number) => pad + ((x - mnx) / (mxx - mnx || 1)) * (W - 2 * pad)
    const sy = (y: number) => H - pad - ((y - mny) / (mxy - mny || 1)) * (H - 2 * pad)
    body = (
      <div className="hmwrap" onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet" style={{ maxHeight: 340 }}>
          <line className="ax" strokeWidth={1.5} x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} />
          <line className="ax" strokeWidth={1.5} x1={pad} y1={pad} x2={pad} y2={H - pad} />
          <text className="axl" fontSize={10} x={W - pad} y={H - pad + 14} textAnchor="end">
            PC1 · {Math.round(traj.explained_variance[0] * 100)}%
          </text>
          <text className="axl" fontSize={10} x={pad - 2} y={pad - 4}>
            PC2 · {Math.round(traj.explained_variance[1] * 100)}%
          </text>
          {tr.map((p, t) => {
            const foc = focus === t
            const col = foc ? 'var(--acc)' : 'var(--text)'
            const op = foc || focus == null ? 1 : 0.16
            return (
              <g key={t}>
                <polyline
                  points={p.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')}
                  fill="none"
                  stroke={col}
                  strokeWidth={foc ? 2.4 : 0.9}
                  opacity={op}
                />
                {p.map(([x, y], i) => {
                  const last = i === p.length - 1
                  const z = last ? 4 : 2.4
                  return (
                    <rect
                      key={i}
                      x={sx(x) - z / 2}
                      y={sy(y) - z / 2}
                      width={z}
                      height={z}
                      fill={col}
                      opacity={op}
                      onMouseMove={(e) => setHover({ t, p: i, x: e.clientX, y: e.clientY })}
                    />
                  )
                })}
                {foc && (
                  <text className="axt" fontSize={11} x={sx(p[p.length - 1][0]) + 6} y={sy(p[p.length - 1][1]) + 3} fill="var(--acc)">
                    {short(traj.tokens[t], 8)}
                  </text>
                )}
              </g>
            )
          })}
        </svg>
        {hover && (
          <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
            {`${cleanToken(traj.tokens[hover.t])} · ${traj.layer_labels[hover.p]}`}
          </div>
        )}
      </div>
    )
  }

  return (
    <section className="mod">
      <div className="mod-head">
        <div className="idx">03</div>
        <div className="ttl">RESIDUAL</div>
        <div className="hmeta">
          <span className="lbl">
            pca · <b>l2-norm</b>
          </span>
        </div>
      </div>
      <div className="mod-body">
        <div className="lbl" style={{ marginBottom: 10 }}>
          per-token path · embed → final
        </div>
        {body}
      </div>
    </section>
  )
}
