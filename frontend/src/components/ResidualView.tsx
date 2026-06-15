import { useState } from 'react'
import { cleanToken, short } from '../util'
import type { TrajectoryResponse } from '../types'

/* Strict monochromatic warm palette (reds, corals, warm greys) */
const PALETTE = [
  '#ff3b30', // acc red
  '#ff7866', // light red
  '#d9695b', // soft red
  '#b3261d', // deep red
  '#ff9980', // coral
  '#d6d3c6', // text (warm grey)
  '#e08c82', // dusty rose
  '#a34c44', // brick
  '#ffb3a6', // peach
  '#8a7b75', // brown grey
]

function tokenColor(index: number, _total: number): string {
  return PALETTE[index % PALETTE.length]
}

export function ResidualView({ traj, focus }: { traj: TrajectoryResponse | null; focus: number | null }) {
  const [hover, setHover] = useState<{ t: number; p: number; x: number; y: number } | null>(null)

  let body
  if (!traj) {
    body = <div className="busy">[ COMPUTING ]</div>
  } else {
    const tr = traj.trajectories
    const nTokens = tr.length
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
    const W = 460
    const H = 340
    const pad = 32
    const sx = (x: number) => pad + ((x - mnx) / (mxx - mnx || 1)) * (W - 2 * pad)
    const sy = (y: number) => H - pad - ((y - mny) / (mxy - mny || 1)) * (H - 2 * pad)

    /* Subtle grid lines for the axes area */
    const nGrid = 5
    const gridLines = []
    for (let i = 0; i <= nGrid; i++) {
      const xp = pad + (i / nGrid) * (W - 2 * pad)
      const yp = pad + (i / nGrid) * (H - 2 * pad)
      gridLines.push(
        <line key={`gx${i}`} x1={xp} y1={pad} x2={xp} y2={H - pad} stroke="var(--grid)" strokeWidth={1} />,
        <line key={`gy${i}`} x1={pad} y1={yp} x2={W - pad} y2={yp} stroke="var(--grid)" strokeWidth={1} />,
      )
    }

    body = (
      <div className="hmwrap" onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet" style={{ maxHeight: 380 }}>
          <defs>
            {/* Glow filter for focused trajectory */}
            <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            {/* Per-token gradient defs for polylines */}
            {tr.map((_, t) => {
              const col = tokenColor(t, nTokens)
              return (
                <linearGradient key={`grad${t}`} id={`tgrad${t}`} x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor={col} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={col} stopOpacity={1} />
                </linearGradient>
              )
            })}
          </defs>

          {/* Grid */}
          {gridLines}

          {/* Axes */}
          <line className="ax" strokeWidth={1.5} x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} />
          <line className="ax" strokeWidth={1.5} x1={pad} y1={pad} x2={pad} y2={H - pad} />
          <text className="axl" fontSize={10} x={W - pad} y={H - pad + 16} textAnchor="end">
            PC1 · {Math.round(traj.explained_variance[0] * 100)}%
          </text>
          <text className="axl" fontSize={10} x={pad - 2} y={pad - 6}>
            PC2 · {Math.round(traj.explained_variance[1] * 100)}%
          </text>

          {/* Trajectories — unfocused first, focused on top */}
          {tr.map((p, t) => {
            const foc = focus === t
            if (foc) return null // draw focused last
            const col = tokenColor(t, nTokens)
            const dim = focus != null
            const op = dim ? 0.12 : 0.55
            return (
              <g key={t}>
                <polyline
                  points={p.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')}
                  fill="none"
                  stroke={col}
                  strokeWidth={1}
                  opacity={op}
                  strokeLinejoin="round"
                />
                {p.map(([x, y], i) => {
                  const last = i === p.length - 1
                  const r = last ? 3.5 : 2
                  return (
                    <circle
                      key={i}
                      cx={sx(x)}
                      cy={sy(y)}
                      r={r}
                      fill={col}
                      opacity={dim ? 0.18 : 0.7}
                      onMouseMove={(e) => setHover({ t, p: i, x: e.clientX, y: e.clientY })}
                      style={{ cursor: 'crosshair' }}
                    />
                  )
                })}
              </g>
            )
          })}

          {/* Focused trajectory — drawn on top with glow */}
          {focus != null && tr[focus] && (() => {
            const t = focus
            const p = tr[t]
            const col = tokenColor(t, nTokens)
            return (
              <g filter="url(#glow)">
                <polyline
                  points={p.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')}
                  fill="none"
                  stroke={col}
                  strokeWidth={2.8}
                  opacity={1}
                  strokeLinejoin="round"
                />
                {p.map(([x, y], i) => {
                  const last = i === p.length - 1
                  const first = i === 0
                  const r = last ? 5 : first ? 4 : 2.6
                  return (
                    <g key={i}>
                      {/* Outer ring for start/end */}
                      {(first || last) && (
                        <circle
                          cx={sx(x)}
                          cy={sy(y)}
                          r={r + 3}
                          fill="none"
                          stroke={col}
                          strokeWidth={1.2}
                          opacity={0.4}
                        />
                      )}
                      <circle
                        cx={sx(x)}
                        cy={sy(y)}
                        r={r}
                        fill={last ? col : '#0a0b0f'}
                        stroke={col}
                        strokeWidth={last ? 0 : 1.6}
                        onMouseMove={(e) => setHover({ t, p: i, x: e.clientX, y: e.clientY })}
                        style={{ cursor: 'crosshair' }}
                      />
                      {/* Layer number labels along the path */}
                      {(first || last || i % 3 === 0) && (
                        <text
                          fontSize={8}
                          fill={col}
                          x={sx(x) + (last ? 7 : -7)}
                          y={sy(y) - 7}
                          textAnchor={last ? 'start' : 'end'}
                          fontFamily="var(--mono)"
                          fontWeight={700}
                          opacity={0.7}
                        >
                          {traj.layer_labels[i]}
                        </text>
                      )}
                    </g>
                  )
                })}
                {/* Token label at the end */}
                <text
                  className="axt"
                  fontSize={12}
                  fontWeight={800}
                  x={sx(p[p.length - 1][0]) + 10}
                  y={sx(p[p.length - 1][1]) > W / 2 ? sy(p[p.length - 1][1]) - 10 : sy(p[p.length - 1][1]) + 16}
                  fill={col}
                  style={{ letterSpacing: '0.5px' }}
                >
                  {short(traj.tokens[t], 10)}
                </text>
              </g>
            )
          })()}
        </svg>
        {hover && (
          <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
            <span style={{ color: tokenColor(hover.t, nTokens), fontWeight: 800 }}>
              {cleanToken(traj.tokens[hover.t])}
            </span>
            {' · '}
            {traj.layer_labels[hover.p]}
          </div>
        )}

        {/* Token legend strip */}
        <div className="traj-legend">
          {traj.tokens.map((tok, t) => (
            <span
              key={t}
              className={`traj-chip${focus === t ? ' sel' : ''}`}
              style={{
                borderColor: tokenColor(t, nTokens),
                color: focus === t ? '#0a0b0f' : tokenColor(t, nTokens),
                background: focus === t ? tokenColor(t, nTokens) : 'transparent',
              }}
            >
              {short(tok, 6)}
            </span>
          ))}
        </div>
      </div>
    )
  }

  return (
    <section className="mod">
      <div className="mod-head">
        <div className="idx">04</div>
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
