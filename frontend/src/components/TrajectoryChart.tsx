import { useState } from 'react'
import { interpolateTurbo } from 'd3-scale-chromatic'
import { cleanToken } from '../util'

interface Props {
  trajectories: number[][][]
  tokens: string[]
  layerLabels: string[]
  explainedVariance: number[]
  width?: number
  height?: number
}

export function TrajectoryChart({
  trajectories,
  tokens,
  layerLabels,
  explainedVariance,
  width = 660,
  height = 460,
}: Props) {
  const [hl, setHl] = useState<number | null>(null)
  const [hover, setHover] = useState<{ t: number; p: number; x: number; y: number } | null>(null)

  const pad = 30
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const tr of trajectories)
    for (const [x, y] of tr) {
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
  const sx = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (width - 2 * pad)
  const sy = (y: number) => pad + (1 - (y - minY) / (maxY - minY || 1)) * (height - 2 * pad)
  const color = (i: number) => interpolateTurbo((i + 0.6) / (tokens.length + 0.2))
  const npts = layerLabels.length

  return (
    <div style={{ position: 'relative' }}>
      <svg width={width} height={height} style={{ display: 'block', maxWidth: '100%' }}>
        {trajectories.map((tr, t) => {
          const dim = hl != null && hl !== t
          const c = color(t)
          const pts = tr.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')
          return (
            <g key={t} opacity={dim ? 0.12 : 1}>
              <polyline points={pts} fill="none" stroke={c} strokeWidth={hl === t ? 2.6 : 1.4} />
              {tr.map(([x, y], p) => (
                <circle
                  key={p}
                  cx={sx(x)}
                  cy={sy(y)}
                  r={p === 0 || p === npts - 1 ? 3.6 : 2}
                  fill={p === npts - 1 ? c : '#0a0b0f'}
                  stroke={c}
                  strokeWidth={1.2}
                  onMouseEnter={(e) => setHover({ t, p, x: e.clientX, y: e.clientY })}
                  onMouseMove={(e) => setHover({ t, p, x: e.clientX, y: e.clientY })}
                  onMouseLeave={() => setHover(null)}
                />
              ))}
              <text
                x={sx(tr[npts - 1][0]) + 6}
                y={sy(tr[npts - 1][1]) + 3}
                fontSize={11}
                fill={c}
                fontFamily="ui-monospace, monospace"
                opacity={dim ? 0.3 : 1}
              >
                {cleanToken(tokens[t])}
              </text>
            </g>
          )
        })}
      </svg>
      {hover && (
        <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {`${cleanToken(tokens[hover.t])} · ${layerLabels[hover.p]}`}
        </div>
      )}
      <div className="legend" style={{ marginTop: 8 }}>
        {tokens.map((tk, t) => (
          <span
            key={t}
            className="traj-chip"
            onMouseEnter={() => setHl(t)}
            onMouseLeave={() => setHl(null)}
            style={{ color: color(t) }}
          >
            <span className="swatch-dot" style={{ background: color(t) }} />
            {cleanToken(tk)}
          </span>
        ))}
      </div>
      <div className="legend">
        hollow = embedding · filled = final layer · PC1 {Math.round(explainedVariance[0] * 100)}% ·
        PC2 {Math.round(explainedVariance[1] * 100)}% of variance
      </div>
    </div>
  )
}
