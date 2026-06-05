import { useState } from 'react'

export interface LineSeries {
  key: string
  label: string
  color: string
  values: number[]
}

const M = { top: 18, right: 18, bottom: 40, left: 46 }

export function LineChart({
  series,
  nx,
  baseline,
  width = 680,
  height = 340,
}: {
  series: LineSeries[]
  nx: number
  baseline?: number
  width?: number
  height?: number
}) {
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null)
  const iw = width - M.left - M.right
  const ih = height - M.top - M.bottom
  const px = (i: number) => M.left + (nx <= 1 ? 0 : (i / (nx - 1)) * iw)
  const py = (v: number) => M.top + (1 - v) * ih
  const yticks = [0, 0.25, 0.5, 0.75, 1]
  const colW = nx <= 1 ? iw : iw / (nx - 1)

  return (
    <div style={{ position: 'relative' }}>
      <svg width={width} height={height} style={{ display: 'block', maxWidth: '100%' }}>
        {yticks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={width - M.right} y1={py(t)} y2={py(t)} stroke="#1f2430" />
            <text
              x={M.left - 8}
              y={py(t)}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize={11}
              fill="#7c8392"
              fontFamily="ui-monospace, monospace"
            >
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {Array.from({ length: nx }, (_, i) => i).map((i) => (
          <text
            key={i}
            x={px(i)}
            y={height - M.bottom + 18}
            textAnchor="middle"
            fontSize={11}
            fill="#7c8392"
            fontFamily="ui-monospace, monospace"
          >
            {i}
          </text>
        ))}
        <text x={M.left + iw / 2} y={height - 6} textAnchor="middle" fontSize={11} fill="#7c8392">
          layer
        </text>
        {baseline != null && (
          <line
            x1={M.left}
            x2={width - M.right}
            y1={py(baseline)}
            y2={py(baseline)}
            stroke="#4b5161"
            strokeDasharray="5 4"
          />
        )}
        {hover && (
          <line x1={px(hover.i)} x2={px(hover.i)} y1={M.top} y2={M.top + ih} stroke="#3a4150" />
        )}
        {series.map((s) => (
          <g key={s.key}>
            <polyline
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              points={s.values.map((v, i) => `${px(i)},${py(v)}`).join(' ')}
            />
            {s.values.map((v, i) => (
              <circle key={i} cx={px(i)} cy={py(v)} r={hover?.i === i ? 4 : 2.4} fill={s.color} />
            ))}
          </g>
        ))}
        {Array.from({ length: nx }, (_, i) => i).map((i) => (
          <rect
            key={i}
            x={px(i) - colW / 2}
            y={M.top}
            width={colW}
            height={ih}
            fill="transparent"
            onMouseEnter={(e) => setHover({ i, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => setHover({ i, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setHover(null)}
          />
        ))}
      </svg>
      {hover && (
        <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {`layer ${hover.i}` +
            series.map((s) => `\n${s.label}  ${Math.round(s.values[hover.i] * 100)}%`).join('')}
        </div>
      )}
    </div>
  )
}
