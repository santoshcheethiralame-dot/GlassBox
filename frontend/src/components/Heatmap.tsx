import { useMemo, useState } from 'react'
import { scaleSequential } from 'd3-scale'
import { interpolateViridis } from 'd3-scale-chromatic'

const DIM = '#7c8392'
const ACCENT = '#a3e635'
const EMPTY = '#0c0d11'

export interface HeatmapProps {
  matrix: number[][]
  rowLabels?: string[]
  colLabels?: string[]
  cell?: number
  maxValue?: number
  domain?: [number, number]
  interpolator?: (t: number) => string
  maskZero?: boolean
  showLabels?: boolean
  tooltipLabel?: (r: number, c: number, v: number) => string
}

export function Heatmap({
  matrix,
  rowLabels,
  colLabels,
  cell = 24,
  maxValue,
  domain,
  interpolator = interpolateViridis,
  maskZero = false,
  showLabels = true,
  tooltipLabel,
}: HeatmapProps) {
  const rows = matrix.length
  const cols = rows ? matrix[0].length : 0
  const [hover, setHover] = useState<{ r: number; c: number; x: number; y: number } | null>(null)

  const color = useMemo(() => {
    let dom = domain
    if (!dom) {
      let m = maxValue ?? 0
      if (maxValue == null) for (const row of matrix) for (const v of row) if (v > m) m = v
      dom = [0, m || 1]
    }
    return scaleSequential(interpolator).domain(dom).clamp(true)
  }, [matrix, maxValue, domain, interpolator])

  const labelW = showLabels && rowLabels ? 84 : 4
  const labelH = showLabels && colLabels ? 84 : 4
  const width = labelW + cols * cell + 4
  const height = labelH + rows * cell + 4

  return (
    <div style={{ position: 'relative' }}>
      <svg width={width} height={height} style={{ display: 'block' }}>
        {showLabels &&
          colLabels?.map((c, i) => {
            const cx = labelW + i * cell + cell / 2
            return (
              <text
                key={`c${i}`}
                x={cx}
                y={labelH - 5}
                transform={`rotate(-90 ${cx} ${labelH - 5})`}
                fontSize={11}
                fill={DIM}
                fontFamily="ui-monospace, monospace"
                textAnchor="start"
                dominantBaseline="middle"
              >
                {c}
              </text>
            )
          })}
        {showLabels &&
          rowLabels?.map((r, i) => (
            <text
              key={`r${i}`}
              x={labelW - 6}
              y={labelH + i * cell + cell / 2}
              fontSize={11}
              fill={DIM}
              fontFamily="ui-monospace, monospace"
              textAnchor="end"
              dominantBaseline="middle"
            >
              {r}
            </text>
          ))}
        {matrix.map((row, r) =>
          row.map((v, c) => {
            const active = hover?.r === r && hover?.c === c
            return (
              <rect
                key={`${r}-${c}`}
                x={labelW + c * cell}
                y={labelH + r * cell}
                width={cell - 1}
                height={cell - 1}
                fill={maskZero && v <= 0 ? EMPTY : color(v)}
                stroke={active ? ACCENT : 'none'}
                strokeWidth={active ? 1.5 : 0}
                onMouseEnter={(e) => setHover({ r, c, x: e.clientX, y: e.clientY })}
                onMouseMove={(e) => setHover({ r, c, x: e.clientX, y: e.clientY })}
                onMouseLeave={() => setHover(null)}
              />
            )
          }),
        )}
      </svg>
      {hover && (
        <div className="tooltip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {tooltipLabel
            ? tooltipLabel(hover.r, hover.c, matrix[hover.r][hover.c])
            : matrix[hover.r][hover.c].toFixed(3)}
        </div>
      )}
    </div>
  )
}
