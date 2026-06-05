import { useState } from 'react'
import { Heatmap } from './Heatmap'
import { cleanToken, clamp } from '../util'
import type { ForwardResponse } from '../types'

export function AttentionView({ data }: { data: ForwardResponse }) {
  const [layer, setLayer] = useState(0)
  const [head, setHead] = useState(0)

  const labels = data.tokens.map(cleanToken)
  const layerHeads = data.attention[layer]
  const pattern = layerHeads[head]
  const seq = labels.length

  const bigCell = clamp(Math.floor(440 / seq), 13, 30)
  const thumbCell = clamp(Math.floor(110 / seq), 3, 9)

  return (
    <div className="panel">
      <h2>Attention patterns</h2>

      <div className="selector">
        <span className="sel-label">LAYER</span>
        {data.attention.map((_, l) => (
          <button
            key={l}
            className={`chip ${l === layer ? 'active' : ''}`}
            onClick={() => setLayer(l)}
          >
            {l}
          </button>
        ))}
      </div>

      <div className="head-grid">
        {layerHeads.map((hp, h) => (
          <div
            key={h}
            className={`head-thumb ${h === head ? 'active' : ''}`}
            onClick={() => setHead(h)}
            title={`Layer ${layer}, Head ${h}`}
          >
            <Heatmap matrix={hp} cell={thumbCell} showLabels={false} maskZero />
            <span className="label">
              L{layer}H{h}
            </span>
          </div>
        ))}
      </div>

      <div className="hm-title">
        Head <b>L{layer}·H{head}</b> &mdash; each row (query) attends to columns (keys); GPT-2 is
        causal, so attention is lower-triangular.
      </div>
      <div className="heatmap-wrap">
        <Heatmap
          matrix={pattern}
          rowLabels={labels}
          colLabels={labels}
          cell={bigCell}
          maskZero
          tooltipLabel={(r, c, v) =>
            `${cleanToken(data.tokens[r])} → ${cleanToken(data.tokens[c])}\n${v.toFixed(3)}`
          }
        />
      </div>
    </div>
  )
}
