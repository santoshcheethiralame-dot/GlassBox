import type { TokenPrediction } from '../types'
import { cleanToken } from '../util'

export function Predictions({ preds }: { preds: TokenPrediction[] }) {
  const max = preds.length ? preds[0].prob : 1
  return (
    <div className="panel">
      <h2>Next-token predictions</h2>
      {preds.map((p, i) => (
        <div className="pred" key={i}>
          <span className="tok">{cleanToken(p.token)}</span>
          <div className="bar" style={{ width: `${Math.max(2, (p.prob / max) * 300)}px` }} />
          <span className="pct">{(p.prob * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}
