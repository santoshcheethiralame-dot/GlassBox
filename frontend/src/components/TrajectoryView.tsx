import { useState } from 'react'
import { TrajectoryChart } from './TrajectoryChart'
import { runTrajectory } from '../api'
import type { TrajectoryResponse } from '../types'

const DEFAULT = 'The cat sat on the mat'

export function TrajectoryView() {
  const [text, setText] = useState(DEFAULT)
  const [res, setRes] = useState<TrajectoryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      setRes(await runTrajectory(text))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="panel">
      <h2>
        Residual-stream trajectory <span className="dim">— PCA of each token across layers</span>
      </h2>
      <p className="hint">
        Each token's residual vector is taken at every layer, L2-normalised (to drop the trivial
        norm growth that otherwise dominates), and projected to 2D with PCA. Every path traces how
        one token's representation moves through the model — from its embedding (hollow dot) to its
        final-layer state (filled dot). Hover a token below to highlight its path.
      </p>
      <div className="prompt-row">
        <input
          className="prompt-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') run()
          }}
          placeholder="Enter a prompt…"
        />
        <button className="btn" onClick={run} disabled={loading}>
          {loading ? 'Projecting…' : 'Plot ▸'}
        </button>
      </div>
      {error && <div className="error">⚠ {error}</div>}
      {res && (
        <div className="heatmap-wrap" style={{ marginTop: 14 }}>
          <TrajectoryChart
            trajectories={res.trajectories}
            tokens={res.tokens}
            layerLabels={res.layer_labels}
            explainedVariance={res.explained_variance}
          />
        </div>
      )}
    </div>
  )
}
