import { useEffect, useState } from 'react'
import { runProbes } from '../api'
import type { ProbeResponse } from '../types'
import { clickable } from '../util'

function stepPoints(acc: number[], px: (i: number) => number, py: (v: number) => number): string {
  const pts: string[] = []
  acc.forEach((v, j) => {
    pts.push(`${px(j)},${py(v)}`)
    if (j < acc.length - 1) pts.push(`${px(j + 1)},${py(v)}`)
  })
  return pts.join(' ')
}

export function ProbeView({ model }: { model: string }) {
  const [res, setRes] = useState<ProbeResponse | null>(null)
  const [off, setOff] = useState<Set<string>>(new Set())
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    setRes(null)
    setError(null)
    runProbes([], model)
      .then((r) => live && setRes(r))
      .catch((e) => live && setError(e instanceof Error ? e.message : String(e)))
    return () => {
      live = false
    }
  }, [model])

  const toggle = (k: string) => {
    const next = new Set(off)
    if (next.has(k)) next.delete(k)
    else next.add(k)
    setOff(next)
  }

  let body
  if (error) body = <div className="err">⚠ {error}</div>
  else if (!res) body = <div className="busy">[ TRAINING PROBES ]</div>
  else {
    const n = res.n_layers
    const W = 440
    const H = 300
    const L = 34
    const B = 28
    const T = 10
    const R = 12
    const px = (i: number) => L + (i / (n - 1)) * (W - L - R)
    const py = (v: number) => T + (1 - v) * (H - T - B)
    const styleOf = (i: number) => (i === 1 ? '6 4' : i === 2 ? '1.5 3' : '')
    body = (
      <>
        <div className="legend">
          {res.results.map((r, i) => {
            const key = r.kind === 'contextual'
            return (
              <span
                key={r.key}
                className={`lt ${off.has(r.key) ? 'off' : ''} ${key ? 'key' : ''}`}
                aria-pressed={!off.has(r.key)}
                {...clickable(() => toggle(r.key))}
              >
                <span
                  className="sw"
                  style={{ borderTopStyle: i === 1 ? 'dashed' : i === 2 ? 'dotted' : 'solid' }}
                />
                {r.key.toUpperCase()} <span className="lbl">· {r.kind}</span>
              </span>
            )
          })}
        </div>
        <div className="hmwrap">
          <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet" style={{ maxHeight: 330 }}>
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <g key={v}>
                <line className="gl" x1={L} y1={py(v)} x2={W - R} y2={py(v)} />
                <text className="axl" fontSize={10} x={L - 5} y={py(v) + 3} textAnchor="end">
                  {v.toFixed(2)}
                </text>
              </g>
            ))}
            <line className="ax" strokeWidth={1.5} x1={L} y1={T} x2={L} y2={H - B} />
            <line className="ax" strokeWidth={1.5} x1={L} y1={H - B} x2={W - R} y2={H - B} />
            {Array.from({ length: n }, (_, i) => i).map((i) => (
              <text key={i} className="axl" fontSize={10} x={px(i)} y={H - B + 15} textAnchor="middle">
                {i}
              </text>
            ))}
            <line x1={L} y1={py(0.5)} x2={W - R} y2={py(0.5)} stroke="var(--dim)" strokeWidth={1} strokeDasharray="4 3" />
            {res.results.map((r, i) => {
              if (off.has(r.key)) return null
              const key = r.kind === 'contextual'
              const col = key ? 'var(--acc)' : 'var(--text)'
              return (
                <g key={r.key}>
                  <polyline
                    points={stepPoints(r.test_acc, px, py)}
                    fill="none"
                    stroke={col}
                    strokeWidth={key ? 2.6 : 1.5}
                    strokeDasharray={styleOf(i)}
                  />
                  {key &&
                    r.test_acc.map((v, j) => (
                      <rect key={j} x={px(j) - 2} y={py(v) - 2} width={4} height={4} fill={col} />
                    ))}
                </g>
              )
            })}
          </svg>
        </div>
      </>
    )
  }

  return (
    <section className="mod">
      <div className="mod-head">
        <div className="idx">04</div>
        <div className="ttl">PROBES</div>
        <div className="hmeta">
          <span className="lbl">
            logreg · <b>acc / layer</b>
          </span>
        </div>
      </div>
      <div className="mod-body">{body}</div>
    </section>
  )
}
