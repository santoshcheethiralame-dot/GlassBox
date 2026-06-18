export type Act = { id: string; n: string; name: string; sub: string }

export const ACTS: Act[] = [
  { id: 'observe', n: '01', name: 'OBSERVE', sub: 'what the model sees' },
  { id: 'decode', n: '02', name: 'DECODE', sub: 'features in the stream' },
  { id: 'intervene', n: '03', name: 'INTERVENE', sub: 'ablate · steer' },
  { id: 'causal', n: '04', name: 'CAUSAL', sub: 'what caused the answer' },
  { id: 'experiment', n: '05', name: 'EXPERIMENT', sub: 'hallucination lab' },
]

export function SideNav({
  active,
  progress,
  tokens,
  model,
}: {
  active: string
  progress: number
  tokens: number
  model: string
}) {
  const ai = ACTS.findIndex((a) => a.id === active)
  return (
    <nav className="sidenav">
      <div className="sn-top">
        <span className="sn-eyebrow">the glass box</span>
        <span className="sn-title">ACTS</span>
      </div>
      <div className="sn-tube">
        <span className="sn-rail" />
        <span className="sn-liquid" style={{ height: `${Math.max(1.5, progress * 100).toFixed(1)}%` }} />
        <div className="sn-items">
          {ACTS.map((a, i) => (
            <a
              key={a.id}
              href={`#${a.id}`}
              className={`sn-item${active === a.id ? ' on' : ''}${i <= ai ? ' lit' : ''}`}
            >
              <span className="sn-node" />
              <span className="sn-n">{a.n}</span>
              <span className="sn-name">{a.name}</span>
              <span className="sn-sub">{a.sub}</span>
            </a>
          ))}
        </div>
      </div>
      <div className="sn-foot">
        <div className="sn-foot-row">
          <span className="sn-foot-k">specimen</span>
          <span className="sn-live">
            <i />
            {model}
          </span>
        </div>
        <div className="sn-depth">
          <span>depth</span>
          <span className="sn-depth-bar">
            <span style={{ width: `${(progress * 100).toFixed(0)}%` }} />
          </span>
          <span>{Math.round(progress * 100)}%</span>
        </div>
        <div className="sn-tokens">{tokens || '—'} tokens in scope</div>
      </div>
    </nav>
  )
}

export function ActHeader({ act }: { act: Act }) {
  return (
    <div className="act" id={act.id}>
      <span className="act-n">{act.n}</span>
      <span className="act-name">{act.name}</span>
      <span className="act-sub lbl">{act.sub}</span>
      <span className="act-rule" />
    </div>
  )
}
