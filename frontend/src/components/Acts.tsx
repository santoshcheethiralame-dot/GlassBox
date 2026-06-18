export type Act = { id: string; n: string; name: string; sub: string }

export const ACTS: Act[] = [
  { id: 'observe', n: '01', name: 'OBSERVE', sub: 'what the model sees' },
  { id: 'decode', n: '02', name: 'DECODE', sub: 'features in the stream' },
  { id: 'intervene', n: '03', name: 'INTERVENE', sub: 'ablate · steer' },
  { id: 'causal', n: '04', name: 'CAUSAL', sub: 'what caused the answer' },
  { id: 'experiment', n: '05', name: 'EXPERIMENT', sub: 'hallucination lab' },
]

export function SideNav({ active }: { active: string }) {
  return (
    <nav className="sidenav">
      <div className="sn-head lbl">acts</div>
      <div className="sn-list">
        {ACTS.map((a) => (
          <a key={a.id} href={`#${a.id}`} className={`sn-item ${active === a.id ? 'on' : ''}`}>
            <span className="sn-n">{a.n}</span>
            <span className="sn-tx">
              <span className="sn-name">{a.name}</span>
              <span className="sn-sub">{a.sub}</span>
            </span>
          </a>
        ))}
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
