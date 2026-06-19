import type { ReactNode } from 'react'

export type Act = { id: string; n: string; name: string; sub: string }

export const ACTS: Act[] = [
  { id: 'observe', n: '01', name: 'OBSERVE', sub: 'what the model sees' },
  { id: 'decode', n: '02', name: 'DECODE', sub: 'features in the stream' },
  { id: 'intervene', n: '03', name: 'INTERVENE', sub: 'ablate · steer' },
  { id: 'causal', n: '04', name: 'CAUSAL', sub: 'what caused the answer' },
  { id: 'experiment', n: '05', name: 'EXPERIMENT', sub: 'hallucination lab' },
]

const ICONS: Record<string, ReactNode> = {
  observe: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12Z" />
      <circle cx="12" cy="12" r="2.6" />
    </svg>
  ),
  decode: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2 2 7l10 5 10-5-10-5Z" />
      <path d="m2 17 10 5 10-5" />
      <path d="m2 12 10 5 10-5" />
    </svg>
  ),
  intervene: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3" />
      <path d="M1 14h6M9 8h6M17 16h6" />
    </svg>
  ),
  causal: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="5" cy="12" r="2.4" />
      <circle cx="19" cy="6" r="2.4" />
      <circle cx="19" cy="18" r="2.4" />
      <path d="m7.3 10.9 9.5-3.8M7.3 13.1l9.5 3.8" />
    </svg>
  ),
  experiment: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 3h6M10 3v5.5L5.3 17a2 2 0 0 0 1.8 3h9.8a2 2 0 0 0 1.8-3L14 8.5V3" />
      <path d="M7.7 14h8.6" />
    </svg>
  ),
}

export function SideNav({
  active,
  progress,
  tokens,
  model,
  prompt,
}: {
  active: string
  progress: number
  tokens: number
  model: string
  prompt: string
}) {
  return (
    <nav className="sidenav" aria-label="Acts">
      <div className="sn-top">
        <span className="sn-eyebrow">interpretability</span>
        <span className="sn-title">ACTS</span>
      </div>
      <div className="sn-acts">
        {ACTS.map((a) => (
          <a key={a.id} href={`#${a.id}`} className={`sn-item${active === a.id ? ' on' : ''}`}>
            <span className="sn-ico" aria-hidden="true">{ICONS[a.id]}</span>
            <span className="sn-tx">
              <span className="sn-name">{a.name}</span>
              <span className="sn-sub">{a.sub}</span>
            </span>
            <span className="sn-num">{a.n}</span>
          </a>
        ))}
      </div>
      <div className="sn-foot">
        <div className="sn-foot-row">
          <span className="sn-foot-k">specimen</span>
          <span className="sn-live">
            <i />
            {model}
          </span>
        </div>
        <div className="sn-spec">"{prompt}"</div>
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
      <h2 className="act-name">{act.name}</h2>
      <span className="act-sub lbl">{act.sub}</span>
      <span className="act-rule" />
    </div>
  )
}
