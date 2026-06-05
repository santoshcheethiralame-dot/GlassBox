import { cleanToken } from '../util'

export function Tokens({ tokens }: { tokens: string[] }) {
  return (
    <div className="panel">
      <h2>
        Tokens <span className="dim">({tokens.length})</span>
      </h2>
      <div className="tokens">
        {tokens.map((t, i) => (
          <span key={i} className={`token ${t === '<|endoftext|>' ? 'bos' : ''}`}>
            {cleanToken(t)}
          </span>
        ))}
      </div>
    </div>
  )
}
