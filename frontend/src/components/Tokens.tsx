import { cleanToken } from '../util'
import type { TokenPrediction } from '../types'

export function Tokens({
  tokens,
  preds,
  focus,
  onFocus,
}: {
  tokens: string[]
  preds: TokenPrediction[]
  focus: number | null
  onFocus: (i: number | null) => void
}) {
  const top = preds[0]
  const mx = top ? top.prob : 1
  return (
    <section className="mod">
      <div className="mod-head">
        <div className="idx">01</div>
        <div className="ttl">TOKENIZATION</div>
        <div className="hmeta">
          <span className="lbl">
            bpe · <b>{tokens.length} tokens</b>
          </span>
        </div>
      </div>
      <div className="mod-body">
        <div className="tokens">
          {tokens.map((t, i) => (
            <div
              key={i}
              className={`tk ${focus === i ? 'sel' : ''}`}
              onClick={() => onFocus(focus === i ? null : i)}
            >
              <span className="t">{cleanToken(t)}</span>
              <span className="i">{i}</span>
            </div>
          ))}
        </div>
        {top && (
          <div className="pred">
            <div className="pl">NEXT</div>
            <div className="big">{cleanToken(top.token).trim() || cleanToken(top.token)}</div>
            <div className="bars">
              {preds.map((p, i) => (
                <div key={i} className={`pbar ${i === 0 ? 'first' : ''}`}>
                  <span className="pt">{cleanToken(p.token)}</span>
                  <span className="pf" style={{ width: `${Math.max(2, (p.prob / mx) * 260)}px` }} />
                  <span className="pp">{(p.prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
