import { useEffect, useState } from 'react'
import type { ChangeEvent } from 'react'
import { runAttribute } from '../api'
import type { AttributeResponse } from '../types'
import { accMix, cleanToken, negMix } from '../util'

const DEF = {
  clean_prompt: 'Alice lives in Paris. Bob lives in London. Alice lives in',
  corrupted_prompt: 'Alice lives in Rome. Bob lives in London. Alice lives in',
  answer: ' Paris',
  corrupted_answer: ' Rome',
}

type Method = 'attribution' | 'activation'

export function AttributionView({ model }: { model: string }) {
  const [form, setForm] = useState(DEF)
  const [method, setMethod] = useState<Method>('attribution')
  const [data, setData] = useState<AttributeResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const run = (input: typeof DEF, m: Method) => {
    setLoading(true)
    setError(null)
    runAttribute({ ...input, method: m, model })
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    run(form, method)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [method, model])

  const set = (k: keyof typeof DEF) => (e: ChangeEvent<HTMLInputElement>) =>
    setForm({ ...form, [k]: e.target.value })

  const maxAbs = data ? data.attribution.reduce((m, a) => Math.max(m, Math.abs(a)), 0) || 1 : 1

  return (
    <section className="mod">
      <div className="mod-head">
        <div className="idx">08</div>
        <div className="ttl">CONTEXT ATTRIBUTION</div>
        <div className="hmeta">
          <span className="lbl">
            causal · <b>resid_pre.0</b>
          </span>
        </div>
      </div>
      <div className="mod-body">
        <div className="note">
          Which token <b>causes</b> the answer? Attribute the clean-vs-corrupted logit gap back to
          each position — <b>FAST</b> (attribution patching) approximates in one forward + backward,{' '}
          <b>VERIFIED</b> (activation patching) confirms by patching each position.
        </div>

        <div className="patch-form">
          <label className="fld">
            <span className="lbl">clean</span>
            <input value={form.clean_prompt} onChange={set('clean_prompt')} />
            <input style={{ maxWidth: 110 }} value={form.answer} onChange={set('answer')} />
          </label>
          <label className="fld">
            <span className="lbl">corrupted</span>
            <input value={form.corrupted_prompt} onChange={set('corrupted_prompt')} />
            <input
              style={{ maxWidth: 110 }}
              value={form.corrupted_answer}
              onChange={set('corrupted_answer')}
            />
          </label>
          <div className="fld">
            <span className="lbl" />
            <span className="seg">
              <button
                className={method === 'attribution' ? 'on' : ''}
                onClick={() => setMethod('attribution')}
              >
                FAST
              </button>
              <button
                className={method === 'activation' ? 'on' : ''}
                onClick={() => setMethod('activation')}
              >
                VERIFIED
              </button>
            </span>
            <button
              className="run"
              style={{ height: 34, padding: '0 22px' }}
              onClick={() => run(form, method)}
              disabled={loading}
            >
              {loading ? 'ATTRIBUTING…' : 'RUN ▶'}
            </button>
          </div>
        </div>

        {error && <div className="err">⚠ {error}</div>}
        {loading && !data && <div className="busy">[ ATTRIBUTING ]</div>}

        {data && (
          <>
            <div className="attrib-meta">
              <span className="lbl">
                logit diff (<b>{cleanToken(data.answer)}</b> − <b>{cleanToken(data.corrupted_answer)}</b>)
              </span>
              <span>
                clean{' '}
                <b>
                  {data.logit_diff_clean >= 0 ? '+' : ''}
                  {data.logit_diff_clean.toFixed(2)}
                </b>
              </span>
              <span>
                corrupted{' '}
                <b>
                  {data.logit_diff_corrupted >= 0 ? '+' : ''}
                  {data.logit_diff_corrupted.toFixed(2)}
                </b>
              </span>
              <span className="lbl" style={{ marginLeft: 'auto' }}>
                {method === 'attribution' ? 'attribution patching · 1 fwd+bwd' : 'activation patching · verified'}
              </span>
            </div>

            <div className="attrib-toks">
              {data.tokens.map((t, i) => {
                const a = data.attribution[i]
                const norm = Math.abs(a) / maxAbs
                const bg = a >= 0 ? accMix(norm) : negMix(norm)
                const diff = data.corrupted_tokens[i] !== t
                return (
                  <span
                    key={i}
                    className={`atok ${diff ? 'diff' : ''}`}
                    style={{ background: bg }}
                    title={`${
                      diff ? `${cleanToken(t)} ↔ ${cleanToken(data.corrupted_tokens[i])}` : cleanToken(t)
                    }  ·  ${a >= 0 ? '+' : ''}${a.toFixed(3)}`}
                  >
                    {cleanToken(t)}
                  </span>
                )
              })}
            </div>

            <div className="lbl" style={{ marginTop: 12, textTransform: 'none', letterSpacing: '0.3px' }}>
              <span style={{ color: 'var(--acc)' }}>red</span> supports "{cleanToken(data.answer)}" ·{' '}
              <span style={{ color: '#6fa8c8' }}>blue</span> supports "{cleanToken(data.corrupted_answer)}" ·
              outlined = corrupted positions · hover for values
            </div>
          </>
        )}
      </div>
    </section>
  )
}
