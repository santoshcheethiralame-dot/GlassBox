import { useEffect, useMemo, useState } from 'react'
import { LineChart } from './LineChart'
import type { LineSeries } from './LineChart'
import { getConcepts, runProbes } from '../api'
import type { ConceptInfo, ProbeResponse } from '../types'

const COLORS = ['#a3e635', '#38bdf8', '#f472b6', '#fbbf24', '#c084fc']

function shortLabel(key: string): string {
  return key === 'subject_number' ? 'subj. number' : key
}

export function ProbeView() {
  const [concepts, setConcepts] = useState<ConceptInfo[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [res, setRes] = useState<ProbeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getConcepts()
      .then((cs) => {
        setConcepts(cs)
        setSelected(new Set(cs.map((c) => c.key)))
      })
      .catch(() => setConcepts([]))
  }, [])

  const colorOf = useMemo(() => {
    const m: Record<string, string> = {}
    concepts.forEach((c, i) => (m[c.key] = COLORS[i % COLORS.length]))
    return m
  }, [concepts])

  const toggle = (key: string) => {
    const next = new Set(selected)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    setSelected(next)
  }

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      setRes(await runProbes([...selected]))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const series: LineSeries[] = res
    ? res.results.map((r) => ({
        key: r.key,
        label: shortLabel(r.key),
        color: colorOf[r.key] ?? '#a3e635',
        values: r.test_acc,
      }))
    : []

  return (
    <div className="panel">
      <h2>
        Linear probing <span className="dim">— when does a concept become linearly decodable?</span>
      </h2>
      <p className="hint">
        For each concept a logistic-regression probe is trained on the last-token residual stream at
        every layer (split by word, so the test set holds out unseen words) and its test accuracy is
        plotted. <b>Lexical</b> features live in the token embeddings, so they are decodable from
        layer 0; a <b>contextual</b> feature like subject number — read past a distracting noun of the
        opposite number — must be computed, so it starts near (or below) chance and emerges over the
        layers.
      </p>

      <div className="selector">
        <span className="sel-label">CONCEPTS</span>
        {concepts.map((c) => (
          <button
            key={c.key}
            className={`chip probe-chip ${selected.has(c.key) ? 'active' : ''}`}
            style={selected.has(c.key) ? { borderColor: colorOf[c.key], color: colorOf[c.key] } : undefined}
            onClick={() => toggle(c.key)}
            title={`${c.label} — ${c.kind}`}
          >
            <span className="swatch-dot" style={{ background: colorOf[c.key] }} />
            {c.label} <span className="kind">· {c.kind}</span>
          </button>
        ))}
        <button className="btn" onClick={run} disabled={loading || selected.size === 0}>
          {loading ? 'Probing…' : 'Run probes ▸'}
        </button>
      </div>

      {error && <div className="error">⚠ {error}</div>}

      {res && series.length > 0 && (
        <>
          <div className="heatmap-wrap">
            <LineChart series={series} nx={res.n_layers} baseline={0.5} />
          </div>
          <div className="legend">
            <span className="swatch" style={{ background: '#4b5161' }} /> dashed = 0.5 chance
            &nbsp;·&nbsp; higher = more linearly decodable at that layer
          </div>
        </>
      )}
    </div>
  )
}
