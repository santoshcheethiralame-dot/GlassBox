export function cleanToken(t: string): string {
  if (t === '<|endoftext|>') return '⟨BOS⟩'
  if (t === '') return '∅'
  return t.replace(/\n/g, '↵')
}

export const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

export const short = (t: string, n: number) => {
  const s = cleanToken(t).trim()
  return s.length > n ? s.slice(0, n) : s
}

function rgb(a: number[], b: number[], t: number): string {
  t = clamp(t, 0, 1)
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},${Math.round(a[1] + (b[1] - a[1]) * t)},${Math.round(a[2] + (b[2] - a[2]) * t)})`
}

export const accMix = (t: number) => rgb([18, 22, 30], [255, 162, 77], t)
export const negMix = (t: number) => rgb([18, 22, 30], [108, 122, 153], t)
