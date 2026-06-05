export function cleanToken(t: string): string {
  if (t === '<|endoftext|>') return '⟨bos⟩'
  if (t === '') return '∅'
  return t.replace(/\n/g, '⏎')
}

export const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))
