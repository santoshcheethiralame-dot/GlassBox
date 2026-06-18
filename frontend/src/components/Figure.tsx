import { useRef } from 'react'
import type { ReactNode } from 'react'
import { toPng } from 'html-to-image'

export function Figure({ name, children }: { name: string; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)

  const onExport = async () => {
    const wrap = ref.current
    if (!wrap) return
    const node = (wrap.querySelector('.mod') as HTMLElement) || wrap
    try {
      const url = await toPng(node, { backgroundColor: '#07080c', pixelRatio: 2 })
      const a = document.createElement('a')
      a.href = url
      a.download = `glassbox-${name}.png`
      a.click()
    } catch (e) {
      console.error('export failed', e)
    }
  }

  return (
    <div className="figure" ref={ref}>
      <button className="fig-export" onClick={onExport} title="export this panel as PNG">
        ⬇ PNG
      </button>
      {children}
    </div>
  )
}
