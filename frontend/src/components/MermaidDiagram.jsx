import { useEffect, useRef } from 'react'
import mermaid from 'mermaid'

mermaid.initialize({
  startOnLoad: false,
  theme: 'base',
  securityLevel: 'loose',
  flowchart: {
    useMaxWidth: true,
    htmlLabels: true,
    curve: 'basis',
    padding: 12,
    nodeSpacing: 28,
    rankSpacing: 36,
  },
  themeVariables: {
    fontSize: '14px',
    fontFamily: 'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    primaryColor: '#f8fafc',
    primaryTextColor: '#0f172a',
    primaryBorderColor: '#94a3b8',
    lineColor: '#64748b',
    secondaryColor: '#f1f5f9',
    tertiaryColor: '#e2e8f0',
  },
})

export default function MermaidDiagram({ definition }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!definition || !containerRef.current) return

    let cancelled = false
    const id = `mermaid-${Math.random().toString(36).slice(2, 9)}`

    mermaid
      .render(id, definition)
      .then(({ svg }) => {
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = svg
        }
      })
      .catch((err) => {
        console.error('Mermaid render failed', err)
      })

    return () => {
      cancelled = true
    }
  }, [definition])

  return <div ref={containerRef} className="mermaid w-full h-full" />
}
