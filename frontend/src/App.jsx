import { useEffect, useRef, useState } from 'react'
import { Search, Loader2, Globe, AlertCircle, CheckCircle, XCircle, HelpCircle, History, Github } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import MermaidDiagram from './components/MermaidDiagram.jsx'

const API = '/api'

function normalizeUrl(input) {
  if (!input) return ''
  let candidate = input.trim()
  if (!candidate) return ''
  if (!/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(candidate)) {
    candidate = `https://${candidate}`
  }
  try {
    const parsed = new URL(candidate)
    const host = parsed.hostname.toLowerCase()
    // Add www. only for bare apex domains (e.g. example.com), not subdomains.
    if (!host.startsWith('www.') && host.split('.').length === 2) {
      parsed.hostname = `www.${host}`
    }
    return parsed.toString()
  } catch {
    return candidate
  }
}

function App() {
  const [companyName, setCompanyName] = useState('')
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [currentNode, setCurrentNode] = useState('')
  const [logs, setLogs] = useState([])
  const [baseMermaid, setBaseMermaid] = useState('')
  const [mermaidDef, setMermaidDef] = useState('')
  const INITIAL_NODES = {
    normalize_input: 'pending',
    searxng: 'pending',
    brave: 'pending',
    tavily: 'pending',
    validate_candidate: 'pending',
    scrape_homepage: 'pending',
    validate_scrape: 'pending',
    retry_scrape: 'pending',
    try_alternate: 'pending',
    assess_evidence: 'pending',
    discover_evidence: 'pending',
    scrape_evidence: 'pending',
    validate_evidence: 'pending',
    classify: 'pending',
    validate_classification: 'pending',
    terminate: 'pending',
  }
  const [nodes, setNodes] = useState(INITIAL_NODES)

  const NODE_ORDER = Object.keys(INITIAL_NODES)
  const NODE_EXPLANATIONS = [
    ['normalize_input', 'Normalize Input', 'Cleans up your input — adds https://www. if missing, and decides whether to search for the company or use your URL directly.'],
    ['searxng', 'SearXNG', 'Searches for the company\'s official website using the self-hosted SearXNG instance.'],
    ['brave', 'Brave Search', 'Fallback search if SearXNG finds nothing usable.'],
    ['tavily', 'Tavily Search', 'Final search fallback before giving up.'],
    ['validate_candidate', 'Validate Candidate', 'Ranks search results by how well the domain matches the company name — rejects Wikipedia, LinkedIn, directories, app stores, sandbox/staging hosts, and foreign-ccTLD lookalikes. Recognizes acronym domains ("Electronic Arts" → ea.com). Prefers the apex domain over portal subdomains (careers.*, shop.*) and deep pages.'],
    ['scrape_homepage', 'Scrape Homepage', 'Downloads the homepage content via Firecrawl.'],
    ['validate_scrape', 'Validate Scrape', 'Verifies the scrape actually returned usable text — rejects empty pages, error pages, captchas, and JavaScript-only stubs. Thin but real pages still proceed to evidence discovery.'],
    ['retry_scrape', 'Retry Scrape', 'Retries with a different strategy: first Firecrawl with extra JS-render wait, then a direct HTTP fetch.'],
    ['try_alternate', 'Try Alternate Candidate', 'When scrape retries are exhausted on a dead or lookalike domain, promotes the next alternate candidate and re-scrapes it instead of giving up.'],
    ['assess_evidence', 'Assess Evidence', 'Checks whether the homepage alone contains corporate-identity signals (headquarters, incorporation, founding location).'],
    ['discover_evidence', 'Discover Evidence Pages', 'Maps the site via Firecrawl\'s /v1/map endpoint to find real corporate pages — About, Legal, Investors, Contact, Careers, Newsroom — falling back to homepage links and canonical path guesses when the map is empty.'],
    ['scrape_evidence', 'Scrape Evidence Pages', 'Scrapes evidence pages in parallel and validates each. Always adds a Wikipedia reference when a company name is available; if identity is still unresolved, tries alternate candidate domains (capped at 3 consecutive failures per domain). If no careers page yielded content, searches for an external careers portal.'],
    ['validate_evidence', 'Validate Evidence', 'Builds the evidence bundle sent to the LLM, annotating each page with detected Canadian and foreign location mentions. If no usable evidence exists, classification is skipped entirely.'],
    ['classify', 'Classify with Ollama', 'The LLM reads only the evidence bundle and answers Yes/No/Unclear as strict JSON, citing verbatim evidence excerpts it relied on.'],
    ['validate_classification', 'Validate Classification', 'Deterministic check: every cited quote is verified against the scraped content — unverifiable quotes are discarded. A "Yes" requires Canadian evidence, a "No" requires foreign evidence — otherwise downgraded to Unclear. Upgrades "Employs Canadians" to Yes when a careers page is on a .ca domain, a Canadian-locale path (/en-ca/, /fr-ca/), or lists Canadian locations.'],
    ['terminate', 'Terminate', 'Final step — assembles the result or the failure reason and ends the workflow.'],
  ]
  const seenNodesRef = useRef(new Set())

  useEffect(() => {
    async function loadMermaid(retries = 5) {
      try {
        const res = await fetch(`${API}/graph/mermaid`, { cache: 'no-store' })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const text = await res.text()
        setBaseMermaid(text)
      } catch (err) {
        if (retries > 0) {
          setTimeout(() => loadMermaid(retries - 1), 1000)
        } else {
          console.error('Failed to load graph', err)
        }
      }
    }
    loadMermaid()
  }, [])

  useEffect(() => {
    if (!baseMermaid) return
    const classDefs = `classDef pending fill:#f1f5f9,stroke:#94a3b8,color:#0f172a\nclassDef running fill:#fef3c7,stroke:#f59e0b,color:#92400e\nclassDef completed fill:#d1fae5,stroke:#10b981,color:#064e3b\nclassDef error fill:#fee2e2,stroke:#ef4444,color:#7f1d1d`
    const classLines = NODE_ORDER.map((id) => `class ${id} ${nodes[id]};`).join('\n')
    setMermaidDef(`${baseMermaid}\n${classDefs}\n${classLines}`)
  }, [baseMermaid, nodes])

  const handleStream = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    setLogs([])
    setCurrentNode('__start__')
    seenNodesRef.current = new Set()
    setNodes({ ...INITIAL_NODES })

    const normalizedUrl = normalizeUrl(url)

    const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
    const MAX_BACKEND_RETRIES = 3
    const RETRY_DELAY_MS = 5000

    const fetchStream = async () => {
      let lastError = null
      for (let attempt = 0; attempt <= MAX_BACKEND_RETRIES; attempt++) {
        try {
          const response = await fetch(`${API}/check/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              company_name: companyName || undefined,
              url: normalizedUrl || undefined,
            }),
          })
          if (response.ok && response.body) return response
          // Backend reachable but not ready (e.g. still starting up).
          if ([404, 502, 503].includes(response.status)) {
            lastError = new Error('The backend is still starting up. Please try again in a moment.')
          } else {
            throw new Error(`Request failed: ${response.status}`)
          }
        } catch (err) {
          // Network error (backend unreachable) — retry silently.
          if (err instanceof TypeError) {
            lastError = new Error('Could not reach the backend. Please try again in a moment.')
          } else {
            throw err
          }
        }
        if (attempt < MAX_BACKEND_RETRIES) await sleep(RETRY_DELAY_MS)
      }
      throw lastError
    }

    try {
      const response = await fetchStream()

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const dataLine = line.trim().split('\n').find((l) => l.startsWith('data:'))
          if (!dataLine) continue
          const payload = dataLine.slice(5).trim()
          if (!payload) continue

          const msg = JSON.parse(payload)
          if (msg.node === '__end__') {
            setCurrentNode('__end__')
            setNodes((prev) => {
              const next = { ...prev }
              NODE_ORDER.forEach((id) => {
                if (next[id] === 'running') next[id] = 'completed'
              })
              return next
            })
            setLoading(false)
            return
          }
          if (msg.node === '__error__') {
            setError(msg.error || 'Stream error')
            setLoading(false)
            return
          }
          setCurrentNode(msg.node)
          const update = msg.update || {}
          if (update.trace) {
            setLogs((prev) => [...prev, ...update.trace])
          }
          if (update.error || update.classification_status === 'not_performed') {
            setNodes((prev) => ({ ...prev, [msg.node]: 'error' }))
          } else if (msg.node && !['__start__', '__end__', '__error__'].includes(msg.node)) {
            seenNodesRef.current.add(msg.node)
            setNodes((prev) => {
              const next = {}
              NODE_ORDER.forEach((id) => {
                if (id === msg.node) next[id] = 'running'
                else if (prev[id] === 'error') next[id] = 'error'
                else if (seenNodesRef.current.has(id)) next[id] = 'completed'
                else next[id] = 'pending'
              })
              return next
            })
          }
          setResult((prev) => {
            const base = prev || { answer: '', reasoning: '', trace: [] }
            return {
              ...base,
              ...update,
              trace: [...base.trace, ...(update.trace || [])],
            }
          })
        }
      }
      setLoading(false)
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  const answerIcon = () => {
    const answer = result?.answer?.toLowerCase() || ''
    if (answer.includes('yes')) return <img src="/favicon.svg" alt="Canadian" className="w-7 h-7" />
    if (answer.includes('no')) return <XCircle className="w-6 h-6 text-red-600" />
    return <HelpCircle className="w-6 h-6 text-amber-600" />
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="bg-gradient-to-r from-red-600 to-red-500 text-white py-5 lg:py-8 shadow">
        <div className="max-w-4xl mx-auto px-4">
          <h1 className="text-3xl font-bold flex items-center justify-center gap-2">
            <img src="/favicon.svg" alt="Maple leaf logo" className="w-9 h-9" />
            Is It Canadian?
          </h1>
          <div className="mt-2 flex justify-center gap-2">
            <a
              href="https://davidrintoul.info"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 bg-white/15 hover:bg-white/25 text-red-50 text-xs font-medium rounded-full px-3 py-1 transition-colors"
            >
              by David Rintoul
            </a>
            <a
              href="https://github.com/drintoul/is-it-canadian"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 bg-white/15 hover:bg-white/25 text-red-50 text-xs font-medium rounded-full px-3 py-1 transition-colors"
            >
              <Github className="w-3.5 h-3.5" />
              Source
            </a>
          </div>
          <p className="text-red-100 mt-2 max-w-3xl mx-auto">
            Enter a company name or paste a URL and we'll check whether the
            company is Canadian — and whether it employs Canadians. The agent
            finds the official website, reads its corporate pages (About,
            Careers, Legal, Investors), and answers Yes, No, or Unclear —
            always citing the exact evidence it relied on, with links to the
            source pages. No evidence, no verdict.
          </p>
          <p className="lg:hidden text-red-200 text-sm mt-2 max-w-3xl mx-auto">
            Visit this page on a desktop or laptop to see the agent's inner
            workings — the live workflow diagram, step-by-step explanations,
            and execution trace.
          </p>
          <p className="hidden lg:block text-red-200 text-sm mt-2 max-w-3xl mx-auto">
            Under the hood, an agentic LangGraph workflow orchestrates the
            check: a SearXNG → Brave → Tavily search fallback chain locates the
            official site, Firecrawl scrapes it with site-map discovery and
            canonical-path guessing, a local Ollama model classifies strictly
            from the evidence bundle, and deterministic post-validation
            verifies every cited quote against the scraped content before the
            answer is shown.
          </p>
        </div>
      </header>

      <main className="max-w-[1800px] mx-auto px-6 py-8 space-y-6">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            handleStream()
          }}
          className="bg-white rounded-2xl shadow p-4 space-y-3 max-w-xl mx-auto w-full"
        >
          <div className="flex gap-2">
            <input
              type="text"
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              placeholder="Company name, e.g., Shopify"
              className="flex-1 min-w-0 border border-slate-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-red-500"
            />
            <button
              type="submit"
              disabled={loading || (!companyName && !url)}
              className="shrink-0 bg-red-600 hover:bg-red-700 disabled:opacity-60 text-white px-4 sm:px-5 py-2 rounded-lg font-medium flex items-center gap-2"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              {loading ? 'Analyzing...' : 'Check'}
            </button>
          </div>
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Or paste a URL directly, e.g., https://www.shopify.com"
            className={`
              w-full border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-red-500
              ${result?.error && !result?.url ? 'border-amber-400 bg-amber-50' : 'border-slate-300'}
            `}
          />
          {result?.error && !result?.url && (
            <p className="text-sm text-amber-700">
              Couldn't find a website automatically. Paste the company's URL above and click <strong>Check</strong>.
            </p>
          )}
        </form>

        {loading && (
          <div className="lg:hidden flex flex-col items-center gap-3 py-4 text-slate-600">
            <Loader2 className="w-10 h-10 animate-spin text-red-600" />
            <p className="text-sm font-medium">Analyzing...</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
          <div className="hidden lg:block bg-white rounded-2xl shadow p-6">
            <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
              <Globe className="w-5 h-5 text-red-600" />
              LangGraph Workflow
            </h2>
            <div className="overflow-x-auto flex justify-center">
              <MermaidDiagram definition={mermaidDef || 'graph TD\n  start[Start]'} />
            </div>
          </div>

          <div className="hidden lg:block bg-white rounded-2xl shadow p-6">
            <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
              <HelpCircle className="w-5 h-5 text-red-600" />
              What Each Step Means
            </h2>
            <ul className="space-y-3 text-sm">
              {NODE_EXPLANATIONS.map(([id, title, text]) => {
                const status = nodes[id]
                const isCurrent = currentNode === id && loading
                return (
                  <li
                    key={id}
                    className={`rounded-lg px-3 py-2 border transition-colors ${
                      isCurrent
                        ? 'border-red-300 bg-red-50'
                        : status === 'completed'
                          ? 'border-green-200 bg-green-50'
                          : status === 'error'
                            ? 'border-red-200 bg-red-50'
                            : 'border-transparent'
                    }`}
                  >
                    <div className="font-medium text-slate-800 flex items-center gap-2">
                      {status === 'completed' && <CheckCircle className="w-3.5 h-3.5 text-green-600" />}
                      {status === 'error' && <XCircle className="w-3.5 h-3.5 text-red-600" />}
                      {isCurrent && <Loader2 className="w-3.5 h-3.5 animate-spin text-red-600" />}
                      {title}
                    </div>
                    <p className="text-slate-600 mt-0.5">{text}</p>
                  </li>
                )
              })}
            </ul>
          </div>

          <div className="space-y-6">
            {error && (
              <div className="bg-red-50 text-red-700 p-4 rounded-xl border border-red-200 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                <div>{error}</div>
              </div>
            )}

            {result && (
              <div className="space-y-6">
                <div className="bg-white rounded-2xl shadow p-6">
                  <div className="flex items-center gap-3 mb-4">
                    {answerIcon()}
                    <h2 className="text-xl font-semibold">
                      {result.classification_status === 'not_performed'
                        ? 'Classification not performed'
                        : `Answer: ${result.answer || '...'}`}
                    </h2>
                  </div>
                  {(result.confidence || result.employs_canadians) && (
                    <div className="flex flex-wrap gap-2 mb-3 text-sm">
                      {result.confidence && (
                        <span className="px-2 py-1 rounded-md bg-slate-100 text-slate-700">
                          Confidence: <strong>{result.confidence}</strong>
                        </span>
                      )}
                      {result.employs_canadians && (
                        <span className="px-2 py-1 rounded-md bg-slate-100 text-slate-700">
                          Employs Canadians: <strong>{result.employs_canadians}</strong>
                        </span>
                      )}
                    </div>
                  )}
                  {result.error && (
                    <p className="text-sm text-red-600 mb-3">{result.error}</p>
                  )}
                  {result.url && (
                    <p className="text-sm text-slate-500 mb-3">
                      Found site:{' '}
                      <a href={result.url} target="_blank" rel="noreferrer" className="underline hover:text-red-600">
                        {result.url}
                      </a>
                    </p>
                  )}
                  {result.reasoning && (
                    <div className="prose prose-slate max-w-none text-slate-700">
                      <ReactMarkdown>{result.reasoning}</ReactMarkdown>
                    </div>
                  )}
                  {(() => {
                    const seen = new Set()
                    const evidence = [
                      ...(result.canadian_evidence || []),
                      ...(result.non_canadian_evidence || []),
                      ...(result.employment_evidence || []),
                    ].filter((e) => {
                      if (!e.quote_or_excerpt && !e.claim) return false
                      const key = `${e.quote_or_excerpt || ''}|${e.source_url || ''}`
                      if (seen.has(key)) return false
                      seen.add(key)
                      return true
                    })
                    if (!evidence.length) return null
                    return (
                      <div className="mt-4 border-t border-slate-200 pt-3">
                        <h3 className="text-sm font-semibold text-slate-700 mb-2">Evidence</h3>
                        <ul className="space-y-2">
                          {evidence.map((e, i) => (
                            <li key={i} className="text-sm bg-slate-50 border border-slate-200 rounded-md px-3 py-2">
                              {e.claim && !(e.quote_or_excerpt && (() => {
                                const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
                                const c = norm(e.claim), q = norm(e.quote_or_excerpt)
                                return q.includes(c) || c.includes(q)
                              })()) && <p className="font-medium text-slate-800">{e.claim}</p>}
                              {e.quote_or_excerpt && (
                                <blockquote className="text-slate-600 italic border-l-2 border-red-300 pl-2 mt-1">
                                  “{e.quote_or_excerpt}”
                                </blockquote>
                              )}
                              {e.source_url && (
                                <a href={e.source_url} target="_blank" rel="noreferrer"
                                   className="text-xs text-blue-600 underline hover:text-blue-800 mt-1 inline-block break-all">
                                  {e.source_url}
                                </a>
                              )}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )
                  })()}
                </div>

              </div>
            )}

            {(loading || logs.length > 0) && (
              <div className="hidden lg:block bg-white rounded-2xl shadow p-6 min-w-0 overflow-hidden">
                <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
                  <History className="w-5 h-5 text-red-600" />
                  Execution Trace
                </h2>
                <ol className="list-decimal list-inside space-y-1 text-slate-700 text-sm">
                  {logs.map((t, i) => (
                    <li key={i} className="break-all">{t}</li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="max-w-4xl mx-auto px-4 pb-16">
        <p className="text-xs text-slate-500 text-center border-t border-slate-200 pt-4">
          Disclaimer: This tool uses AI to analyze publicly available information
          and can make mistakes. While every effort has been made to eliminate
          hallucinations — including requiring cited evidence for every verdict —
          errors are still possible. The author is not responsible for decisions
          made based on this information. Always verify important details
          independently.
        </p>
      </footer>
    </div>
  )
}

export default App
