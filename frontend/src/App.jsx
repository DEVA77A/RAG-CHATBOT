import React, { useState, useEffect, useRef } from 'react';

function App() {
  // State
  const [url, setUrl] = useState('');
  const [maxPages, setMaxPages] = useState(10);
  const [status, setStatus] = useState('idle'); // idle, crawling, ready, error
  const [errorMessage, setErrorMessage] = useState('');

  // KB Data
  const [analysisId, setAnalysisId] = useState(null);
  const [kbStats, setKbStats] = useState(null);
  const [indexedPages, setIndexedPages] = useState([]);
  const [timing, setTiming] = useState(null);
  const [domain, setDomain] = useState('');

  // Chat Data
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [debugData, setDebugData] = useState(null);
  const [showDebug, setShowDebug] = useState(false);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Handle URL indexing
  const handleIndex = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;

    setStatus('crawling');
    setErrorMessage('');
    setAnalysisId(null);
    setMessages([]);
    setDebugData(null);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, max_pages: parseInt(maxPages, 10) })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to analyze website');
      }

      setAnalysisId(data.id);
      setKbStats(data.kb_stats);
      setIndexedPages(data.indexed_pages || []);
      setDomain(data.domain || url);
      setTiming({
        crawl: data.crawl_time,
        index: data.index_time,
        total: data.total_time
      });
      setStatus('ready');

      // Add welcome message
      setMessages([{
        role: 'assistant',
        content: `I've finished crawling **${data.domain}**. Indexed ${data.kb_stats.total_pages} pages and created ${data.kb_stats.total_chunks} search chunks. What would you like to know about this website?`
      }]);

    } catch (err) {
      console.error("Index error:", err);
      setStatus('error');
      setErrorMessage(err.message);
    }
  };

  // Handle chat submission
  const handleChat = async (e) => {
    if (e) e.preventDefault();
    if (!input.trim() || !analysisId || isTyping) return;

    const userMsg = input.trim();
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsTyping(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysis_id: analysisId,
          message: userMsg,
          top_k: 5
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to get response');
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        debug: data.debug
      }]);

      if (data.debug) {
        setDebugData(data.debug);
      }

    } catch (err) {
      console.error("Chat error:", err);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.message}`,
        isError: true
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleChat();
    }
  };

  const adjustTextareaHeight = (e) => {
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
  };

  return (
    <div className="app-container">

      {/* ─── LEFT SIDEBAR ─── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>WebIntel AI</h2>
          <span className="badge">Production RAG</span>
        </div>

        <div className="sidebar-content">
          <div className="card">
            <h3>Knowledge Base</h3>
            <form onSubmit={handleIndex} className="index-form">
              <div className="input-group">
                <input
                  type="url"
                  placeholder="https://example.com"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  disabled={status === 'crawling'}
                  required
                />
              </div>
              <div className="input-group inline">
                <label>Max Pages:</label>
                <select
                  value={maxPages}
                  onChange={(e) => setMaxPages(e.target.value)}
                  disabled={status === 'crawling'}
                >
                  <option value={5}>5 (Fast)</option>
                  <option value={10}>10 (Standard)</option>
                  <option value={20}>20 (Deep)</option>
                </select>
              </div>
              <button
                type="submit"
                className="btn-primary"
                disabled={status === 'crawling' || !url}
              >
                {status === 'crawling' ? 'Crawling...' : 'Crawl & Index'}
              </button>
            </form>

            {status === 'crawling' && (
              <div className="progress-container">
                <div className="spinner-small"></div>
                <span>Crawling website in parallel...</span>
              </div>
            )}

            {status === 'error' && (
              <div className="error-alert">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                {errorMessage}
              </div>
            )}
          </div>

          {status === 'ready' && kbStats && (
            <>
              <div className="stats-grid">
                <div className="stat-box">
                  <div className="stat-value">{kbStats.total_pages}</div>
                  <div className="stat-label">Pages</div>
                </div>
                <div className="stat-box">
                  <div className="stat-value">{kbStats.total_chunks}</div>
                  <div className="stat-label">Chunks</div>
                </div>
                <div className="stat-box">
                  <div className="stat-value">{(kbStats.total_chars / 1024).toFixed(1)}k</div>
                  <div className="stat-label">KB Data</div>
                </div>
              </div>

              <div className="sources-list-container">
                <h3>Indexed Sources</h3>
                <ul className="sources-list">
                  {indexedPages.map((page, idx) => (
                    <li key={idx} title={page.url}>
                      <div className="source-title">{page.title || page.url.split('/').pop() || page.url}</div>
                      <div className="source-meta">
                        <span className="badge-small">{page.chunk_count} chunks</span>
                        <a href={page.url} target="_blank" rel="noreferrer" className="source-link">View</a>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>
      </aside>

      {/* ─── MAIN CHAT PANEL ─── */}
      <main className="main-panel">
        <div className="top-nav">
          <h3>{domain ? `Chatting with ${domain}` : 'Waiting for context...'}</h3>
          <button
            className={`btn-toggle ${showDebug ? 'active' : ''}`}
            onClick={() => setShowDebug(!showDebug)}
            disabled={!debugData}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"></path></svg>
            Judge Mode
          </button>
        </div>

        <div className="chat-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">🌐</div>
              <h2>Retrieval-Augmented Generation</h2>
              <p>Enter a URL in the sidebar to build a knowledge base, then ask questions grounded in that context.</p>
            </div>
          ) : (
            <div className="messages-wrapper">
              {messages.map((msg, idx) => (
                <div key={idx} className={`message-row ${msg.role}`}>
                  <div className="avatar">
                    {msg.role === 'user' ? 'U' : 'AI'}
                  </div>
                  <div className="message-content">
                    <div className="bubble">
                      {msg.content}
                    </div>
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="citations-inline">
                        <strong>Sources cited:</strong>
                        <div className="citation-chips">
                          {msg.sources.map((s, i) => (
                            <span key={i} className="citation-chip" title={s.url}>
                              [Source {s.chunk_id + 1}] Similarity: {s.score.toFixed(2)}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="message-row assistant">
                  <div className="avatar">AI</div>
                  <div className="message-content">
                    <div className="typing-indicator">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="input-area">
          <form onSubmit={handleChat}>
            <div className="input-wrapper">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  adjustTextareaHeight(e);
                }}
                onKeyDown={handleKeyDown}
                placeholder={status === 'ready' ? "Ask a question based on the website..." : "Index a website first..."}
                disabled={status !== 'ready' || isTyping}
                rows={1}
              />
              <button
                type="submit"
                className="btn-send"
                disabled={!input.trim() || status !== 'ready' || isTyping}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
              </button>
            </div>
            <div className="footer-text">
              Strictly grounded in retrieved context. Hallucinations minimized.
            </div>
          </form>
        </div>
      </main>

      {/* ─── RIGHT DEBUG PANEL (JUDGE MODE) ─── */}
      {showDebug && debugData && (
        <aside className="debug-panel">
          <div className="debug-header">
            <h3>Judge Mode</h3>
            <button className="close-btn" onClick={() => setShowDebug(false)}>✕</button>
          </div>

          <div className="debug-content">

            <div className="debug-section">
              <h4>Performance Metrics</h4>
              <div className="metric-row">
                <span>Retrieval Latency</span>
                <span className="mono">{(debugData.retrieval_time * 1000).toFixed(0)} ms</span>
              </div>
              <div className="metric-row">
                <span>Generation Latency</span>
                <span className="mono">{(debugData.generation_time * 1000).toFixed(0)} ms</span>
              </div>
              <div className="metric-row">
                <span>Total Chat Latency</span>
                <span className="mono">{(debugData.total_time * 1000).toFixed(0)} ms</span>
              </div>
            </div>

            <div className="debug-section">
              <h4>Retrieval Engine (Hybrid)</h4>
              <div className="metric-row">
                <span>Expanded Queries</span>
                <span className="mono">{debugData.expanded_queries?.length || 1}</span>
              </div>
              {debugData.expanded_queries?.map((q, i) => (
                <div key={i} className="small-text muted">↳ "{q}"</div>
              ))}
              <div className="metric-row mt-2">
                <span>Dense Hits (FAISS)</span>
                <span className="mono">{debugData.dense_hits}</span>
              </div>
              <div className="metric-row">
                <span>Sparse Hits (BM25)</span>
                <span className="mono">{debugData.sparse_hits}</span>
              </div>
              <div className="metric-row">
                <span>Top-K Sent to LLM</span>
                <span className="mono">{debugData.retrieved_chunks?.length || 0}</span>
              </div>
            </div>

            <div className="debug-section">
              <h4>LLM Payload</h4>
              <div className="metric-row">
                <span>Context Length</span>
                <span className="mono">{debugData.context_length} chars</span>
              </div>
              <div className="metric-row">
                <span>Tokens Sent</span>
                <span className="mono">{debugData.tokens_sent}</span>
              </div>
            </div>

            <div className="debug-section">
              <h4>Retrieved Context Chunks</h4>
              {debugData.retrieved_chunks?.map((chunk, idx) => (
                <div key={idx} className="chunk-card">
                  <div className="chunk-header">
                    <span className="chunk-id">Source {chunk.chunk_id + 1}</span>
                    <span className="chunk-score" title="FAISS Cosine Similarity">Sim: {chunk.score.toFixed(2)}</span>
                  </div>
                  {chunk.heading && <div className="chunk-heading">H: {chunk.heading}</div>}
                  <div className="chunk-text">{chunk.chunk_text}</div>
                </div>
              ))}
            </div>

          </div>
        </aside>
      )}

    </div>
  );
}

export default App;
