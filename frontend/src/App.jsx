import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function App() {
  // Config & Status
  const [url, setUrl] = useState('');
  const [maxPages, setMaxPages] = useState(10);
  const [status, setStatus] = useState('idle'); // idle, crawling, ready, error
  const [errorMessage, setErrorMessage] = useState('');

  // KB Data
  const [analysisId, setAnalysisId] = useState(null);
  const [kbStats, setKbStats] = useState(null);
  const [indexedPages, setIndexedPages] = useState([]);
  const [domain, setDomain] = useState('');

  // Chat Data
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [expandedContext, setExpandedContext] = useState({}); // msgIdx -> boolean
  const [chatMetrics, setChatMetrics] = useState(null); // {retrieval_time, generation_time}

  // History Data
  const [historyList, setHistoryList] = useState([]);

  // Search Filters
  const [historySearchQuery, setHistorySearchQuery] = useState('');
  const [sourceSearchQuery, setSourceSearchQuery] = useState('');

  // Debug Data
  const [debugData, setDebugData] = useState(null);
  const [showDebug, setShowDebug] = useState(false);
  const [llmHealth, setLlmHealth] = useState({ gemini: "Checking..." });
  const [showDocRec, setShowDocRec] = useState(false);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Sidebar open/close states
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(() => window.innerWidth > 1024);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(() => window.innerWidth > 1024);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth <= 1024);

  // Responsive: auto-collapse sidebars on resize
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth <= 1024;
      setIsMobile(mobile);
      if (mobile) {
        setLeftSidebarOpen(false);
        setRightSidebarOpen(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Scroll to bottom when messages update
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Automatically recommend higher depth for docs sites
  useEffect(() => {
    const isDoc = /docs|doc|learn|guide|tutorial|wiki|api|reference|react\.dev|python\.org|tiangolo\.com|github\.com|huggingface\.co|geeksforgeeks\.org/i.test(url);
    if (isDoc && url.trim().length > 10) {
      setShowDocRec(true);
      if (maxPages === 10) {
        setMaxPages(25);
      }
    } else {
      setShowDocRec(false);
    }
  }, [url]);

  // Load history list, health, and restore active session on mount
  useEffect(() => {
    fetchHistoryList();
    fetchLlmHealth();

    const savedAnalysisId = localStorage.getItem('active_analysis_id');
    if (savedAnalysisId) {
      loadSession(savedAnalysisId);
    }
  }, []);

  const fetchLlmHealth = async () => {
    try {
      const res = await fetch('/api/health_llm');
      if (res.ok) {
        const data = await res.json();
        setLlmHealth(data);
      }
    } catch (e) {
      console.error("Error fetching LLM health", e);
    }
  };

  const fetchHistoryList = async () => {
    try {
      const res = await fetch('/api/analyses');
      if (res.ok) {
        const data = await res.json();
        setHistoryList(data.analyses || []);
      }
    } catch (err) {
      console.error("Failed to fetch history:", err);
    }
  };

  const formatDate = (isoString) => {
    if (!isoString) return '';
    // SQLite format is YYYY-MM-DD HH:MM:SS. Convert to standard YYYY-MM-DDTHH:MM:SSZ
    const formattedStr = isoString.includes('T') ? isoString : isoString.replace(' ', 'T');
    // If the timestamp already has a timezone indicator (e.g. Z or +00:00), don't append Z
    const hasTimezone = /Z$|[+-]\d{2}:\d{2}$/.test(formattedStr);
    const date = new Date(hasTimezone ? formattedStr : formattedStr + "Z");
    if (isNaN(date.getTime())) return isoString;

    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    const isToday = date.toDateString() === today.toDateString();
    const isYesterday = date.toDateString() === yesterday.toDateString();

    const timeOpts = { hour: 'numeric', minute: '2-digit' };
    if (isToday) return `Today ${date.toLocaleTimeString(undefined, timeOpts)}`;
    if (isYesterday) return `Yesterday`;

    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  const startNewChat = () => {
    localStorage.removeItem('active_analysis_id');
    setAnalysisId(null);
    setKbStats(null);
    setIndexedPages([]);
    setDomain('');
    setMessages([]);
    setStatus('idle');
    setErrorMessage('');
    setUrl('');
    setChatMetrics(null);
  };

  const deleteSession = async (id, event) => {
    event.stopPropagation(); // prevent loading the session when clicking delete
    if (!window.confirm("Are you sure you want to delete this chat session and its vector index? This cannot be undone.")) {
      return;
    }
    try {
      const res = await fetch(`/api/analyze/${id}`, { method: 'DELETE' });
      if (res.ok) {
        if (analysisId === id) {
          localStorage.removeItem('active_analysis_id');
          startNewChat();
        }
        fetchHistoryList();
      } else {
        alert("Failed to delete the session.");
      }
    } catch (e) {
      console.error("Error deleting session", e);
      alert("Error deleting session: " + e.message);
    }
  };

  const loadSession = async (id) => {
    try {
      setStatus('crawling'); // reuse loading state
      localStorage.setItem('active_analysis_id', id);
      const [analysisRes, chatRes] = await Promise.all([
        fetch(`/api/analyze/${id}`),
        fetch(`/api/chat/${id}`)
      ]);

      if (!analysisRes.ok) throw new Error("Failed to load analysis");
      const analysisData = await analysisRes.json();

      setAnalysisId(analysisData.id);
      setKbStats(analysisData.kb_stats);
      setIndexedPages(analysisData.indexed_pages || []);
      setDomain(analysisData.domain);
      setUrl(analysisData.url);

      if (chatRes.ok) {
        const chatData = await chatRes.json();
        const loadedMsgs = chatData.messages.map(m => {
          let parsedSources = [];
          if (Array.isArray(m.sources)) {
            parsedSources = m.sources;
          } else if (typeof m.sources === 'string') {
            try { parsedSources = JSON.parse(m.sources); } catch (e) { }
          }
          if (!Array.isArray(parsedSources)) parsedSources = [];

          return {
            role: m.role,
            content: m.content,
            sources: parsedSources,
          };
        });
        setMessages(loadedMsgs.length ? loadedMsgs : [{
          role: 'assistant',
          content: `Welcome back! This is your previous session for **${analysisData.domain}**.`
        }]);
      }
      setStatus('ready');
      setChatMetrics(null);
    } catch (err) {
      console.error(err);
      setStatus('error');
      setErrorMessage("Could not load session.");
    }
  };

  const handleIndex = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;

    setStatus('crawling');
    setErrorMessage('');
    setAnalysisId(null);
    setMessages([]);
    setDebugData(null);
    setChatMetrics(null);

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
      localStorage.setItem('active_analysis_id', data.id);
      setKbStats(data.kb_stats);
      setIndexedPages(data.indexed_pages || []);
      setDomain(data.domain || url);
      setStatus('ready');

      setMessages([{
        role: 'assistant',
        content: `I've finished crawling **${data.domain}**. Indexed ${data.kb_stats.total_pages} pages and created ${data.kb_stats.total_chunks} search chunks. What would you like to know about this website?`
      }]);

      fetchHistoryList();

    } catch (err) {
      console.error("Index error:", err);
      setStatus('error');
      setErrorMessage(err.message);
    }
  };

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
    setChatMetrics(null);

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
        setChatMetrics({
          retrieval: data.debug.retrieval_time,
          generation: data.debug.generation_time
        });
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

  const toggleContext = (idx) => {
    setExpandedContext(prev => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

  const filteredHistory = historyList.filter(h => {
    const q = historySearchQuery.toLowerCase();
    return (
      (h.title && h.title.toLowerCase().includes(q)) ||
      (h.domain && h.domain.toLowerCase().includes(q)) ||
      (h.url && h.url.toLowerCase().includes(q))
    );
  });

  const filteredSources = indexedPages.filter(page => {
    const q = sourceSearchQuery.toLowerCase();
    return (
      (page.title && page.title.toLowerCase().includes(q)) ||
      (page.url && page.url.toLowerCase().includes(q))
    );
  });

  return (
    <div className="app-container">
      {/* Mobile sidebar overlay backdrop */}
      {isMobile && (leftSidebarOpen || rightSidebarOpen) && (
        <div className="sidebar-overlay" onClick={() => { setLeftSidebarOpen(false); setRightSidebarOpen(false); }} />
      )}
      {/* ─── LEFT SIDEBAR ─── */}
      <aside className={`sidebar ${leftSidebarOpen ? 'open' : 'collapsed'}`}>
        <div className="sidebar-header">
          <div className="app-logo">
            <div className="logo-circle">
              <img src="/logo.jpg" alt="RAGX Logo" />
            </div>
            <div className="logo-info">
              <h2>RAG <span className="logo-x">X</span></h2>
              <span className="logo-version">Enterprise Alpha</span>
            </div>
          </div>
        </div>

        <div className="llm-health-status">
          <div className="health-row">
            <span className={`status-dot ${llmHealth.gemini?.includes('Connected') ? 'online' : 'offline'}`}></span>
            <span className="health-name">Gemini — {llmHealth.gemini}</span>
          </div>
        </div>

        <div className="sidebar-content">
          <div className="new-chat-container">
            <button className="btn-primary w-full" onClick={startNewChat}>
              + New Chat
            </button>
          </div>

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
              <div className="input-group">
                <label>Max Pages</label>
                <select
                  value={maxPages}
                  onChange={(e) => setMaxPages(parseInt(e.target.value, 10))}
                  disabled={status === 'crawling'}
                >
                  <option value={10}>10 (Standard)</option>
                  <option value={25}>25 (Deep)</option>
                  <option value={50}>50 (Full)</option>
                </select>
              </div>
              {showDocRec && (
                <div style={{ fontSize: '11px', color: '#10b981', marginTop: '-6px', marginBottom: '10px', fontWeight: '500' }}>
                  💡 Recommended: 25 or 50 pages for documentation!
                </div>
              )}
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

          {/* Indexed Sources moved to Right Sidebar */}

          {historyList.length > 0 && (
            <div className="history-section mt-4">
              <h3>Previous Chats</h3>
              <input 
                type="text" 
                placeholder="Search chats..." 
                value={historySearchQuery} 
                onChange={(e) => setHistorySearchQuery(e.target.value)} 
                className="sidebar-search-input"
              />
              <div className="history-list">
                {filteredHistory.map(h => (
                  <div key={h.id} className={`history-item ${h.id === analysisId ? 'active' : ''}`} onClick={() => { loadSession(h.id); if (isMobile) setLeftSidebarOpen(false); }}>
                    <div className="history-item-body">
                      <div className="history-title">{h.title || h.domain || h.url}</div>
                      <div className="history-time">{formatDate(h.created_at)}</div>
                    </div>
                    <button className="btn-delete-history" onClick={(e) => deleteSession(h.id, e)} title="Delete session">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* ─── MAIN CHAT PANEL ─── */}
      <main className="main-panel">
        <div className="top-nav flex-between">
          <div className="top-nav-left">
            <button
              className="sidebar-toggle-btn left-toggle"
              onClick={() => setLeftSidebarOpen(!leftSidebarOpen)}
              title={leftSidebarOpen ? "Collapse Left Sidebar" : "Expand Left Sidebar"}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                {leftSidebarOpen ? (
                  <polyline points="15 18 9 12 15 6"></polyline>
                ) : (
                  <polyline points="9 18 15 12 9 6"></polyline>
                )}
              </svg>
            </button>
            <h3>
              {domain ? (
                <>
                  Chatting with {domain}
                  {url && (
                    <a
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="nav-scraped-link"
                      style={{
                        marginLeft: '12px',
                        fontSize: '0.8rem',
                        fontWeight: '600',
                        color: 'var(--accent-light)',
                        textDecoration: 'none',
                        borderBottom: '1px dashed var(--accent)',
                        paddingBottom: '2px',
                        transition: 'opacity 0.2s',
                      }}
                      onMouseOver={(e) => e.target.style.opacity = 0.8}
                      onMouseOut={(e) => e.target.style.opacity = 1}
                      title={`Visit scraped site: ${url}`}
                    >
                      Visit Site ↗
                    </a>
                  )}
                </>
              ) : 'Waiting for context...'}
            </h3>
          </div>

          <div className="top-nav-right">
            <button
              className="sidebar-toggle-btn right-toggle"
              onClick={() => setRightSidebarOpen(!rightSidebarOpen)}
              title={rightSidebarOpen ? "Collapse Sources Panel" : "Expand Sources Panel"}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                {rightSidebarOpen ? (
                  <polyline points="9 18 15 12 9 6"></polyline>
                ) : (
                  <polyline points="15 18 9 12 15 6"></polyline>
                )}
              </svg>
            </button>
          </div>
        </div>


        <div className="chat-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-logo-center">
                <img src="/logo.jpg" alt="RAG X Logo" />
              </div>
              <h2>Welcome to RAG <span className="welcome-x">X</span></h2>
            </div>
          ) : (
            <div className="messages-wrapper">
              {messages.map((msg, idx) => (
                <div key={idx} className={`message-row ${msg.role}`}>
                  <div className="avatar assistant-avatar">
                    {msg.role === 'user' ? 'U' : <img src="/logo.jpg" alt="RAG X Logo" />}
                  </div>
                  <div className="message-content">
                    <div className="bubble markdown-body">
                      {msg.role === 'assistant' ? (
                        <>
                          {msg.debug && msg.debug["LLM Provider"] && (
                            <div className="provider-badge" style={{ marginBottom: '12px', padding: '4px 8px', display: 'inline-block', background: '#2d2d2d', borderRadius: '4px', fontSize: '11px', color: '#a3a3a3', fontWeight: '600', letterSpacing: '0.5px' }}>
                              ⚡ {msg.debug["LLM Provider"].toUpperCase()}
                              {msg.debug["Fallback Activated"] && <span style={{ color: '#ef4444', marginLeft: '6px' }}>(FALLBACK)</span>}
                              {msg.debug["Cache Hit"] && <span style={{ color: '#3b82f6', marginLeft: '6px' }}>(CACHED)</span>}
                            </div>
                          )}
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />
                            }}
                          >
                            {msg.content}
                          </ReactMarkdown>
                        </>
                      ) : (
                        msg.content
                      )}
                    </div>
                    {Array.isArray(msg.sources) && msg.sources.length > 0 && (
                      <div className="citations-inline">
                        <div className="citations-header">
                          <strong>Sources cited:</strong>
                        </div>
                        <div className="citation-chips">
                          {msg.sources.map((s, i) => (
                            <div key={i} className="citation-chip">
                              <a href={s.source_url} target="_blank" rel="noopener noreferrer">
                                {s.source_title || 'Unknown Page'}
                              </a>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="message-row assistant">
                  <div className="avatar assistant-avatar">
                    <img src="/logo.jpg" alt="RAG X Logo" />
                  </div>
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
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>
              </button>
            </div>
            <div className="footer-text">
              Strictly grounded in retrieved context. Hallucinations minimized.
            </div>
          </form>
        </div>
      </main>

      {/* ─── RIGHT SIDEBAR (INDEXED SOURCES) ─── */}
      <aside className={`right-sidebar ${rightSidebarOpen ? 'open' : 'collapsed'}`}>
        <div className="right-sidebar-header">
          <h3>Indexed Sources</h3>
          {indexedPages.length > 0 && <span className="badge-small-red">{indexedPages.length} files</span>}
        </div>
        {indexedPages.length > 0 && (
          <div className="right-sidebar-search" style={{ padding: '0 20px 12px 20px', borderBottom: '1px solid var(--border)' }}>
            <input 
              type="text" 
              placeholder="Search pages..." 
              value={sourceSearchQuery} 
              onChange={(e) => setSourceSearchQuery(e.target.value)} 
              className="sidebar-search-input"
              style={{ marginBottom: 0 }}
            />
          </div>
        )}
        <div className="right-sidebar-content">
          {filteredSources.length > 0 ? (
            <ul className="sources-list">
              {filteredSources.map((page, idx) => (
                <li key={idx} title={page.url}>
                  <div className="source-title">{page.title || page.url.split('/').pop() || page.url}</div>
                  <div className="source-meta">
                    <span className="badge-small">{page.chunk_count} chunks</span>
                    <a href={page.url} target="_blank" rel="noreferrer" className="source-link">View</a>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="empty-sources-placeholder">
              <div className="placeholder-icon">📂</div>
              <p>
                {status === 'crawling' 
                  ? 'Crawling pages...' 
                  : indexedPages.length > 0 
                    ? 'No matching pages found.' 
                    : 'No indexed sources. Crawl a website in the sidebar to populate.'}
              </p>
            </div>
          )}
        </div>
      </aside>

    </div>
  );
}

export default App;
