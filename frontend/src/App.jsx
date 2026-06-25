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

  // Debug Data
  const [debugData, setDebugData] = useState(null);
  const [showDebug, setShowDebug] = useState(false);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Scroll to bottom when messages update
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Load history list on mount
  useEffect(() => {
    fetchHistoryList();
  }, []);

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
    const date = new Date(isoString + "Z"); // Ensure UTC parsing
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

  const loadSession = async (id) => {
    try {
      setStatus('crawling'); // reuse loading state
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
            try { parsedSources = JSON.parse(m.sources); } catch(e) {}
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

  return (
    <div className="app-container">
      {/* ─── LEFT SIDEBAR ─── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>WebIntel AI</h2>
          <span className="badge">Production RAG</span>
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
          )}

          {historyList.length > 0 && (
            <div className="history-section mt-4">
              <h3>Previous Chats</h3>
              <div className="history-list">
                {historyList.map(h => (
                  <div key={h.id} className={`history-item ${h.id === analysisId ? 'active' : ''}`} onClick={() => loadSession(h.id)}>
                    <div className="history-title">{h.title || h.domain || h.url}</div>
                    <div className="history-time">{formatDate(h.created_at)}</div>
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
          <h3>{domain ? `Chatting with ${domain}` : 'Waiting for context...'}</h3>
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
                    <div className="bubble markdown-body">
                      {msg.role === 'assistant' ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                      ) : (
                        msg.content
                      )}
                    </div>
                    {Array.isArray(msg.sources) && msg.sources.length > 0 && (
                      <div className="citations-inline">
                        <div className="citations-header">
                          <strong>Sources cited:</strong>
                          <button className="btn-text-small" onClick={() => toggleContext(idx)}>
                            {expandedContext[idx] ? 'Hide Context' : 'View Retrieved Context'}
                          </button>
                        </div>
                        <div className="citation-chips">
                          {msg.sources.map((s, i) => (
                            <a key={i} href={s.url} target="_blank" rel="noreferrer" className="citation-chip" title={s.url}>
                              {s.title || `Source ${s.chunk_id + 1}`}
                            </a>
                          ))}
                        </div>
                        
                        {expandedContext[idx] && (
                          <div className="retrieved-context-viewer">
                            {msg.sources.map((s, i) => (
                              <div key={i} className="context-chunk">
                                <div className="context-meta">
                                  <span className="context-title"><strong>{s.title || `Chunk ${s.chunk_id + 1}`}</strong></span>
                                  <span className="badge-small" style={{ marginLeft: "8px", marginRight: "8px" }}>Score: {s.score.toFixed(2)}</span>
                                  <a href={s.url} target="_blank" rel="noreferrer" className="context-url" title={s.url}>View Page</a>
                                </div>
                                <div className="context-text">{s.chunk_text}</div>
                              </div>
                            ))}
                          </div>
                        )}
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

    </div>
  );
}

export default App;
