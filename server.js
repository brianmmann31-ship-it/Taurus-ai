require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const session = require('express-session');
const FileStore = require('session-file-store')(session);
const fs = require('fs');
const path = require('path');
const fetch = (...args) => import('node-fetch').then(({default: f}) => f(...args));

const app = express();
const PORT = process.env.PORT || 3001;

// ── Directories ──────────────────────────────────────────────────────────────
const CHATS_DIR = path.join(__dirname, 'data', 'chats');
const BUILDS_DIR = path.join(__dirname, 'data', 'builds');
const MEMORY_DIR = path.join(__dirname, 'data', 'memory');
const FILES_DIR = path.join(__dirname, 'data', 'files');
[CHATS_DIR, BUILDS_DIR, MEMORY_DIR, FILES_DIR].forEach(d => fs.mkdirSync(d, { recursive: true }));

// ── Security ──────────────────────────────────────────────────────────────────
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: false }));
app.use(express.json({ limit: '10mb' }));

const loginLimiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 10, message: { error: 'Too many attempts' } });
const apiLimiter = rateLimit({ windowMs: 60 * 1000, max: 60 });

app.use(session({
  store: new FileStore({ path: path.join(__dirname, 'data', 'sessions'), retries: 1 }),
  secret: process.env.SESSION_SECRET || 'taurus-iron-bull-secret-2026',
  resave: false,
  saveUninitialized: false,
  cookie: { secure: false, httpOnly: true, maxAge: 365 * 24 * 60 * 60 * 1000 }
}));

// ── Auth ──────────────────────────────────────────────────────────────────────
const PASSWORD = process.env.TAURUS_PASSWORD || 'gx2';

function requireAuth(req, res, next) {
  if (req.session && req.session.authenticated) return next();
  res.status(401).json({ error: 'Unauthorized' });
}

app.post('/api/login', loginLimiter, (req, res) => {
  const { password } = req.body;
  if (password === PASSWORD) {
    req.session.authenticated = true;
    req.session.save();
    res.json({ success: true });
  } else {
    res.status(401).json({ error: 'Invalid password' });
  }
});

app.post('/api/logout', (req, res) => {
  req.session.destroy();
  res.json({ success: true });
});

app.get('/api/auth/check', (req, res) => {
  res.json({ authenticated: !!(req.session && req.session.authenticated) });
});

// ── Key & Model Config ────────────────────────────────────────────────────────
const KEYS = {
  cerebras: [process.env.CEREBRAS_KEY, process.env.CEREBRAS_KEY2].filter(Boolean),
  groq: [process.env.GROQ_KEY1, process.env.GROQ_KEY2].filter(Boolean),
  gemini: [process.env.GEMINI_KEY].filter(Boolean),
  sambanova: [process.env.SAMBANOVA_KEY].filter(Boolean),
  aiml: [process.env.AIML_KEY].filter(Boolean),
  openrouter: [process.env.OPENROUTER_KEY].filter(Boolean),
  unfiltered: [process.env.UNFILTERED_KEY1, process.env.UNFILTERED_KEY2].filter(Boolean),
  huggingface: [process.env.HUGGINGFACE_KEY].filter(Boolean),
  deepseek: [process.env.DEEPSEEK_KEY].filter(Boolean),
};

const keyUsage = {};
let buildMode = false;
let freeKeyIndex = { cerebras: 0, groq: 0, unfiltered: 0 };

function getStatus(provider) {
  const usage = keyUsage[provider] || { calls: 0, errors: 0 };
  const keys = KEYS[provider] || [];
  if (!keys.length) return 'missing';
  if (usage.errors >= 3) return 'limit';
  if (usage.calls > 80) return 'near';
  return 'good';
}

app.get('/api/keys/status', requireAuth, (req, res) => {
  const status = {};
  Object.keys(KEYS).forEach(p => {
    status[p] = {
      status: getStatus(p),
      count: (KEYS[p] || []).length,
      calls: (keyUsage[p] || {}).calls || 0
    };
  });
  status.buildMode = buildMode;
  res.json(status);
});

app.post('/api/build/toggle', requireAuth, (req, res) => {
  buildMode = req.body.active;
  res.json({ buildMode });
});

// ── Model Router ──────────────────────────────────────────────────────────────
async function callFreeModel(messages, systemPrompt) {
  // Rotate through free providers
  const providers = ['cerebras', 'groq', 'gemini', 'sambanova', 'aiml', 'openrouter'];
  
  for (const provider of providers) {
    const keys = KEYS[provider];
    if (!keys || !keys.length) continue;
    if (getStatus(provider) === 'limit') continue;

    try {
      if (!keyUsage[provider]) keyUsage[provider] = { calls: 0, errors: 0 };
      
      let result;
      if (provider === 'groq') {
        result = await callGroq(messages, systemPrompt, keys[freeKeyIndex.groq % keys.length]);
        freeKeyIndex.groq++;
      } else if (provider === 'cerebras') {
        result = await callCerebras(messages, systemPrompt, keys[freeKeyIndex.cerebras % keys.length]);
        freeKeyIndex.cerebras++;
      } else if (provider === 'gemini') {
        result = await callGemini(messages, systemPrompt, keys[0]);
      } else if (provider === 'openrouter') {
        result = await callOpenRouter(messages, systemPrompt, keys[0]);
      } else if (provider === 'aiml') {
        result = await callAIML(messages, systemPrompt, keys[0]);
      } else if (provider === 'sambanova') {
        result = await callSambanova(messages, systemPrompt, keys[0]);
      }

      if (result) {
        keyUsage[provider].calls++;
        return { text: result, provider };
      }
    } catch (e) {
      if (keyUsage[provider]) keyUsage[provider].errors++;
      continue;
    }
  }
  throw new Error('All free providers failed');
}

async function callBuildModel(messages, systemPrompt, forceR1 = false) {
  const key = KEYS.deepseek[0];
  if (!key) throw new Error('No DeepSeek key');
  if (!keyUsage.deepseek) keyUsage.deepseek = { calls: 0, errors: 0 };

  // Try V3 first unless R1 forced
  const model = forceR1 ? 'deepseek-reasoner' : 'deepseek-chat';
  
  try {
    const result = await callDeepSeek(messages, systemPrompt, key, model);
    keyUsage.deepseek.calls++;
    return { text: result, provider: 'deepseek', model };
  } catch (e) {
    // V3 failed — step up to R1
    if (!forceR1) {
      try {
        const result = await callDeepSeek(messages, systemPrompt, key, 'deepseek-reasoner');
        keyUsage.deepseek.calls++;
        return { text: result, provider: 'deepseek', model: 'deepseek-reasoner' };
      } catch (e2) {
        keyUsage.deepseek.errors++;
        throw e2;
      }
    }
    keyUsage.deepseek.errors++;
    throw e;
  }
}

// ── Provider Callers ──────────────────────────────────────────────────────────
async function callGroq(messages, system, key) {
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'llama-3.3-70b-versatile', messages: system ? [{ role: 'system', content: system }, ...messages] : messages, max_tokens: 4096 })
  });
  const d = await res.json();
  if (!res.ok) throw new Error(d.error?.message || 'Groq error');
  return d.choices[0].message.content;
}

async function callCerebras(messages, system, key) {
  const res = await fetch('https://api.cerebras.ai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'llama-3.3-70b', messages: system ? [{ role: 'system', content: system }, ...messages] : messages, max_tokens: 4096 })
  });
  const d = await res.json();
  if (!res.ok) throw new Error(d.error?.message || 'Cerebras error');
  return d.choices[0].message.content;
}

async function callGemini(messages, system, key) {
  const contents = messages.map(m => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] }));
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${key}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ contents, systemInstruction: system ? { parts: [{ text: system }] } : undefined, generationConfig: { maxOutputTokens: 4096 } })
  });
  const d = await res.json();
  if (!res.ok) throw new Error(d.error?.message || 'Gemini error');
  return d.candidates[0].content.parts[0].text;
}

async function callOpenRouter(messages, system, key) {
  const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'mistralai/mistral-7b-instruct:free', messages: system ? [{ role: 'system', content: system }, ...messages] : messages, max_tokens: 4096 })
  });
  const d = await res.json();
  if (!res.ok) throw new Error(d.error?.message || 'OpenRouter error');
  return d.choices[0].message.content;
}

async function callAIML(messages, system, key) {
  const res = await fetch('https://api.aimlapi.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'mistralai/Mistral-7B-Instruct-v0.2', messages: system ? [{ role: 'system', content: system }, ...messages] : messages, max_tokens: 4096 })
  });
  const d = await res.json();
  if (!res.ok) throw new Error(d.error?.message || 'AIML error');
  return d.choices[0].message.content;
}

async function callSambanova(messages, system, key) {
  const res = await fetch('https://api.sambanova.ai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'Meta-Llama-3.3-70B-Instruct', messages: system ? [{ role: 'system', content: system }, ...messages] : messages, max_tokens: 4096 })
  });
  const d = await res.json();
  if (!res.ok) throw new Error(d.error?.message || 'Sambanova error');
  return d.choices[0].message.content;
}

async function callDeepSeek(messages, system, key, model) {
  const res = await fetch('https://api.deepseek.com/chat/completions', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages: system ? [{ role: 'system', content: system }, ...messages] : messages, max_tokens: 8192 })
  });
  const d = await res.json();
  if (!res.ok) throw new Error(d.error?.message || 'DeepSeek error');
  return d.choices[0].message.content;
}

// ── Web Search ────────────────────────────────────────────────────────────────
async function webSearch(query) {
  const key = KEYS.groq[0];
  if (!key) return null;
  try {
    const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        messages: [{ role: 'user', content: `Search and answer: ${query}` }],
        max_tokens: 2048,
        tools: [{ type: 'function', function: { name: 'web_search', description: 'Search the web', parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } } }]
      })
    });
    const d = await res.json();
    return d.choices[0].message.content || 'Search completed';
  } catch (e) {
    return null;
  }
}

// ── Chat API ──────────────────────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are Taurus AI — a top-tier code writer, problem solver, designer, and reasoning engine.

SERVER CONTEXT (permanent):
- DO Server: 157.230.191.50, Ubuntu 24.04, 1GB RAM
- Taurus: /var/www/taurus-app/ port 3001
- Scripture Deep: /var/www/scripture/ port 3000
- nginx: / → 3001, /bible → 3000
- Deploy Taurus: pm2 restart taurus
- Deploy Scripture: pm2 restart scripture
- GitHub: brianmman31-ship-it/Taurus-ai

CODE RULES:
- Complete working files only. No examples. No placeholders. No truncation.
- File output first, then exact DO deploy commands after.
- Dark fintech UI always. Mobile-first always.
- 1GB RAM aware — optimize for low memory.
- Every build includes: save file → push to GitHub → pull to server → pm2 restart

DESIGN RULES:
- Top tier agency quality design
- Premium, intentional, memorable
- Mobile-first, Android-optimized

REASONING:
- For complex problems, plan first then build
- Proactively spot issues the user didn't ask about
- Surface only critical findings with exact fix commands`;

app.post('/api/chat', requireAuth, apiLimiter, async (req, res) => {
  const { messages, forceR1, searchEnabled } = req.body;
  if (!messages || !messages.length) return res.status(400).json({ error: 'No messages' });

  try {
    // Auto web search detection
    const lastMsg = messages[messages.length - 1].content.toLowerCase();
    const needsSearch = searchEnabled && (lastMsg.includes('latest') || lastMsg.includes('current') || lastMsg.includes('version') || lastMsg.includes('today') || lastMsg.includes('news') || lastMsg.includes('search'));

    let searchContext = '';
    if (needsSearch) {
      const searchResult = await webSearch(messages[messages.length - 1].content);
      if (searchResult) searchContext = `\n\nWeb search result: ${searchResult}`;
    }

    const augmentedMessages = searchContext
      ? [...messages.slice(0, -1), { role: 'user', content: messages[messages.length - 1].content + searchContext }]
      : messages;

    let result;
    if (buildMode) {
      result = await callBuildModel(augmentedMessages, SYSTEM_PROMPT, forceR1);
    } else {
      result = await callFreeModel(augmentedMessages, SYSTEM_PROMPT);
    }

    res.json({ text: result.text, provider: result.provider, model: result.model || null, buildMode });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ── Chat Storage ──────────────────────────────────────────────────────────────
app.post('/api/chats/save', requireAuth, (req, res) => {
  const { name, messages } = req.body;
  const id = Date.now().toString();
  const chat = { id, name: name || `Chat ${new Date().toLocaleDateString()}`, messages, savedAt: new Date().toISOString() };
  fs.writeFileSync(path.join(CHATS_DIR, `${id}.json`), JSON.stringify(chat));
  res.json({ success: true, id });
});

app.get('/api/chats', requireAuth, (req, res) => {
  const files = fs.readdirSync(CHATS_DIR).filter(f => f.endsWith('.json'));
  const chats = files.map(f => {
    const c = JSON.parse(fs.readFileSync(path.join(CHATS_DIR, f)));
    return { id: c.id, name: c.name, savedAt: c.savedAt, messageCount: c.messages.length };
  }).sort((a, b) => new Date(b.savedAt) - new Date(a.savedAt));
  res.json(chats);
});

app.get('/api/chats/:id', requireAuth, (req, res) => {
  const file = path.join(CHATS_DIR, `${req.params.id}.json`);
  if (!fs.existsSync(file)) return res.status(404).json({ error: 'Not found' });
  res.json(JSON.parse(fs.readFileSync(file)));
});

app.delete('/api/chats/:id', requireAuth, (req, res) => {
  const file = path.join(CHATS_DIR, `${req.params.id}.json`);
  if (fs.existsSync(file)) fs.unlinkSync(file);
  res.json({ success: true });
});

// ── Build History ─────────────────────────────────────────────────────────────
app.post('/api/builds/save', requireAuth, (req, res) => {
  const { title, files, commands } = req.body;
  const id = Date.now().toString();
  const build = { id, title, files, commands, builtAt: new Date().toISOString() };
  fs.writeFileSync(path.join(BUILDS_DIR, `${id}.json`), JSON.stringify(build));
  res.json({ success: true, id });
});

app.get('/api/builds', requireAuth, (req, res) => {
  const files = fs.readdirSync(BUILDS_DIR).filter(f => f.endsWith('.json'));
  const builds = files.map(f => {
    const b = JSON.parse(fs.readFileSync(path.join(BUILDS_DIR, f)));
    return { id: b.id, title: b.title, builtAt: b.builtAt };
  }).sort((a, b) => new Date(b.builtAt) - new Date(a.builtAt));
  res.json(builds);
});

// ── File Manager ──────────────────────────────────────────────────────────────
app.post('/api/files/save', requireAuth, (req, res) => {
  const { name, content } = req.body;
  const safeName = path.basename(name);
  fs.writeFileSync(path.join(FILES_DIR, safeName), content);
  res.json({ success: true });
});

app.get('/api/files', requireAuth, (req, res) => {
  const files = fs.readdirSync(FILES_DIR).map(f => {
    const stat = fs.statSync(path.join(FILES_DIR, f));
    return { name: f, size: stat.size, modified: stat.mtime };
  });
  res.json(files);
});

app.get('/api/files/:name', requireAuth, (req, res) => {
  const file = path.join(FILES_DIR, path.basename(req.params.name));
  if (!fs.existsSync(file)) return res.status(404).json({ error: 'Not found' });
  res.json({ name: req.params.name, content: fs.readFileSync(file, 'utf8') });
});

// ── Memory ────────────────────────────────────────────────────────────────────
app.post('/api/memory/save', requireAuth, (req, res) => {
  const { key, value } = req.body;
  const mem = getMemory();
  mem[key] = { value, savedAt: new Date().toISOString() };
  fs.writeFileSync(path.join(MEMORY_DIR, 'memory.json'), JSON.stringify(mem));
  res.json({ success: true });
});

app.get('/api/memory', requireAuth, (req, res) => {
  res.json(getMemory());
});

function getMemory() {
  const file = path.join(MEMORY_DIR, 'memory.json');
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file)) : {};
}

// ── Server Health ─────────────────────────────────────────────────────────────
app.get('/api/health', requireAuth, async (req, res) => {
  const { execSync } = require('child_process');
  let pm2Status = {};
  try {
    const out = execSync('pm2 jlist', { timeout: 5000 }).toString();
    const list = JSON.parse(out);
    list.forEach(p => { pm2Status[p.name] = { status: p.pm2_env.status, memory: Math.round(p.monit.memory / 1024 / 1024) + 'MB', uptime: p.pm2_env.pm_uptime }; });
  } catch (e) {
    pm2Status = { error: 'pm2 not available' };
  }
  res.json({ pm2: pm2Status, server: { uptime: process.uptime(), memory: process.memoryUsage(), port: PORT } });
});

// ── TTS ───────────────────────────────────────────────────────────────────────
app.post('/api/tts', requireAuth, async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).json({ error: 'No text' });
  const key = process.env.ELEVENLABS_KEY;
  const voice = process.env.ELEVENLABS_VOICE || '6FiCmD8eY5VyjOdG5Zjk';
  if (!key) return res.status(500).json({ error: 'No ElevenLabs key' });
  try {
    const r = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voice}`, {
      method: 'POST',
      headers: { 'xi-api-key': key, 'Content-Type': 'application/json', 'Accept': 'audio/mpeg' },
      body: JSON.stringify({ text: text.slice(0, 500), model_id: 'eleven_monolingual_v1', voice_settings: { stability: 0.4, similarity_boost: 0.8, style: 0.6, use_speaker_boost: true } })
    });
    if (!r.ok) { const e = await r.json(); return res.status(500).json({ error: e.detail?.message || 'TTS failed' }); }
    res.setHeader('Content-Type', 'audio/mpeg');
    r.body.pipe(res);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ── Static ────────────────────────────────────────────────────────────────────
app.use(express.static(path.join(__dirname, 'public')));
app.get('*', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

app.listen(PORT, () => console.log(`Taurus AI running on port ${PORT}`));
