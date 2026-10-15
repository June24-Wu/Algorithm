'use strict';

const app = document.getElementById('app');
const strings = {
  en: { back: 'All collections', contents: 'Contents', search: 'Search the library…', noResults: 'No matching pages or questions.', original: 'Original source', questions: 'questions', questionList: 'Questions', minutes: 'min read', source: 'View source', collapse: 'Collapse answers', expand: 'Expand answers', onPage: 'On this page', menu: 'Browse topics', copy: 'Copy', copied: 'Copied', end: 'A little more understood.', top: 'Back to top ↑', fallback: 'This page is available in English. A Chinese translation has not been added yet.', subtitle: 'Interview questions and study notes on AI infrastructure. Understand the mechanism, then follow the evidence.' },
  zh: { back: '所有目录', contents: '目录', search: '搜索题目与笔记…', noResults: '没有找到匹配的题目或页面。', original: '原始题目来源', questions: '道问题', questionList: '题目', minutes: '分钟阅读', source: '查看源文件', collapse: '收起全部答案', expand: '展开全部答案', onPage: '本页题目', menu: '浏览目录', copy: '复制', copied: '已复制', end: '每一次理解，都更进一步。', top: '回到顶部 ↑', fallback: '这个页面暂时只有英文版本，添加中文翻译后即可切换。', subtitle: '围绕 AI 基础设施的面试题与学习笔记。先理解原理，再用数据验证判断。' },
};

const icons = {
  layers: '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/>',
  code: '<path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18"/>',
  chip: '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3"/>',
  file: '<path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8l-5-5Z"/><path d="M14 3v5h5M8 12h8m-8 4h6"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
};

let catalog;
let language = readPreference('atlas-language', 'en');
if (!strings[language]) language = 'en';
let currentKey = '';
let renderSequence = 0;
let toastTimer;
const documents = new Map();

function readPreference(key, fallback) { try { return localStorage.getItem(key) || fallback; } catch (_) { return fallback; } }
function savePreference(key, value) { try { localStorage.setItem(key, value); } catch (_) {} }
function escapeHTML(value) { return String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char])); }
function icon(name) { return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.file}</svg>`; }
function route(collection, doc, question) { const values = new URLSearchParams({ collection }); if (doc) values.set('doc', doc); if (question) values.set('question', question); return '#' + values.toString(); }
function state() { return new URLSearchParams(location.hash.slice(1)); }
function localized(value, lang = language) { return value?.[lang] || value?.en || ''; }
function textFor(collection) { return strings[collection?.id === 'ai-infra' ? language : 'en']; }
function questionTitle(title) { return title.replace(/^\d+\.\s*/, ''); }
function ordinal(number) { return String(number).padStart(2, '0'); }

async function getJSON(path) {
  const response = await fetch(new URL(path, document.baseURI));
  if (!response.ok) throw new Error(`Could not load ${path} (HTTP ${response.status}).`);
  return response.json();
}

function searchField(label = strings.en.search) {
  return `<div class="search-wrap"><label class="search-field">${icon('search')}<input type="search" data-search placeholder="${escapeHTML(label)}" aria-label="${escapeHTML(label)}" autocomplete="off"><kbd aria-hidden="true">/</kbd></label><div class="search-results" role="region" aria-label="Search results" hidden></div></div>`;
}

function renderHome() {
  currentKey = '';
  document.documentElement.lang = 'en';
  document.title = 'Algorithm Atlas · A study library';
  document.getElementById('language-switch').hidden = true;
  document.getElementById('reading-progress').style.width = '0';
  const questions = catalog.documents.reduce((count, doc) => count + doc.questions, 0);
  const recent = [...catalog.documents].sort((a, b) => b.questions - a.questions).slice(0, 3);
  app.innerHTML = `<main id="main" class="home">
    <section class="hero" aria-labelledby="hero-title">
      <div><div class="eyebrow"><span class="status-dot"></span> Notes for a deeper understanding</div>
        <h1 id="hero-title">Understand the systems.<br><em>Build the intuition.</em></h1>
        <p>A growing collection of interview answers, from-scratch implementations, and GPU experiments. One concept at a time.</p>
      </div>
      <aside class="hero-aside" aria-label="Illustration of the Roofline performance model"><div class="diagram-label"><span>A NOTE ON PERFORMANCE</span><span>FIG. 01</span></div>
        <svg class="roofline" viewBox="0 0 270 135" aria-hidden="true"><path class="grid" fill="none" d="M25 10v100h230M25 85h230M25 60h230M25 35h230M70 10v100M115 10v100M160 10v100M205 10v100M250 10v100"/><path class="roof" d="M26 109 130 33h122"/><path d="M119 33v77" stroke="currentColor" opacity=".25" stroke-dasharray="3 4"/><circle cx="90" cy="77" r="4" fill="currentColor" opacity=".35"/><circle cx="195" cy="40" r="4" fill="currentColor" opacity=".6"/><text x="24" y="126">MEMORY</text><text x="210" y="126">COMPUTE</text><text x="144" y="22">THE ROOFLINE</text></svg>
        <div class="diagram-caption">Find the limit. Understand the tradeoff.</div>
      </aside>
    </section>
    <div class="library-tools"><h2 class="section-heading">Explore the collections <span>${ordinal(catalog.collections.length)}</span></h2>${searchField()}</div>
    <section class="collection-grid" aria-label="Study collections">${catalog.collections.map((collection, index) => `<a class="collection-card ${escapeHTML(collection.color)}" href="${route(collection.id)}">
      <div class="card-top"><div class="collection-icon">${icon(collection.icon)}</div><span class="card-number">${ordinal(index + 1)} / ${ordinal(catalog.collections.length)}</span></div>
      <div class="card-eyebrow">${escapeHTML(collection.eyebrow)}</div><h2>${escapeHTML(collection.title)}</h2><p>${escapeHTML(collection.description)}</p>
      <div class="card-bottom"><span>${collection.documents} ${collection.id === 'ai-infra' ? 'guide' : 'practice files'}${collection.questions ? ` <span class="dot">·</span> ${collection.questions} questions` : ''}</span><span class="arrow" aria-hidden="true">→</span></div>
    </a>`).join('')}</section>
    <div class="library-note"><span>${ordinal(catalog.collections.length)} collections <span class="dot">/</span> ${ordinal(catalog.documents.length)} pages <span class="dot">/</span> ${ordinal(questions)} interview questions</span><span class="right-note">Curated from real questions. Built for understanding.</span></div>
    <section class="recent-section"><div class="recent-heading"><h2 class="section-heading">A few places to begin</h2><span>THE STUDY DESK</span></div>${recent.map((doc, index) => `<a class="reading-row" href="${route(doc.collection, doc.id)}"><span class="row-index">${ordinal(index + 1)}</span><div><div class="reading-title">${escapeHTML(doc.title.en)}</div><div class="reading-category">${escapeHTML(catalog.collections.find(item => item.id === doc.collection).title)}</div></div><span class="reading-badge">${doc.questions ? `${doc.questions} questions · EN / 中文` : 'Python · Practice'}</span></a>`).join('')}</section>
  </main>`;
  attachSearch();
  window.scrollTo(0, 0);
}

function renderTree(node, selected, collection, lang) {
  return node.children.map(child => {
    const title = localized(child.title, lang);
    if (child.children.length) {
      return `<details class="tree-group" open><summary><span class="chevron" aria-hidden="true">›</span>${escapeHTML(title)}</summary><div class="tree-children">${child.documentId ? treeLink(child.documentId, title, selected, collection) : ''}${renderTree(child, selected, collection, lang)}</div></details>`;
    }
    return child.documentId ? treeLink(child.documentId, title, selected, collection) : '';
  }).join('');
}

function treeLink(identifier, title, selected, collection) {
  return `<a class="tree-link ${identifier === selected ? 'active' : ''}" ${identifier === selected ? 'aria-current="page"' : ''} href="${route(collection.id, identifier)}"><span class="page-icon">${icon('file')}</span><span>${escapeHTML(title)}</span></a>`;
}

async function render() {
  if (!catalog) return;
  const sequence = ++renderSequence;
  closeMenu();
  const params = state();
  const collectionId = params.get('collection');
  if (!collectionId) { renderHome(); return; }
  const collection = catalog.collections.find(item => item.id === collectionId);
  if (!collection) { renderError('This collection could not be found. Return to the library to choose an available collection.'); return; }
  const identifier = params.get('doc') || collection.defaultDocument;
  const metadata = catalog.documents.find(doc => doc.id === identifier && doc.collection === collection.id);
  if (!metadata) { renderError('This page could not be found. Its source may have been moved or removed.'); return; }
  const lang = collection.id === 'ai-infra' ? language : 'en';
  const key = identifier + ':' + lang;
  if (key === currentKey) {
    if (params.get('question')) scrollQuestion(params.get('question'));
    else window.scrollTo(0, 0);
    return;
  }
  const labels = textFor(collection);
  document.getElementById('language-switch').hidden = collection.id !== 'ai-infra';
  document.querySelectorAll('[data-language]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.language === language)));
  try {
    if (!documents.has(identifier)) documents.set(identifier, getJSON('data/documents/' + metadata.file));
    const payload = await documents.get(identifier);
    if (sequence !== renderSequence) return;
    const version = payload.versions[lang] || payload.versions.en;
    const translated = Boolean(payload.versions[lang]);
    currentKey = key;
    document.documentElement.lang = translated ? lang : 'en';
    document.title = `${version.title} · Algorithm Atlas`;
    const questionCount = version.sections.length;
    app.innerHTML = `<div class="mobile-toolbar"><button type="button" id="menu-toggle" aria-expanded="false" aria-controls="sidebar">${icon('menu')}${labels.menu}</button><span>${escapeHTML(collection.title)}</span></div>
      <div class="workspace"><button type="button" class="sidebar-backdrop" aria-label="Close navigation"></button>
      <aside class="sidebar" id="sidebar" aria-label="${escapeHTML(labels.contents)}"><a class="back-link" href="#">← ${labels.back}</a><div class="sidebar-collection"><div class="collection-icon">${icon(collection.icon)}</div>${escapeHTML(collection.title)}</div><p class="sidebar-path">${escapeHTML(collection.directory)}/</p>
        ${searchField(labels.search)}<div class="sidebar-label">${labels.contents}</div><nav aria-label="Topic navigation">${renderTree(collection.tree, identifier, collection, lang)}</nav><a class="sidebar-source" href="${escapeHTML(collection.source)}" target="_blank" rel="noopener noreferrer">${labels.original}<span aria-hidden="true">↗</span></a></aside>
      <main id="main" class="reader-shell"><div class="breadcrumb"><a href="#">Library</a><span class="slash">/</span><a href="${route(collection.id)}">${escapeHTML(collection.title)}</a><span class="slash">/</span><span>${escapeHTML(localized(metadata.title, lang))}</span></div>
        <div class="reader-columns"><article class="article"><header class="article-header"><div class="article-kicker">${escapeHTML(collection.eyebrow)} ${metadata.kind === 'questions' ? ' / ' + ordinal(catalog.documents.filter(doc => doc.collection === collection.id).findIndex(doc => doc.id === identifier) + 1) : ' / PYTHON'}</div><h1>${escapeHTML(version.title)}</h1><p class="article-subtitle">${collection.id === 'ai-infra' ? labels.subtitle : escapeHTML(collection.description)}</p>
          <div class="article-metadata">${questionCount ? `<span>${questionCount} ${labels.questions}</span>` : '<span>Python source</span>'}<span>${version.minutes} ${labels.minutes}</span><span>${metadata.languages.length > 1 ? 'EN / 中文' : 'EN'}</span><a href="${escapeHTML(metadata.sourceUrl)}" target="_blank" rel="noopener noreferrer">${labels.source} ↗</a></div></header>
          ${translated ? '' : `<div class="translation-notice">${labels.fallback}</div>`}<div class="article-intro prose">${version.intro}</div>
          ${questionCount ? `<div class="question-toolbar"><span>${labels.questionList} / ${ordinal(questionCount)}</span><button type="button" id="toggle-answers">${labels.collapse}</button></div>${version.sections.map((section, index) => `<details class="question-card" id="${escapeHTML(section.id)}" open><summary><span class="question-number">${ordinal(index + 1)}</span><h2 class="question-title">${escapeHTML(questionTitle(section.title))}</h2><span class="question-chevron" aria-hidden="true">⌄</span></summary><div class="answer prose">${section.html}</div></details>`).join('')}` : ''}
          <div class="reader-end"><span>${labels.end}</span><a href="#" id="back-to-top">${labels.top}</a>${metadata.kind === 'practice' ? `<a href="${escapeHTML(metadata.downloadUrl)}" download>Download .py ↓</a>` : ''}</div>
        </article><aside class="toc" aria-label="${escapeHTML(labels.onPage)}"><div class="toc-label">${questionCount ? labels.onPage : 'Source file'}</div>${version.sections.map(section => `<a href="${route(collection.id, identifier, section.id)}" data-question="${escapeHTML(section.id)}">${escapeHTML(section.title)}</a>`).join('')}<a class="toc-source" href="${escapeHTML(metadata.sourceUrl)}" target="_blank" rel="noopener noreferrer">${labels.source} ↗</a></aside></div>
      </main></div>`;
    attachReader(labels);
    attachSearch();
    if (params.get('question')) requestAnimationFrame(() => scrollQuestion(params.get('question')));
    else window.scrollTo(0, 0);
    updateProgress();
  } catch (error) {
    documents.delete(identifier);
    if (sequence === renderSequence) renderError(error.message);
  }
}

function attachReader(labels) {
  app.querySelectorAll('.prose table').forEach(table => { const wrapper = document.createElement('div'); wrapper.className = 'table-scroll'; table.before(wrapper); wrapper.append(table); });
  app.querySelectorAll('.prose pre').forEach(pre => {
    let wrapper = pre.parentElement;
    if (!wrapper.classList.contains('codehilite')) { wrapper = document.createElement('div'); wrapper.className = 'plain-code-wrap'; pre.before(wrapper); wrapper.append(pre); }
    const button = document.createElement('button'); button.type = 'button'; button.className = 'copy-code'; button.textContent = labels.copy; button.setAttribute('aria-label', 'Copy code');
    button.addEventListener('click', async () => {
      try {
        const code = pre.textContent;
        if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(code);
        else { const field = document.createElement('textarea'); field.value = code; field.style.position = 'fixed'; field.style.opacity = '0'; document.body.append(field); field.select(); const copied = document.execCommand('copy'); field.remove(); if (!copied) throw new Error('Clipboard unavailable'); }
        button.textContent = labels.copied; showToast(labels.copied); setTimeout(() => { button.textContent = labels.copy; }, 1800);
      } catch (_) { showToast(language === 'zh' ? '请选中代码后手动复制。' : 'Select the code and copy it manually.'); }
    });
    wrapper.append(button);
  });
  document.getElementById('toggle-answers')?.addEventListener('click', event => {
    const cards = [...app.querySelectorAll('.question-card')];
    const expand = cards.some(card => !card.open);
    cards.forEach(card => { card.open = expand; });
    event.currentTarget.textContent = expand ? labels.collapse : labels.expand;
    updateProgress();
  });
  app.querySelectorAll('.question-card').forEach(card => card.addEventListener('toggle', updateProgress));
  document.getElementById('back-to-top')?.addEventListener('click', event => { event.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }); });
  document.getElementById('menu-toggle')?.addEventListener('click', () => { const opened = document.body.classList.toggle('nav-open'); document.getElementById('menu-toggle').setAttribute('aria-expanded', String(opened)); if (opened) document.querySelector('.sidebar .tree-link.active')?.focus(); });
  document.querySelector('.sidebar-backdrop')?.addEventListener('click', closeMenu);
}

function closeMenu() { document.body.classList.remove('nav-open'); document.getElementById('menu-toggle')?.setAttribute('aria-expanded', 'false'); }
function scrollQuestion(identifier) { if (!identifier) return; const target = document.getElementById(identifier); if (target?.classList.contains('question-card')) { target.open = true; target.scrollIntoView({ block: 'start' }); } }

function attachSearch() {
  app.querySelectorAll('[data-search]').forEach(input => {
    const results = input.closest('.search-wrap').querySelector('.search-results');
    input.addEventListener('input', () => {
      const query = input.value.trim().toLocaleLowerCase();
      if (!query) { results.hidden = true; return; }
      const matches = [];
      for (const doc of catalog.documents) {
        const collection = catalog.collections.find(item => item.id === doc.collection);
        if ([...Object.values(doc.title), collection.title, doc.relative].some(value => value.toLocaleLowerCase().includes(query))) matches.push({ title: localized(doc.title, doc.collection === 'ai-infra' ? language : 'en'), doc, collection });
        for (let index = 0; index < (doc.outline.en || []).length; index++) {
          const outlines = Object.values(doc.outline).map(outline => outline[index]).filter(Boolean);
          if (outlines.some(section => section.title.toLocaleLowerCase().includes(query))) {
            const section = (doc.outline[language] || doc.outline.en)[index];
            matches.push({ title: section.title, question: section.id, doc, collection });
          }
        }
      }
      results.innerHTML = matches.length ? matches.slice(0, 9).map(match => `<a href="${route(match.doc.collection, match.doc.id, match.question)}"><strong>${escapeHTML(match.title)}</strong><small>${escapeHTML(match.collection.title)} ${match.question ? ' / ' + escapeHTML(match.doc.title.en) : ''}</small></a>`).join('') : `<div class="empty-search">${strings[language].noResults}</div>`;
      results.hidden = false;
    });
    input.addEventListener('keydown', event => { if (event.key === 'Escape') { results.hidden = true; input.blur(); } if (event.key === 'ArrowDown') { event.preventDefault(); results.querySelector('a')?.focus(); } });
  });
}

function updateProgress() {
  const article = document.querySelector('.article');
  if (!article) return;
  const box = article.getBoundingClientRect();
  const progress = Math.min(1, Math.max(0, (110 - box.top) / Math.max(1, box.height - window.innerHeight + 110)));
  document.getElementById('reading-progress').style.width = (progress * 100) + '%';
  const sections = [...document.querySelectorAll('.question-card')];
  const current = sections.filter(section => section.getBoundingClientRect().top < 180).pop() || sections[0];
  document.querySelectorAll('.toc [data-question]').forEach(link => link.classList.toggle('current', link.dataset.question === current?.id));
}

function showToast(message) { const element = document.getElementById('toast'); element.textContent = message; element.classList.add('visible'); clearTimeout(toastTimer); toastTimer = setTimeout(() => element.classList.remove('visible'), 1800); }
function renderError(message) { currentKey = ''; app.innerHTML = `<main id="main" class="loading-page error-page" role="alert"><div class="eyebrow" style="justify-content:center">A PAGE IS MISSING</div><h1>Let's find your place again.</h1><p>${escapeHTML(message)}</p><p>If you are running locally, start <code>python study_site/backend/serve.py</code> and open the HTTP address.</p><a class="primary-button" href="#">Back to the library</a> <button type="button" id="retry">Try again</button></main>`; document.getElementById('retry').addEventListener('click', () => location.reload()); }

function updateThemeToggle() {
  const label = document.documentElement.dataset.theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme';
  const button = document.getElementById('theme-toggle');
  button.setAttribute('aria-label', label);
  button.title = label;
}
document.getElementById('theme-toggle').addEventListener('click', () => {
  const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  savePreference('atlas-theme', theme);
  updateThemeToggle();
});
const fontSizeControl = document.getElementById('font-size');
fontSizeControl.value = document.documentElement.dataset.fontSize || '18';
fontSizeControl.addEventListener('change', () => {
  document.documentElement.dataset.fontSize = fontSizeControl.value;
  savePreference('atlas-font-size', fontSizeControl.value);
  updateProgress();
});
document.querySelectorAll('[data-language]').forEach(button => button.addEventListener('click', () => { language = button.dataset.language; savePreference('atlas-language', language); render(); }));
document.querySelector('.skip-link').addEventListener('click', event => { event.preventDefault(); const main = document.getElementById('main'); if (main) { main.setAttribute('tabindex', '-1'); main.focus(); main.scrollIntoView({ block: 'start' }); } });
document.addEventListener('click', event => {
  const anchor = event.target.closest('a');
  if (anchor?.hash?.startsWith('#collection=') && anchor.origin === location.origin && anchor.pathname === location.pathname) {
    closeMenu();
    app.querySelectorAll('.search-results').forEach(results => { results.hidden = true; });
    if (anchor.hash === location.hash) { event.preventDefault(); scrollQuestion(state().get('question')); }
  }
  document.querySelectorAll('.search-wrap').forEach(wrapper => { if (!wrapper.contains(event.target)) wrapper.querySelector('.search-results').hidden = true; });
});
document.addEventListener('keydown', event => { if (event.key === '/' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) { event.preventDefault(); document.querySelector('[data-search]')?.focus(); } if (event.key === 'Escape') { const wasOpen = document.body.classList.contains('nav-open'); closeMenu(); if (wasOpen) document.getElementById('menu-toggle')?.focus(); } });
window.addEventListener('hashchange', render);
window.addEventListener('scroll', updateProgress, { passive: true });
updateThemeToggle();
getJSON('data/catalog.json').then(data => { catalog = data; document.getElementById('collection-count').textContent = ordinal(data.collections.length); render(); }).catch(error => renderError(error.message));
