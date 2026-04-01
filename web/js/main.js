/**
 * obeaver docs — JavaScript
 * Theme toggle, search, sidebar, copy buttons, scroll-to-top
 */
(function () {
  'use strict';

  // ── Theme Toggle ──────────────────────────────────────────
  const themeToggle = document.getElementById('theme-toggle');
  const html = document.documentElement;

  function getPreferredTheme() {
    const stored = localStorage.getItem('obeaver-theme');
    if (stored) return stored;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function setTheme(theme) {
    html.setAttribute('data-theme', theme);
    localStorage.setItem('obeaver-theme', theme);
    if (themeToggle) {
      themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
  }

  setTheme(getPreferredTheme());

  if (themeToggle) {
    themeToggle.addEventListener('click', function () {
      const current = html.getAttribute('data-theme');
      setTheme(current === 'dark' ? 'light' : 'dark');
    });
  }

  // ── Language Toggle ────────────────────────────────────────
  var langToggle = document.getElementById('lang-toggle');
  function getPreferredLang() {
    var stored = localStorage.getItem('obeaver-lang');
    if (stored) return stored;
    return (navigator.language || '').startsWith('zh') ? 'zh' : 'en';
  }
  function setLang(lang) {
    html.setAttribute('data-lang', lang);
    localStorage.setItem('obeaver-lang', lang);
    if (langToggle) {
      langToggle.textContent = lang === 'zh' ? 'EN' : '中文';
    }

    document.querySelectorAll('[data-placeholder-en][data-placeholder-zh]').forEach(function (el) {
      el.setAttribute('placeholder', lang === 'zh' ? el.getAttribute('data-placeholder-zh') : el.getAttribute('data-placeholder-en'));
    });

    document.querySelectorAll('[data-aria-label-en][data-aria-label-zh]').forEach(function (el) {
      el.setAttribute('aria-label', lang === 'zh' ? el.getAttribute('data-aria-label-zh') : el.getAttribute('data-aria-label-en'));
    });

    document.querySelectorAll('[data-alt-en][data-alt-zh]').forEach(function (el) {
      el.setAttribute('alt', lang === 'zh' ? el.getAttribute('data-alt-zh') : el.getAttribute('data-alt-en'));
    });

    document.querySelectorAll('.copy-btn[data-copy-en][data-copy-zh]').forEach(function (btn) {
      btn.textContent = lang === 'zh' ? btn.getAttribute('data-copy-zh') : btn.getAttribute('data-copy-en');
    });

    var titleEl = document.querySelector('title[data-title-en][data-title-zh]');
    if (titleEl) {
      titleEl.textContent = lang === 'zh' ? titleEl.getAttribute('data-title-zh') : titleEl.getAttribute('data-title-en');
    }

    var metaDescription = document.querySelector('meta[name="description"][data-content-en][data-content-zh]');
    if (metaDescription) {
      metaDescription.setAttribute('content', lang === 'zh' ? metaDescription.getAttribute('data-content-zh') : metaDescription.getAttribute('data-content-en'));
    }
  }
  setLang(getPreferredLang());
  if (langToggle) {
    langToggle.addEventListener('click', function () {
      var current = html.getAttribute('data-lang');
      setLang(current === 'zh' ? 'en' : 'zh');
    });
  }

  // ── Mobile Sidebar Toggle ─────────────────────────────────
  const menuToggle = document.querySelector('.menu-toggle');
  const sidebar = document.getElementById('sidebar');
  const sidebarOverlay = document.getElementById('sidebar-overlay');

  function closeSidebar() {
    if (sidebar) sidebar.classList.remove('open');
    if (sidebarOverlay) sidebarOverlay.classList.remove('active');
  }

  if (menuToggle) {
    menuToggle.addEventListener('click', function () {
      sidebar.classList.toggle('open');
      sidebarOverlay.classList.toggle('active');
    });
  }

  if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', closeSidebar);
  }

  // ── Copy Buttons ──────────────────────────────────────────
  document.querySelectorAll('.copy-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var code = btn.getAttribute('data-code');
      if (!code) {
        // Fallback: get text from sibling pre
        var pre = btn.closest('.code-block').querySelector('pre code');
        if (pre) code = pre.textContent;
      }
      if (!code) return;

      navigator.clipboard.writeText(code).then(function () {
        var lang = html.getAttribute('data-lang') || 'en';
        btn.textContent = lang === 'zh' ? (btn.getAttribute('data-copied-zh') || '已复制！') : (btn.getAttribute('data-copied-en') || 'Copied!');
        btn.classList.add('copied');
        setTimeout(function () {
          btn.textContent = lang === 'zh' ? (btn.getAttribute('data-copy-zh') || '复制') : (btn.getAttribute('data-copy-en') || 'Copy');
          btn.classList.remove('copied');
        }, 2000);
      });
    });
  });

  // ── Scroll-to-Top ─────────────────────────────────────────
  var scrollTop = document.getElementById('scroll-top');
  if (scrollTop) {
    window.addEventListener('scroll', function () {
      if (window.scrollY > 400) {
        scrollTop.classList.add('visible');
      } else {
        scrollTop.classList.remove('visible');
      }
    });
    scrollTop.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // ── Client-side Search ────────────────────────────────────
  var searchInput = document.getElementById('search-input');
  var searchResults = document.getElementById('search-results');

  // Search index — maps keywords to pages
  var searchIndex = [
    { title: 'Home', url: 'index.html', section: 'Getting Started', keywords: 'home welcome obeaver local llm inference onnx openai api' },
    { title: 'Quickstart', url: 'quickstart.html', section: 'Getting Started', keywords: 'quickstart install setup requirements python pip foundry ort check environment' },
    { title: 'Models', url: 'models.html', section: 'Getting Started', keywords: 'models phi qwen gemma onnx foundry catalog download embedding vl vision huggingface' },
    { title: 'Model Conversion', url: 'convert.html', section: 'Getting Started', keywords: 'convert conversion onnx int4 fp16 olive ort vl vision-language model builder build-from-source' },
    { title: 'Features', url: 'features.html', section: 'Usage', keywords: 'features dual engine embeddings tool calling function agentic vision language vl dashboard monitor' },
    { title: 'API Reference', url: 'api.html', section: 'Usage', keywords: 'api reference chat completions embeddings health models streaming sse curl endpoint v1 post get' },
    { title: 'Integrations', url: 'integrations.html', section: 'Usage', keywords: 'integrations langchain llamaindex openai sdk python ci evaluation agent framework crewai' },
    { title: 'Docker', url: 'docker.html', section: 'Deployment', keywords: 'docker container cpu arm64 amd64 build run environment variables deploy' },
    { title: 'Benchmark', url: 'benchmark.html', section: 'Usage', keywords: 'benchmark performance ttft tok/s tokens timing stats metrics monitor dashboard memory cpu gpu npu latency speed' },
    { title: 'Architecture', url: 'architecture.html', section: 'Deployment', keywords: 'architecture engine selection foundry ort tool calling internals cli server code structure' },
    { title: 'Function Tools', url: 'tools.html', section: 'Usage', keywords: 'function tools tool calling agent agentic workflow openai function-calling parse_tool_call inject_tools' },
    // Sub-sections
    { title: 'Installation', url: 'quickstart.html#installation', section: 'Quickstart', keywords: 'install pip editable setup' },
    { title: 'Foundry Local Engine', url: 'quickstart.html#foundry-local-engine', section: 'Quickstart', keywords: 'foundry local engine macos windows brew winget' },
    { title: 'ORT Engine', url: 'quickstart.html#ort-engine', section: 'Quickstart', keywords: 'ort onnxruntime genai linux cpu' },
    { title: 'Chat Completions', url: 'api.html#chat-completions', section: 'API Reference', keywords: 'chat completions post v1 streaming messages' },
    { title: 'Embeddings API', url: 'api.html#embeddings-api', section: 'API Reference', keywords: 'embeddings post v1 embed text rag retrieval' },
    { title: 'Tool Calling', url: 'features.html#tool-calling', section: 'Features', keywords: 'tool calling function agent agentic workflow' },
    { title: 'Text Embeddings', url: 'features.html#embeddings', section: 'Features', keywords: 'embeddings rag retrieval qwen gemma' },
    { title: 'Vision-Language Models', url: 'features.html#vision', section: 'Features', keywords: 'vision language vl multimodal image qwen' },
    { title: 'Text Model Conversion', url: 'convert.html#text-models', section: 'Model Conversion', keywords: 'text conversion onnx int4 fp16 convert run ort' },
    { title: 'VL Model Conversion', url: 'convert.html#vl-models', section: 'Model Conversion', keywords: 'vl conversion qwen olive vision.onnx build-from-source' },
    { title: 'Conversion Flags', url: 'convert.html#flags-reference', section: 'Model Conversion', keywords: 'flags model type output precision ep cache extra options' },
    { title: 'When to Convert', url: 'convert.html#when-to-convert', section: 'Model Conversion', keywords: 'foundry local ort convert requirement hf download safetensors' },
    { title: 'VL Models', url: 'models.html#vl-models', section: 'Models', keywords: 'vl vision language qwen multimodal' },
    { title: 'LangChain', url: 'integrations.html#langchain', section: 'Integrations', keywords: 'langchain chatopenai llm' },
    { title: 'LlamaIndex', url: 'integrations.html#llamaindex', section: 'Integrations', keywords: 'llamaindex openailike llm' },
    { title: 'Docker Build', url: 'docker.html#build', section: 'Docker', keywords: 'docker build buildx platform arm64 amd64' },
    { title: 'Docker Run', url: 'docker.html#run', section: 'Docker', keywords: 'docker run serve chat volume mount' },
    { title: 'CLI Timing Stats', url: 'benchmark.html#cli-timing-stats', section: 'Benchmark', keywords: 'cli timing ttft tok/s timings flag run' },
    { title: 'System Resource Gauges', url: 'benchmark.html#system-resource-gauges', section: 'Benchmark', keywords: 'cpu gpu npu memory gauge resource monitor dashboard' },
    { title: 'Chat Performance Metrics', url: 'benchmark.html#chat-performance-metrics', section: 'Benchmark', keywords: 'chat performance ttft tok/s tokens response speed' },
    { title: 'Performance Tips', url: 'benchmark.html#tips', section: 'Benchmark', keywords: 'tips optimization speed model quantization int4 accelerator' },
  ];

  function performSearch(query) {
    if (!query || query.length < 2) {
      searchResults.classList.remove('active');
      return;
    }

    var terms = query.toLowerCase().split(/\s+/);
    var results = searchIndex.filter(function (item) {
      var haystack = (item.title + ' ' + item.keywords + ' ' + item.section).toLowerCase();
      return terms.every(function (term) {
        return haystack.indexOf(term) !== -1;
      });
    });

    if (results.length === 0) {
      var lang = html.getAttribute('data-lang') || 'en';
      searchResults.innerHTML = '<div class="search-result-item"><span class="result-title">' +
        (lang === 'zh' ? '未找到结果' : 'No results found') +
        '</span></div>';
      searchResults.classList.add('active');
      return;
    }

    searchResults.innerHTML = results.slice(0, 8).map(function (item) {
      return '<a href="' + item.url + '" class="search-result-item">' +
        '<span class="result-title">' + escapeHtml(item.title) + '</span>' +
        '<span class="result-section">' + escapeHtml(item.section) + '</span>' +
        '</a>';
    }).join('');
    searchResults.classList.add('active');
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  if (searchInput) {
    searchInput.addEventListener('input', function () {
      performSearch(this.value.trim());
    });

    searchInput.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        searchResults.classList.remove('active');
        searchInput.blur();
      }
    });

    // Close search on click outside
    document.addEventListener('click', function (e) {
      if (!e.target.closest('.search-box')) {
        searchResults.classList.remove('active');
      }
    });

    // Keyboard shortcut: / to focus search
    document.addEventListener('keydown', function (e) {
      if (e.key === '/' && !e.ctrlKey && !e.metaKey && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        searchInput.focus();
      }
    });
  }

  // ── Add Anchor Links to Headers ───────────────────────────
  document.querySelectorAll('.content h2[id], .content h3[id]').forEach(function (heading) {
    var anchor = document.createElement('a');
    anchor.className = 'anchor';
    anchor.href = '#' + heading.id;
    anchor.textContent = '#';
    anchor.setAttribute('aria-label', 'Link to ' + heading.textContent);
    heading.appendChild(anchor);
  });

})();
