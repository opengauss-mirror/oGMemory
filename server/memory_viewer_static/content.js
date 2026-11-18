window.renderTeamMemoryContent = (() => {
  const escape = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[character]);

  function inline(text) {
    const tokens = /`([^`\n]+)`|\*\*([^*\n]+)\*\*/g;
    let html = '';
    let offset = 0;
    for (const match of text.matchAll(tokens)) {
      html += escape(text.slice(offset, match.index));
      html += match[1] === undefined ? `<strong>${escape(match[2])}</strong>` : `<code>${escape(match[1])}</code>`;
      offset = match.index + match[0].length;
    }
    return html + escape(text.slice(offset));
  }

  function markdown(text) {
    const output = [];
    let paragraph = [];
    let items = [];
    let listType = 'ul';
    let listStart = 1;
    let code = null;
    let fence = '';
    const flushParagraph = () => {
      if (paragraph.length) output.push(`<p>${inline(paragraph.join('\n')).replace(/\n/g, '<br>')}</p>`);
      paragraph = [];
    };
    const flushList = () => {
      if (items.length) output.push(`<${listType}${listType === 'ol' ? ` start="${listStart}"` : ''}>${items.map((item) => `<li>${inline(item)}</li>`).join('')}</${listType}>`);
      items = [];
    };
    const flushCode = () => {
      output.push(`<pre><code>${escape(code.join('\n'))}</code></pre>`);
      code = null;
    };
    for (const line of text.replace(/\r\n?/g, '\n').split('\n')) {
      const marker = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
      if (code !== null) {
        if (marker && marker[1][0] === fence[0] && marker[1].length >= fence.length && !marker[2].trim()) flushCode();
        else code.push(line);
        continue;
      }
      if (marker) {
        flushParagraph(); flushList();
        code = []; fence = marker[1];
        continue;
      }
      if (!line.trim()) { flushParagraph(); flushList(); continue; }
      const heading = line.match(/^ {0,3}(#{1,6})\s+(.+)$/);
      if (heading) {
        flushParagraph(); flushList();
        const level = Math.min(6, heading[1].length + 2);
        output.push(`<h${level}>${inline(heading[2])}</h${level}>`);
        continue;
      }
      const item = line.match(/^ {0,3}([-+*]|\d+[.)])\s+(.+)$/);
      if (item) {
        flushParagraph();
        const kind = /^\d/.test(item[1]) ? 'ol' : 'ul';
        if (listType !== kind) flushList();
        if (!items.length) { listType = kind; listStart = kind === 'ol' ? Number.parseInt(item[1], 10) : 1; }
        items.push(item[2]);
        continue;
      }
      flushList();
      paragraph.push(line);
    }
    flushParagraph(); flushList();
    if (code !== null) flushCode();
    return output.join('');
  }

  function render(value) {
    if (Array.isArray(value)) return `<ul>${value.map((item) => `<li>${render(item)}</li>`).join('')}</ul>`;
    if (value !== null && typeof value === 'object') {
      return `<dl class="space-content-properties">${Object.entries(value).map(([key, item]) => `<div><dt>${escape(key)}</dt><dd>${render(item)}</dd></div>`).join('')}</dl>`;
    }
    return typeof value === 'string' ? markdown(value) : `<p>${escape(value ?? '—')}</p>`;
  }

  return (value) => `<div class="space-rich-content">${render(value)}</div>`;
})();
