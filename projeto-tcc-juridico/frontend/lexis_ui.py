"""HTML, CSS e fragmentos reutilizáveis do dashboard LexisAnalysis."""
from __future__ import annotations

from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent
KB_PAGE_SIZE = 4
PROFILE_IMG = (
    "https://lh3.googleusercontent.com/aida-public/AB6AXuBU53BTcs0kNo21UqxoLMNQ57V79RFSorb_VrgUREih6T0i-RLtJopGyv9osmPga9TMeLfGScAFiWAVi_7sWqY36pulHcUJ3nP37zjS8VwWxaAUlxa9TU_KQTRHKDkOUG5c7gUe_eyayjJGtqUsDzHXmfYGkg5ka6awenxXa8LJWCQAoY0JW9ylqLw_AnsYFYVUnqUwZZVpmhWOxQSg_dQ7e0piErMhI62W8P8TOYAnQsL6zw3KKmPMCoIaZlHy9rIxgUuGZvYmnHtc"
)


def load_theme_css() -> str:
    path = FRONTEND_DIR / "assets" / "lexis_bootstrap.css"
    if not path.is_file():
        path = FRONTEND_DIR / "assets" / "lexis_theme.css"
    return path.read_text(encoding="utf-8")


def q(val) -> str:
    if val is None or val == "—":
        return '"—"'
    if isinstance(val, float):
        return f'"{val:.0f}"' if val == int(val) else f'"{val:.1f}"'
    return f'"{val}"'


def icon(name: str, size: str = "") -> str:
    style = f' style="font-size:{size}"' if size else ""
    return f'<span class="material-symbols-outlined"{style}>{name}</span>'


def step_header(num: int, title: str) -> str:
    return (
        f'<div class="step-header">'
        f'<div class="step-title"><span class="step-num">{num}</span>{title}</div>'
        f"</div>"
    )


SIDEBAR_HTML = """
<div class="sidebar-overlay" id="lex-sidebar-overlay" onclick="lexCloseSidebar()" aria-hidden="true"></div>
<aside class="sidebar" id="lex-sidebar">
  <div class="sidebar-brand">
    <div class="brand-icon-wrap"><span class="material-symbols-outlined">gavel</span></div>
    <div class="sidebar-brand-text">
      <h1 class="brand-title">LexisAnalysis</h1>
      <p class="brand-sub">Institutional Legal Suite</p>
    </div>
  </div>
  <nav class="sidebar-nav">
    <a class="nav-item nav-item--active" href="#" onclick="lexCloseSidebar();return false">
      <span class="material-symbols-outlined">database</span><span class="nav-label">Knowledge Base</span>
    </a>
    <a class="nav-item" href="#" onclick="lexCloseSidebar();return false">
      <span class="material-symbols-outlined">upload_file</span><span class="nav-label">Petition Upload</span>
    </a>
    <a class="nav-item" href="#" onclick="lexCloseSidebar();return false">
      <span class="material-symbols-outlined">analytics</span><span class="nav-label">Analysis Results</span>
    </a>
  </nav>
  <div class="sidebar-cta">
    <button type="button" class="btn-cta" onclick="lexCloseSidebar();window.scrollTo({top:0,behavior:'smooth'})">
      <span class="material-symbols-outlined">add</span><span class="sidebar-cta-label">New Analysis</span>
    </button>
  </div>
  <footer class="sidebar-footer">
    <a class="nav-item nav-item--footer" href="#"><span class="material-symbols-outlined">help</span><span class="footer-label">Support</span></a>
    <a class="nav-item nav-item--footer" href="#"><span class="material-symbols-outlined">history</span><span class="footer-label">Archive</span></a>
  </footer>
</aside>
"""

NAVBAR_HTML = f"""
<header class="navbar">
  <div class="navbar-left">
    <button type="button" class="sidebar-toggle" onclick="lexToggleSidebar()" aria-label="Abrir menu">
      <span class="material-symbols-outlined">menu</span>
    </button>
    <h2 class="navbar-title">Dashboard</h2>
  </div>
  <div class="navbar-right">
    <div class="navbar-search-wrap">
      <input type="search" placeholder="Pesquisar processos..." aria-label="Pesquisar"/>
      <span class="material-symbols-outlined">search</span>
    </div>
    <div class="nav-actions">
      <button type="button" class="icon-btn" title="Notificações">{icon("notifications", "22px")}</button>
      <button type="button" class="icon-btn" title="Configurações">{icon("settings", "22px")}</button>
      <div class="avatar"><img alt="Perfil" src="{PROFILE_IMG}"/></div>
    </div>
  </div>
</header>
"""

RESPONSIVE_JS = """
<script>
function lexToggleSidebar() { document.body.classList.toggle('lex-sidebar-open'); }
function lexCloseSidebar() { document.body.classList.remove('lex-sidebar-open'); }
window.lexToggleSidebar = lexToggleSidebar;
window.lexCloseSidebar = lexCloseSidebar;
window.addEventListener('resize', () => { if (window.innerWidth > 768) lexCloseSidebar(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') lexCloseSidebar(); });
</script>
"""

TAB_SELECT_JS = """
<script>
function selectTab(tabId) {
  document.querySelectorAll('.results-tab').forEach(t => t.classList.remove('active'));
  const tab = document.querySelector(`.results-tab[data-tab="${tabId}"]`);
  if (tab) tab.classList.add('active');
  const map = {comparacao:0, melhorias:1, analise:2, indicadores:3, relatorio:4};
  const btn = document.querySelectorAll('.tab-nav button')[map[tabId]];
  if (btn) btn.click();
}
</script>
"""

PAGE_HEADER_HTML = """
<div class="page-header">
  <h3>Painel de Análise Estratégica</h3>
  <p>Gerencie sua base de conhecimento jurídico e execute comparações analíticas entre petições modelos e novos documentos com precisão algorítmica.</p>
</div>
"""

UPLOAD_ZONE_HTML = """
<div class="upload-zone" onclick="document.querySelector('input[type=file]').click()">
  <div class="upload-icon"><span class="material-symbols-outlined">cloud_upload</span></div>
  <h5 class="upload-title">Upload de Documento</h5>
  <p class="upload-hint">Arraste e solte o seu PDF ou TXT aqui para iniciar o processamento.</p>
  <div class="upload-pill">Selecionar Arquivo</div>
</div>
"""

OLLAMA_CHECKBOX_HTML = """
<label class="ollama-check">
  <input type="checkbox" checked/>
  <div>
    <span class="ollama-check__title">Análise textual detalhada com IA (Ollama)</span>
    <span class="ollama-check__sub">Processamento local seguro para máxima privacidade de dados.</span>
  </div>
</label>
"""

RESULTS_SECTION_HEAD = """
<section class="results-section step-card">
  <div class="step-header step-header--compact">
    <div class="step-title"><span class="step-num">3</span>Resultados</div>
  </div>
  <div class="results-tabs results-tabs--static">
    <button class="results-tab active"><span class="material-symbols-outlined">compare</span> Comparação</button>
    <button class="results-tab"><span class="material-symbols-outlined">auto_fix_high</span> Melhorias</button>
    <button class="results-tab"><span class="material-symbols-outlined">psychology</span> Análise</button>
    <button class="results-tab"><span class="material-symbols-outlined">leaderboard</span> Indicadores</button>
    <button class="results-tab"><span class="material-symbols-outlined">description</span> Relatório</button>
  </div>
"""

RESULTS_TABS_INTERACTIVE = """
<div class="results-tabs" id="results-tabs">
  <button class="results-tab active" data-tab="comparacao" onclick="selectTab('comparacao')">
    <span class="material-symbols-outlined">compare</span><span>Comparação</span>
  </button>
  <button class="results-tab" data-tab="melhorias" onclick="selectTab('melhorias')">
    <span class="material-symbols-outlined">auto_fix_high</span><span>Melhorias</span>
  </button>
  <button class="results-tab" data-tab="analise" onclick="selectTab('analise')">
    <span class="material-symbols-outlined">psychology</span><span>Análise</span>
  </button>
  <button class="results-tab" data-tab="indicadores" onclick="selectTab('indicadores')">
    <span class="material-symbols-outlined">leaderboard</span><span>Indicadores</span>
  </button>
  <button class="results-tab" data-tab="relatorio" onclick="selectTab('relatorio')">
    <span class="material-symbols-outlined">description</span><span>Relatório</span>
  </button>
</div>
"""

FOOTER_HTML = """
<div class="app-footer">
  LexisAnalysis · TCC — Comparação semântica (embeddings) + regras estruturais · Ollama opcional para análise crítica
</div>
"""


def _classe_score(nivel: str, score: float) -> str:
    n = (nivel or "").lower()
    if score >= 80 or "excelente" in n:
        return "score-excelente"
    if score >= 65 or "bom" in n:
        return "score-bom"
    if score >= 45 or "regular" in n:
        return "score-regular"
    return "score-baixo"


def _icone_nivel(nivel: str) -> str:
    n = nivel.lower()
    if "excelente" in n:
        return "🏆"
    if "bom" in n:
        return "✅"
    if "regular" in n:
        return "📋"
    return "⚠️"


def html_score(score: float, nivel: str, classificacao: str, referencia: str) -> str:
    ref = referencia if referencia and referencia != "—" else "—"
    return f"""
<div class="score-hero {_classe_score(nivel, score)}">
  <div class="score-icon">{_icone_nivel(nivel)}</div>
  <div class="score-value">{q(int(round(score)))}</div>
  <div class="score-label">{nivel.upper()} · {classificacao}</div>
  <div class="score-meta">📎 Modelo mais próximo: {ref}</div>
</div>
"""


def _kb_table_html(pdfs: list[str], page_size: int = KB_PAGE_SIZE) -> str:
    total = len(pdfs)
    pages = max(1, (total + page_size - 1) // page_size)
    rows = "".join(
        f'<tr data-idx="{i}">'
        f'<td><div class="file-name">{icon("picture_as_pdf", "18px")}{Path(p).name}</div></td>'
        f'<td class="file-actions"><button type="button" class="action-btn" title="Visualizar">'
        f'{icon("visibility", "18px")}</button></td></tr>'
        for i, p in enumerate(pdfs)
    )
    shown = min(page_size, total) if total else 0
    return f"""
<div id="kb-table-container">
  <table class="file-table">
    <thead><tr><th>Nome do Arquivo</th><th class="text-right">Ações</th></tr></thead>
    <tbody id="kb-tbody">{rows or ""}</tbody>
  </table>
  <div class="kb-pagination">
    <span id="kb-page-info">1–{shown} de {total}</span>
    <div class="kb-pagination__nav">
      <button type="button" class="pag-btn" onclick="kbPrevPage()">{icon("chevron_left", "16px")}</button>
      <span id="kb-page-num">1 / {pages}</span>
      <button type="button" class="pag-btn" onclick="kbNextPage()">{icon("chevron_right", "16px")}</button>
    </div>
  </div>
</div>
<script>
(function() {{
  const PAGE_SIZE = {page_size};
  const totalPdfs = {total};
  const totalPages = {pages};
  let currentPage = 1;
  function updateTable() {{
    const tbody = document.getElementById('kb-tbody');
    if (!tbody) return;
    const start = (currentPage - 1) * PAGE_SIZE;
    const end = start + PAGE_SIZE;
    tbody.querySelectorAll('tr[data-idx]').forEach((row, idx) => {{
      row.style.display = (idx >= start && idx < end) ? 'table-row' : 'none';
    }});
    const info = document.getElementById('kb-page-info');
    const pageNum = document.getElementById('kb-page-num');
    if (info) info.textContent = `${{start + 1}}–${{Math.min(end, totalPdfs)}} de ${{totalPdfs}}`;
    if (pageNum) pageNum.textContent = `${{currentPage}} / ${{totalPages}}`;
  }}
  window.kbPrevPage = () => {{ if (currentPage > 1) {{ currentPage--; updateTable(); }} }};
  window.kbNextPage = () => {{ if (currentPage < totalPages) {{ currentPage++; updateTable(); }} }};
  setTimeout(updateTable, 50);
}})();
</script>
"""


def html_status_base() -> str:
    from backend.knowledge_train import status_treino

    s = status_treino()
    pdfs = s.get("pdfs_em_knowledge") or []
    total = s.get("total_pdfs", len(pdfs))
    pronto = (s.get("chroma") or {}).get("juridico_docs", 0) > 0 and len(pdfs) > 0
    status_text = f'Status: "{total}" Petições model, "Pronta"' if pronto else f'Status: "{total}" Petições model, "Treinar"'
    dot_cls = "status-dot status-dot--ready" if pronto else "status-dot status-dot--pending"

    tabela = _kb_table_html(pdfs) if pdfs else (
        '<p class="kb-empty">Nenhum PDF em <code>data/knowledge/</code> ainda.</p>'
    )

    return f"""
<div class="kb-step-header">
  <div>
    <div class="step-title-row">
      <span class="step-num">1</span>
      <h4 class="step-card-title">Base de Conhecimento</h4>
    </div>
    <div class="status-line">
      <span class="{dot_cls}"></span>
      <span class="status-text">{status_text}</span>
    </div>
  </div>
  <div class="kb-step-actions">
    <button type="button" class="btn-outline" data-refresh-btn onclick="document.querySelector('.refresh-btn button').click()">
      {icon("refresh", "16px")} Atualizar
    </button>
    <button type="button" class="btn-secondary" data-train-btn onclick="document.querySelector('.train-btn button').click()">
      {icon("model_training", "16px")} Treinar base
    </button>
  </div>
</div>
<div class="legal-scroll kb-panel">{tabela}</div>
"""


def estado_vazio() -> tuple:
    placeholder = """
<div class="empty-state">
  <div class="empty-state__icon"><span class="material-symbols-outlined">pending_actions</span></div>
  <h5 class="empty-state__title">Aguardando Execução</h5>
  <p class="empty-state__text">Conclua o Passo 1 e o Passo 2, depois clique em <strong>Analisar</strong> para gerar os insights estratégicos.</p>
  <div class="empty-steps-grid">
    <div class="step-mini-card"><div class="step-mini-card__title">Passo 1</div><div>Treinar a base com PDFs modelo</div></div>
    <div class="step-mini-card"><div class="step-mini-card__title">Passo 2</div><div class="step-mini-card__text">Enviar PDF ou colar texto</div></div>
    <div class="step-mini-card"><div class="step-mini-card__title">Passo 3</div><div>Visualizar resultados</div></div>
  </div>
</div>
"""
    return placeholder, "", "", "", "", [], [], [], [], "", [], ""


def markdown_melhorias(rows: list) -> str:
    if not rows or (len(rows) == 1 and rows[0][0] == "—"):
        return "_Sem melhorias listadas — treine a base e repita a análise._"
    linhas = [
        "| Secção | Prioridade | O que melhorar | Sugestão |",
        "|--------|------------|----------------|----------|",
    ]
    for row in rows:
        if len(row) < 4:
            continue
        sec, pri, prob, sug = row[0], row[1], row[2], row[3]
        linhas.append(f"| **{sec}** | {pri} | {prob} | {sug} |")
    return "\n".join(linhas)


def alert(msg: str, kind: str = "success") -> str:
    return f'<div class="lex-alert lex-alert--{kind}">{msg}</div>'
