# Plataforma de Inteligência ESG & ODS 11 — Ciclo Completo de Ciência de Dados e IA

Aplicação ponta a ponta de Ciência de Dados Aplicada e Engenharia de Inteligência Artificial desenvolvida para monitoramento socioambiental e apoio à tomada de decisão em **Governança Corporativa (ESG)**, alinhada à **Agenda 2030 da ONU** (**ODS 11 — Cidades e Comunidades Sustentáveis** e **ODS 12 — Consumo e Produção Responsáveis**).

---

## Arquitetura e Ciclo de Ciência de Dados (TDSP / CRISP-DM)

1. **Aquisição de Dados (`src/data/`):**
   - `scraping_noticias.py`: Coleta automatizada de manchetes e resumos via `Requests` + `BeautifulSoup4`.
   - `scraping_dinamico.py`: Extração de conteúdos dinâmicos renderizados em JavaScript via `Selenium WebDriver`.
2. **Preparação e Sanitização (`src/data/preparacao_llm.py`):**
   - Limpeza de ruídos HTML, normalização textual, deduplicação por título e serialização em `data/processed/dataset_llm_prep.json`.
3. **Backend RESTful & Validação (`src/api/main.py`):**
   - API construída com `FastAPI` e `Pydantic` (`Response Models` estritos e tratamento de erros HTTP `400`, `404`, `422`, `500`, `503` e `504`).
4. **Modelos de IA Locais & Prompt Engineering (`src/ai/agente_esg.py`):**
   - **Classificação Analítica:** `lxyuan/distilbert-base-multilingual-cased-sentiments-student` para categorização de polaridade ESG (`Positivo`, `Neutro`, `Negativo`).
   - **Agente Inteligente de Decisão:** `Qwen/Qwen2.5-1.5B-Instruct` operado com *Role Prompting*, *Context Grounding* e *Structured Output* para diagnóstico de riscos e recomendações de governança.
5. **Dashboard Executivo Interativo (`src/app.py` e `src/pages/`):**
   - Interface multipáginas em `Streamlit`, culminando em `5_Dashboard_Final_IA.py`, que integra métricas de coleta, gráficos de distribuição de sentimentos, auditoria de prompts e geração de pareceres estratégicos em tempo real.

---

## Estrutura do Projeto

```text
├── data/
│   ├── raw/                  # Bases brutas extraídas (CSV estático e dinâmico)
│   └── processed/            # Corpus sanitizado para LLMs (JSON)
├── docs/
│   ├── project_charter.md    # Artefato TDSP: Escopo, problema de negócio e alinhamento IA
│   ├── data_summary_report.md# Artefato TDSP: Dicionário de dados e integração
│   └── modelo_llm_justificativa.md # Justificativa técnica da arquitetura LLM local
├── src/
│   ├── ai/
│   │   └── agente_esg.py     # Pipeline híbrido DistilBERT + Qwen2.5 + Prompt Engineering
│   ├── api/
│   │   └── main.py           # Servidor FastAPI com Response Models e endpoints de IA
│   ├── data/                 # Scripts de Web Scraping e preparação textual
│   ├── pages/                # Páginas do dashboard multipáginas Streamlit
│   └── app.py                # Ponto de entrada da aplicação Streamlit
└── requirements.txt          # Dependências do ambiente Python
```

---

## Como Executar Localmente

1. **Ativar o ambiente virtual e instalar dependências:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Iniciar o servidor FastAPI (Terminal 1):**
   ```powershell
   uvicorn src.api.main:app --reload
   ```
3. **Iniciar o Dashboard Streamlit (Terminal 2):**
   ```powershell
   streamlit run src/app.py
   ```