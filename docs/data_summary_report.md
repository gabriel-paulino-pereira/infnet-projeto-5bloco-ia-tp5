# Data Summary Report — Plataforma de Inteligência ESG & ODS 11

## 1. Visão Geral do Ciclo de Dados
Este documento consolida a arquitetura de dados do projeto sob as metodologias **TDSP** e **CRISP-DM**, descrevendo como os dados não estruturados extraídos da web foram tratados, enriquecidos por modelos de linguagem natural (LLMs) e disponibilizados para consumo analítico.

| Etapa do Ciclo | Módulo / Arquivo | Tecnologia | Artefato Gerado |
| :--- | :--- | :--- | :--- |
| **1. Coleta Estática** | `src/data/scraping_noticias.py` | `Requests` + `BeautifulSoup4` | `data/raw/noticias_sustentabilidade.csv` |
| **2. Coleta Dinâmica** | `src/data/scraping_dinamico.py` | `Selenium WebDriver` (Headless) | `data/raw/noticias_dinamicas.csv` |
| **3. Sanitização & NLP** | `src/data/preparacao_llm.py` | `Python` / `Regex` / `JSON` | `data/processed/dataset_llm_prep.json` |
| **4. Disponibilização REST** | `src/api/main.py` | `FastAPI` + `Pydantic` | Endpoints `/noticias` e `/processar_texto` |
| **5. Enriquecimento & IA** | `src/api/main.py` + `src/pages/` | `DistilBERT` + `Prompt Engineering` | Classificação ESG, Resumos e Recomendações |

---

## 2. Fontes e Integração dos Dados Coletados
A base de conhecimento é alimentada por fontes jornalísticas públicas (com foco na Agência Brasil e portais de notícias socioambientais) alinhadas aos **Objetivos de Desenvolvimento Sustentável (ODS 11 — Cidades e Comunidades Sustentáveis e ODS 12 — Consumo e Produção Responsáveis)**:

1. **Extração Estática (`noticias_sustentabilidade.csv`):** Captura manchetes, resumos/lides e links de páginas HTML estáticas, estruturando os registros iniciais com rastreabilidade de origem (`Fonte = BeautifulSoup_Static`).
2. **Extração Dinâmica (`noticias_dinamicas.csv`):** Contorna carregamentos assíncronos em JavaScript utilizando navegador automatizado (`Selenium`), ampliando a cobertura temática para ocorrências climáticas, mobilidade urbana e políticas ambientais (`Fonte = Selenium_Dynamic`).
3. **Consolidação em Memória e Deduplicação:** Na camada de serviço (`src/api/main.py`) e nos pipelines analíticos, ambas as fontes são unificadas via `pandas.concat()`, aplicando remoção de duplicatas pela chave `Titulo`, preenchimento de valores ausentes (`fillna`) e atribuição de identificadores únicos (`id`).

---

## 3. Esquema do Dataset e Dicionário de Variáveis

### 3.1. Dados Tabulares Consolidados (Camada Raw & API)
- **`id` (`int`):** Identificador sequencial único do registro consolidado.
- **`Titulo` (`string`):** Manchete principal da notícia coletada.
- **`Resumo` (`string`):** Texto de apoio ou lide jornalístico (tratado para ausência de nulos).
- **`Link` (`string`):** URL original da publicação para auditoria e verificação da fonte.
- **`Fonte` (`string`):** Metadado indicador do coletor utilizado (`BeautifulSoup_Static`, `Selenium_Dynamic` ou `API_Post`).

### 3.2. Corpus Textual Sanitizado (`dataset_llm_prep.json`)
Para evitar ruídos semânticos e estouro da janela de contexto dos modelos Transformer, o script `preparacao_llm.py` executa limpeza de tags HTML residuais, normalização de espaços e concatenação estruturada (`Título + Resumo`), gerando objetos JSON com:
- **`id` (`int`):** Chave de rastreamento.
- **`texto` (`string`):** Texto limpo e pronto para inferência.
- **`tamanho_caracteres` (`int`):** Métrica de controle para truncamento seguro (limite de 512 tokens no encoder).

---

## 4. Transformação em Insights via Inteligência Artificial
Os dados brutos passam por duas camadas complementares de inteligência:
1. **Classificação de Polaridade ESG (Modelo Local DistilBERT):** Cada notícia é classificada em **Positivo** (avanços sustentáveis, investimentos verdes, políticas públicas), **Neutro** (informes técnicos/administrativos) ou **Negativo** (desastres climáticos, desmatamento, crises urbanas), acompanhada do score de confiança probabilístico.
2. **Síntese e Apoio à Decisão via Prompt Engineering:** Os textos classificados e suas estatísticas agregadas são injetados como contexto em templates de prompts estruturados (delimitadores de contexto, definição de papel analítico e restrições de grounding), transformando notícias dispersas em diagnósticos executivos e recomendações de governança.

---

## 5. Qualidade, Limitações e Governança dos Dados
- **Qualidade:** Integridade garantida por schemas `Pydantic` na API e validações de tamanho mínimo e presença de caracteres alfabéticos antes de qualquer inferência.
- **Limitações Mitigadas:** Coletas web estão sujeitas a alterações de layout HTML; por isso, o projeto adota arquitetura híbrida (estática + dinâmica) e fallback de leitura em cache local (`CSV`/`JSON`), assegurando disponibilidade contínua do dashboard mesmo em modo offline.