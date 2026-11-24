# Project Charter — Plataforma de Inteligência ESG & ODS 11

## 1. Problema de Negócio e Contexto Estratégico
Gestores públicos, analistas de sustentabilidade e equipes de governança corporativa (ESG) enfrentam grande volume e dispersão de informações diárias sobre impactos ambientais, eventos climáticos extremos, mobilidade e políticas urbanas. O monitoramento manual dessas notícias é lento, reativo e dificulta a priorização de riscos alinhados à **Agenda 2030 da ONU**, especialmente aos **ODS 11 (Cidades e Comunidades Sustentáveis)** e **ODS 12 (Consumo e Produção Responsáveis)**.

**Solução Proposta:** Desenvolver uma plataforma ponta a ponta de Ciência de Dados Aplicada capaz de coletar notícias reais automaticamente, estruturá-las em uma API validada e aplicar modelos de linguagem natural (LLMs) com *Prompt Engineering* para classificar riscos, sintetizar cenários e apoiar a tomada de decisão baseada em evidências.

---

## 2. Objetivos do Projeto
1. **Automatizar a Captura de Dados:** Implementar extratores estáticos (`BeautifulSoup`) e dinâmicos (`Selenium`) para consolidar um corpus atualizado sobre sustentabilidade e governança urbana.
2. **Garantir Confiabilidade e Arquitetura Desacoplada:** Disponibilizar os dados e inferências por meio de uma API RESTful (`FastAPI`) com contratos rigorosos (`Response Models`) e tratamento resiliente de falhas HTTP.
3. **Aplicar IA Local e Prompt Engineering:** Integrar processamento de linguagem natural local (`DistilBERT`) e técnicas de engenharia de prompts para transformar textos não estruturados em métricas de sentimento, sínteses executivas e recomendações acionáveis.
4. **Entregar Visualização Analítica Intuitiva:** Consolidar todo o fluxo em um dashboard interativo (`Streamlit`) que permita explorar desde os dados brutos até os insights gerados pela IA.

---

## 3. Alinhamento entre Dados, IA, Prompt Engineering e Negócio
A arquitetura do projeto foi desenhada para que cada componente técnico gere valor direto ao problema de negócio:

- **Dados Reais como Base de Evidência:** O uso de notícias reais coletadas via web scraping evita análises genéricas, ancorando o sistema em eventos concretos (como desastres climáticos, obras de infraestrutura verde e regulações ambientais).
- **Inferência Local de Baixo Custo (`DistilBERT`):** A classificação automática de sentimento filtra rapidamente o fluxo informativo, permitindo identificar alertas críticos (`Negativo`) e boas práticas (`Positivo`) em milissegundos na CPU local, sem custos de nuvem ou vazamento de dados sensíveis.
- **Engenharia de Prompts Orientada a Decisão:** A aplicação de *Prompt Engineering* (utilizando *Role Prompting*, delimitação explícita de contexto recuperado da base coletada e instruções de saída estruturada) reduz alucinações e direciona o LLM a atuar como um especialista em ESG, cruzando indicadores da base para propor planos de ação e mitigação de riscos.

---

## 4. Ciclo Completo de Ciência de Dados (TDSP / CRISP-DM)
1. **Entendimento do Negócio:** Definição do escopo ESG/ODS 11 e dos indicadores de monitoramento socioambiental.
2. **Aquisição e Entendimento dos Dados:** Construção dos pipelines de web scraping (`scraping_noticias.py` e `scraping_dinamico.py`) e análise exploratória no `data_summary_report.md`.
3. **Preparação dos Dados:** Limpeza textual, deduplicação e estruturação em JSON (`preparacao_llm.py`) e exposição via `FastAPI` (`src/api/main.py`).
4. **Modelagem e Integração de IA:** Embarque local do modelo `lxyuan/distilbert-base-multilingual-cased-sentiments-student` e implementação da camada de *Prompt Engineering* para análise inteligente.
5. **Implantação e Visualização:** Disponibilização do dashboard multipáginas em `Streamlit` (`src/app.py` e `src/pages/`), permitindo interação em tempo real e exportação de resultados.

---

## 5. Critérios de Sucesso e Entregáveis Finais
- [x] Pipeline funcional de coleta estática e dinâmica com armazenamento estruturado (`CSV`/`JSON`).
- [x] Backend `FastAPI` com `Response Models` (`Pydantic`) e exceções HTTP tratadas (`400`, `404`, `422`, `500`, `503`, `504`).
- [x] Modelo LLM local integrado e documentado (`docs/modelo_llm_justificativa.md`).
- [x] Funcionalidade avançada com *Prompt Engineering* integrada ao dashboard final e artefatos TDSP concluídos.