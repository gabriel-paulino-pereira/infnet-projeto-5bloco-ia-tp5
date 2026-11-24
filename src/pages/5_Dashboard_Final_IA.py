"""
Página 5: Dashboard Final Integrado — Ciclo Completo de Ciência de Dados & IA (TP5)
Descrição: Consolida os dados coletados via web scraping (BeautifulSoup e Selenium),
exibe gráficos da classificação de sentimento ESG (DistilBERT) e integra o Agente
Inteligente de Tomada de Decisão (Qwen2.5-1.5B + Prompt Engineering).
"""

import os
import sys
import json
import requests
import pandas as pd
import streamlit as st

RAIZ_PROJETO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if RAIZ_PROJETO not in sys.path:
    sys.path.insert(0, RAIZ_PROJETO)

from src.ai.agente_esg import consolidar_e_classificar_dados

st.set_page_config(page_title="Dashboard Final IA - ESG", page_icon="📊", layout="wide")

st.title("📊 Dashboard Executivo ESG & ODS 11 — Ciclo Completo de IA")
st.caption("Da coleta automatizada (Web Scraping) à tomada de decisão assistida por LLMs locais.")

URL_API = "http://127.0.0.1:8000"

# ==========================================================
# 1. VISÃO DO PIPELINE DE DADOS COLETADOS (ETAPAS 1 A 3)
# ==========================================================
st.subheader("1. Panorama da Coleta e Preparação de Dados")

caminho_estatico = os.path.join("data", "raw", "noticias_sustentabilidade.csv")
caminho_dinamico = os.path.join("data", "raw", "noticias_dinamicas.csv")
caminho_json = os.path.join("data", "processed", "dataset_llm_prep.json")

qtd_estatico = len(pd.read_csv(caminho_estatico)) if os.path.exists(caminho_estatico) else 0
qtd_dinamico = len(pd.read_csv(caminho_dinamico)) if os.path.exists(caminho_dinamico) else 0
qtd_sanitizado = 0
if os.path.exists(caminho_json):
    with open(caminho_json, "r", encoding="utf-8") as f:
        qtd_sanitizado = len(json.load(f))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Coleta Estática (BeautifulSoup)", f"{qtd_estatico} registros")
m2.metric("Coleta Dinâmica (Selenium)", f"{qtd_dinamico} registros")
m3.metric("Corpus Sanitizado (JSON)", f"{qtd_sanitizado} docs")
m4.metric("Modelos Locais Ativos", "DistilBERT + Qwen 2.5")

st.divider()

# ==========================================================
# 2. ANÁLISE QUANTITATIVA DE SENTIMENTOS ESG (DISTILBERT)
# ==========================================================
st.subheader("2. Monitoramento de Polaridade ESG nas Notícias Coletadas")

@st.cache_data(show_spinner="Classificando base consolidada com DistilBERT local...")
def obter_base_analisada(limite: int = 25):
    return consolidar_e_classificar_dados(limite_noticias=limite)

df_esg = obter_base_analisada(limite=25)

if not df_esg.empty:
    col_graf1, col_graf2 = st.columns(2)

    with col_graf1:
        st.markdown("**Distribuição de Sentimentos ESG (Classificação IA)**")
        contagem_sent = df_esg["Sentimento"].value_counts().reset_index()
        contagem_sent.columns = ["Sentimento", "Quantidade"]
        st.bar_chart(contagem_sent.set_index("Sentimento"), width="stretch")

    with col_graf2:
        st.markdown("**Volume de Coleta por Fonte de Extração**")
        contagem_origem = df_esg["Origem"].value_counts().reset_index()
        contagem_origem.columns = ["Origem", "Quantidade"]
        st.bar_chart(contagem_origem.set_index("Origem"), width="stretch")

    with st.expander("Inspecionar Base Consolidada e Classificada pela IA"):
        filtro_sent = st.multiselect(
            "Filtrar por Sentimento:",
            options=df_esg["Sentimento"].unique().tolist(),
            default=df_esg["Sentimento"].unique().tolist()
        )
        df_filtrado = df_esg[df_esg["Sentimento"].isin(filtro_sent)]
        st.dataframe(
            df_filtrado[["Titulo", "Origem", "Sentimento", "Confianca"]],
            width="stretch",
            hide_index=True
        )
else:
    st.warning("Nenhum dado CSV encontrado em `data/raw/`. Execute os coletores antes de prosseguir.")

st.divider()

# ==========================================================
# 3. AGENTE INTELIGENTE DE TOMADA DE DECISÃO (PROMPT ENGINEERING)
# ==========================================================
st.subheader("3. Agente Inteligente para Tomada de Decisão (ODS 11 e ODS 12)")
st.write(
    "O agente cruza os indicadores quantitativos acima com as manchetes críticas para gerar "
    "diagnóstico de riscos e recomendações práticas de governança."
)

col_cfg1, col_cfg2 = st.columns([3, 1])

with col_cfg1:
    foco_estrategico = st.text_input(
        "Direcionamento / Pergunta Estratégica para o Agente:",
        value="Avaliação geral de riscos ambientais, eventos climáticos e governança urbana (ODS 11 e 12)"
    )

with col_cfg2:
    qtd_analise = st.slider("Amostra de notícias:", min_value=5, max_value=25, value=15, step=5)

if st.button("Gerar Diagnóstico e Recomendações do Agente ESG", type="primary"):
    with st.spinner("Agente analisando indicadores e gerando parecer via Qwen2.5-1.5B local (aguarde ~30-60s na CPU)..."):
        try:
            resposta = requests.post(
                f"{URL_API}/agente_decisao",
                json={
                    "foco_analise": foco_estrategico,
                    "limite_noticias": qtd_analise
                },
                timeout=None
            )

            if resposta.status_code == 200:
                dados_agente = resposta.json()
                st.session_state["ultimo_parecer_agente"] = dados_agente
                st.success(f"Parecer estratégico gerado com sucesso a partir de {dados_agente['total_noticias_analisadas']} notícias!")
            else:
                erro_api = resposta.json().get("detail", resposta.text)
                st.error(f"Erro HTTP {resposta.status_code}: {erro_api}")

        except requests.exceptions.ConnectionError:
            st.error("Servidor FastAPI offline. Inicie com `uvicorn src.api.main:app --reload` na porta 8000.")
        except requests.exceptions.Timeout:
            st.error("Tempo limite excedido na geração local.")
        except Exception as e:
            st.error(f"Erro inesperado: {e}")

# Exibe o resultado persistido na sessão para permitir download e inspeção sem reprocessar
if "ultimo_parecer_agente" in st.session_state:
    res = st.session_state["ultimo_parecer_agente"]

    st.markdown("### 📋 Parecer Técnico do Agente ESG")
    st.info(res["parecer_agente"])

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        st.download_button(
            label="📥 Baixar Parecer (.txt)",
            data=f"FOCO: {res['foco_analise']}\nMODELO: {res['modelo_generativo']}\n\n{res['parecer_agente']}",
            file_name="parecer_agente_esg_ods11.txt",
            mime="text/plain"
        )

    with st.expander("🔍 Auditar Prompt Estruturado Enviado ao LLM (Prompt Engineering)"):
        st.code(res["prompt_utilizado"], language="markdown")