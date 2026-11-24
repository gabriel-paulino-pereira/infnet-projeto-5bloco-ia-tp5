"""
Módulo do Agente Inteligente para Tomada de Decisão ESG (Tarefa 2 - Opção 3)
Descrição: Consolida os dados coletados via web scraping, executa análise quantitativa
de sentimentos com DistilBERT e aplica Prompt Engineering em um LLM local
(Qwen2.5-1.5B-Instruct) para gerar diagnósticos e recomendações de governança.
"""

import os
import pandas as pd
from transformers import pipeline
import torch

# Otimiza o PyTorch para utilizar os 8 núcleos físicos do Ryzen 7 na CPU
torch.set_num_threads(8)

MODELO_SENTIMENTO = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
MODELO_GENERATIVO = "Qwen/Qwen2.5-1.5B-Instruct"

_pipe_sentimento = None
_pipe_gerador = None


def obter_pipelines_locais():
    """Carrega e mantém em cache na memória os pipelines locais de classificação e geração."""
    global _pipe_sentimento, _pipe_gerador

    if _pipe_sentimento is None:
        _pipe_sentimento = pipeline(
            task="text-classification",
            model=MODELO_SENTIMENTO,
            device=-1
        )

    if _pipe_gerador is None:
        _pipe_gerador = pipeline(
            task="text-generation",
            model=MODELO_GENERATIVO,
            device=-1
        )

    return _pipe_sentimento, _pipe_gerador


def consolidar_e_classificar_dados(limite_noticias: int = 20) -> pd.DataFrame:
    """
    Carrega os datasets coletados (estático e dinâmico), remove duplicatas
    e classifica o sentimento ESG de cada notícia usando o modelo DistilBERT local.
    """
    caminho_estatico = os.path.join("data", "raw", "noticias_sustentabilidade.csv")
    caminho_dinamico = os.path.join("data", "raw", "noticias_dinamicas.csv")

    dfs = []
    if os.path.exists(caminho_estatico):
        df_est = pd.read_csv(caminho_estatico)
        df_est["Origem"] = "BeautifulSoup (Estático)"
        dfs.append(df_est)

    if os.path.exists(caminho_dinamico):
        df_din = pd.read_csv(caminho_dinamico)
        df_din["Origem"] = "Selenium (Dinâmico)"
        dfs.append(df_din)

    if not dfs:
        return pd.DataFrame(columns=["Titulo", "Resumo", "Link", "Origem", "Sentimento", "Confianca"])

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["Titulo"]).head(limite_noticias).copy()
    df["Resumo"] = df["Resumo"].fillna("Sem resumo adicional.")

    pipe_sent, _ = obter_pipelines_locais()
    mapa_pt = {"positive": "Positivo", "neutral": "Neutro", "negative": "Negativo"}

    sentimentos = []
    confiancas = []

    for _, linha in df.iterrows():
        texto = f"{linha['Titulo']}. {linha['Resumo']}"[:512]
        pred = pipe_sent(texto)[0]
        rotulo = mapa_pt.get(pred["label"].lower(), pred["label"].capitalize())
        sentimentos.append(rotulo)
        confiancas.append(round(float(pred["score"]), 4))

    df["Sentimento"] = sentimentos
    df["Confianca"] = confiancas
    return df


def construir_prompt_agente_esg(df_classificado: pd.DataFrame, foco_usuario: str) -> str:
    """
    Aplica técnicas de Prompt Engineering (Role Prompting, Context Grounding e
    Structured Output) para orientar o LLM na tomada de decisão baseada nos dados reais.
    """
    total = len(df_classificado)
    contagem = df_classificado["Sentimento"].value_counts().to_dict()
    qtd_pos = contagem.get("Positivo", 0)
    qtd_neu = contagem.get("Neutro", 0)
    qtd_neg = contagem.get("Negativo", 0)

    # Seleciono as notícias críticas (Negativas) e positivas para ancorar o contexto
    amostras_negativas = df_classificado[df_classificado["Sentimento"] == "Negativo"]["Titulo"].head(5).tolist()
    amostras_positivas = df_classificado[df_classificado["Sentimento"] == "Positivo"]["Titulo"].head(5).tolist()

    lista_neg = "\n".join([f"- [ALERTA] {t}" for t in amostras_negativas]) or "- Nenhuma ocorrência negativa crítica."
    lista_pos = "\n".join([f"- [OPORTUNIDADE] {t}" for t in amostras_positivas]) or "- Nenhuma ocorrência positiva registrada."

    prompt_estruturado = f"""Você é um Agente Especialista Sênior em Governança Corporativa (ESG) e Desenvolvimento Sustentável (Agenda 2030 - ODS 11 e ODS 12).
Sua tarefa é analisar exclusivamente os dados reais coletados abaixo e fornecer um parecer técnico objetivo para apoiar a tomada de decisão de gestores.

<DADOS_COLETADOS>
Total de notícias analisadas: {total}
Distribuição de Sentimento ESG:
- Positivo (Avanços e Boas Práticas): {qtd_pos} ({(qtd_pos/total*100) if total else 0:.1f}%)
- Neutro (Informativo/Institucional): {qtd_neu} ({(qtd_neu/total*100) if total else 0:.1f}%)
- Negativo (Riscos e Impactos Socioambientais): {qtd_neg} ({(qtd_neg/total*100) if total else 0:.1f}%)

Principais Manchetes de Alerta (Sentimento Negativo):
{lista_neg}

Principais Manchetes de Destaque (Sentimento Positivo):
{lista_pos}
</DADOS_COLETADOS>

Foco específico solicitado pelo gestor: "{foco_usuario}"

Responda em português, de forma clara, direta e ancorada nos dados acima, seguindo exatamente esta estrutura:
1. DIAGNÓSTICO DO CENÁRIO ESG: Síntese do panorama atual com base nos percentuais e manchetes.
2. RISCOS PRIORITÁRIOS IDENTIFICADOS: Principais pontos de atenção extraídos das manchetes negativas.
3. RECOMENDAÇÕES DE AÇÃO (ODS 11 E ODS 12): Três ações práticas de governança e mitigação recomendadas para o gestor."""

    return prompt_estruturado


def executar_agente_decisao(foco_usuario: str = "Avaliação geral de riscos ambientais e governança urbana", limite_noticias: int = 20) -> dict:
    """
    Executa o fluxo completo do Agente Inteligente:
    1. Consolida e classifica a base real com DistilBERT;
    2. Constrói o prompt estruturado com os indicadores;
    3. Gera o parecer estratégico localmente via Qwen2.5-1.5B-Instruct.
    """
    df_classificado = consolidar_e_classificar_dados(limite_noticias=limite_noticias)

    if df_classificado.empty:
        raise ValueError("Nenhuma notícia encontrada nos arquivos CSV para alimentar o agente.")

    prompt_final = construir_prompt_agente_esg(df_classificado, foco_usuario)
    _, pipe_gerador = obter_pipelines_locais()

    mensagens = [
        {"role": "system", "content": "Você é um consultor analítico de ESG e Ciência de Dados. Baseie-se estritamente nos dados fornecidos."},
        {"role": "user", "content": prompt_final}
    ]

    resposta_llm = pipe_gerador(
        mensagens,
        max_new_tokens=650,
        do_sample=True,
        temperature=0.3,
        top_p=0.9
    )

    texto_gerado = resposta_llm[0]["generated_text"][-1]["content"].strip()

    return {
        "foco_analise": foco_usuario,
        "total_noticias_analisadas": len(df_classificado),
        "distribuicao_sentimentos": df_classificado["Sentimento"].value_counts().to_dict(),
        "prompt_utilizado": prompt_final,
        "parecer_agente": texto_gerado,
        "dados_classificados": df_classificado.to_dict(orient="records")
    }