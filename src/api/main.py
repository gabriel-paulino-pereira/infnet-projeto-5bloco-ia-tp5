"""
API RESTful com FastAPI Integrada a LLM Local - Plataforma ESG / ODS 11
Descrição: Servidor backend com Response Models estritos (Pydantic), tratamento
robusto e padronizado de exceções HTTP e inferência local via HuggingFace Transformers.
"""

import os
import time
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from transformers import pipeline
from src.ai.agente_esg import executar_agente_decisao, MODELO_GENERATIVO

# ==========================================================
# 1. CONFIGURAÇÃO DA APLICAÇÃO E CARREGAMENTO DO MODELO LOCAL
# ==========================================================
app = FastAPI(
    title="API de Gestão ESG & ODS 11 com LLM Local",
    description="Backend RESTful com Response Models validados, Exception Handlers HTTP e inferência local via DistilBERT.",
    version="2.2.0"
)

NOME_MODELO_LLM = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
TEMPO_MAXIMO_INFERENCIA_SEG = 10.0
classificador_llm = None

try:
    print(f"Carregando modelo LLM local ({NOME_MODELO_LLM}) na CPU...")
    classificador_llm = pipeline(
        task="text-classification",
        model=NOME_MODELO_LLM,
        device=-1
    )
    print("Modelo LLM carregado com sucesso na memória local!")
except Exception as erro_carga:
    print(f"Erro crítico ao carregar o modelo local: {erro_carga}")
    classificador_llm = None


# ==========================================================
# 2. REQUEST & RESPONSE MODELS (CONTRATOS PYDANTIC)
# ==========================================================
class ErroHTTPResponse(BaseModel):
    codigo_http: int = Field(..., description="Código de status HTTP retornado")
    erro: str = Field(..., description="Categoria resumida do erro")
    detail: str = Field(..., description="Descrição clara e informativa do problema ocorrido")
    sugestao: str = Field(..., description="Orientação prática para o usuário ou cliente")


class StatusAPIResponse(BaseModel):
    status: str = Field(..., description="Estado atual do serviço")
    mensagem: str = Field(..., description="Mensagem informativa")
    llm_local_carregado: bool = Field(..., description="Indica se o modelo HuggingFace está em memória")
    modelo: str = Field(..., description="Identificador do modelo LLM configurado")


class NoticiaRequest(BaseModel):
    Titulo: str = Field(..., min_length=3, description="Título da notícia")
    Resumo: str = Field(..., min_length=3, description="Resumo ou lide da matéria")
    Link: str = Field(..., description="URL de origem da notícia")
    Fonte: Optional[str] = Field(default="API_Post", description="Origem da coleta")


class NoticiaResponse(BaseModel):
    id: int = Field(..., description="Identificador único do registro")
    Titulo: str = Field(..., description="Título da notícia")
    Resumo: str = Field(..., description="Resumo ou lide da matéria")
    Link: str = Field(..., description="URL de origem da notícia")
    Fonte: str = Field(..., description="Fonte de extração do dado")


class CadastroNoticiaResponse(BaseModel):
    mensagem: str = Field(..., description="Confirmação da operação de cadastro")
    registro: NoticiaResponse = Field(..., description="Objeto recém-criado validado")


class ProcessamentoTextoRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Texto ou manchete de notícia sobre sustentabilidade para análise (entre 5 e 2000 caracteres)."
    )


class ProcessamentoTextoResponse(BaseModel):
    texto_analisado: str = Field(..., description="Texto original submetido à inferência")
    modelo_utilizado: str = Field(..., description="Identificador do modelo local HuggingFace")
    tarefa: str = Field(..., description="Tipo de processamento de linguagem natural executado")
    sentimento: str = Field(..., description="Classificação resultante (Positivo, Neutro ou Negativo)")
    confianca: float = Field(..., description="Score de probabilidade da predição (0.0 a 1.0)")
    tempo_processamento_ms: float = Field(..., description="Tempo gasto pela inferência local em milissegundos")

class AgenteDecisaoRequest(BaseModel):
    foco_analise: str = Field(
        default="Avaliação geral de riscos ambientais e governança urbana (ODS 11 e 12)",
        min_length=5,
        max_length=500,
        description="Direcionamento ou pergunta estratégica do gestor para o agente ESG."
    )
    limite_noticias: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Quantidade de notícias da base real a serem analisadas pelo agente."
    )


class AgenteDecisaoResponse(BaseModel):
    foco_analise: str
    modelo_generativo: str
    total_noticias_analisadas: int
    distribuicao_sentimentos: dict
    prompt_utilizado: str
    parecer_agente: str

# ==========================================================
# 3. HANDLER GLOBAL DE EXCEÇÕES DE VALIDAÇÃO (HTTP 422)
# ==========================================================
@app.exception_handler(RequestValidationError)
async def manipulador_erro_validacao(request: Request, exc: RequestValidationError):
    """Intercepta falhas de schema JSON (ex: campo ausente ou acima de 2000 caracteres) e retorna mensagem amigável."""
    campos_invalidos = [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()]
    corpo_erro = ErroHTTPResponse(
        codigo_http=status.HTTP_422_UNPROCESSABLE_ENTITY,
        erro="Dados de Entrada Inválidos",
        detail=f"A requisição não atende ao formato esperado: {'; '.join(campos_invalidos)}",
        sugestao="Verifique se o JSON possui o campo obrigatório 'texto' dentro do limite de 2000 caracteres."
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=corpo_erro.model_dump()
    )


# ==========================================================
# 4. CARREGAMENTO INICIAL DA BASE DE DADOS EM MEMÓRIA
# ==========================================================
def carregar_base_dados():
    caminho_estatico = os.path.join("data", "raw", "noticias_sustentabilidade.csv")
    caminho_dinamico = os.path.join("data", "raw", "noticias_dinamicas.csv")
    
    dfs = []
    if os.path.exists(caminho_estatico):
        df_est = pd.read_csv(caminho_estatico)
        if "Fonte" not in df_est.columns:
            df_est["Fonte"] = "BeautifulSoup_Static"
        dfs.append(df_est)
        
    if os.path.exists(caminho_dinamico):
        df_din = pd.read_csv(caminho_dinamico)
        if "Fonte" not in df_din.columns:
            df_din["Fonte"] = "Selenium_Dynamic"
        dfs.append(df_din)
        
    if dfs:
        df_consolidado = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["Titulo"])
        df_consolidado["Resumo"] = df_consolidado["Resumo"].fillna("Resumo não disponível.")
        df_consolidado["Fonte"] = df_consolidado["Fonte"].fillna("Desconhecida")
        df_consolidado["id"] = range(1, len(df_consolidado) + 1)
        return df_consolidado
    
    return pd.DataFrame(columns=["id", "Titulo", "Resumo", "Link", "Fonte"])

base_noticias = carregar_base_dados()


# ==========================================================
# 5. ROTAS DA API COM RESPONSE MODELS E EXCEÇÕES HTTP
# ==========================================================
@app.get("/", response_model=StatusAPIResponse, summary="Rota Raiz e Status do Modelo", tags=["Geral"])
def rota_raiz():
    """Retorna o status operacional da API validado pelo `StatusAPIResponse`."""
    return StatusAPIResponse(
        status="Online",
        mensagem="API ESG & ODS 11 operacional.",
        llm_local_carregado=classificador_llm is not None,
        modelo=NOME_MODELO_LLM
    )


@app.get("/noticias", response_model=List[NoticiaResponse], summary="Listar Notícias", tags=["Notícias"])
def listar_noticias():
    """Retorna a lista completa de notícias validada como lista de `NoticiaResponse`."""
    try:
        return base_noticias.fillna("").to_dict(orient="records")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao serializar a base de notícias: {str(e)}"
        )


@app.get(
    "/noticias/{noticia_id}",
    response_model=NoticiaResponse,
    responses={404: {"model": ErroHTTPResponse}},
    summary="Consultar Notícia por ID",
    tags=["Notícias"]
)
def obter_noticia_por_id(noticia_id: int):
    """Consulta uma notícia específica pelo ID validada por `NoticiaResponse`."""
    if noticia_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O identificador da notícia deve ser um número inteiro positivo maior que zero."
        )
    registro = base_noticias[base_noticias["id"] == noticia_id]
    if registro.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nenhuma notícia encontrada com o ID {noticia_id}."
        )
    return registro.fillna("").to_dict(orient="records")[0]


@app.post(
    "/noticias",
    response_model=CadastroNoticiaResponse,
    summary="Cadastrar Nova Notícia",
    tags=["Notícias"],
    status_code=status.HTTP_201_CREATED
)
def cadastrar_noticia(nova_noticia: NoticiaRequest):
    """Insere um novo registro de notícia e retorna o contrato `CadastroNoticiaResponse`."""
    global base_noticias
    novo_id = len(base_noticias) + 1 if not base_noticias.empty else 1
    
    novo_registro = {
        "id": novo_id,
        "Titulo": nova_noticia.Titulo.strip(),
        "Resumo": nova_noticia.Resumo.strip(),
        "Link": nova_noticia.Link.strip(),
        "Fonte": nova_noticia.Fonte or "API_Post"
    }
    
    df_novo = pd.DataFrame([novo_registro])
    base_noticias = pd.concat([base_noticias, df_novo], ignore_index=True)
    
    return CadastroNoticiaResponse(
        mensagem="Notícia cadastrada com sucesso via API!",
        registro=NoticiaResponse(**novo_registro)
    )

@app.post(
    "/agente_decisao",
    response_model=AgenteDecisaoResponse,
    responses={400: {"model": ErroHTTPResponse}, 500: {"model": ErroHTTPResponse}},
    summary="Executar Agente Inteligente de Decisão ESG",
    tags=["Inteligência Artificial (LLM)"]
)
def rota_agente_decisao(payload: AgenteDecisaoRequest):
    """
    Executa o Agente Inteligente ESG: consolida os dados coletados, classifica
    sentimentos com DistilBERT e gera recomendações via Prompt Engineering com Qwen2.5-1.5B.
    """
    try:
        resultado = executar_agente_decisao(
            foco_usuario=payload.foco_analise.strip(),
            limite_noticias=payload.limite_noticias
        )
        return AgenteDecisaoResponse(
            foco_analise=resultado["foco_analise"],
            modelo_generativo=MODELO_GENERATIVO,
            total_noticias_analisadas=resultado["total_noticias_analisadas"],
            distribuicao_sentimentos=resultado["distribuicao_sentimentos"],
            prompt_utilizado=resultado["prompt_utilizado"],
            parecer_agente=resultado["parecer_agente"]
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha ao executar o agente inteligente ESG: {str(e)}"
        )

@app.post(
    "/processar_texto",
    response_model=ProcessamentoTextoResponse,
    responses={
        400: {"model": ErroHTTPResponse},
        422: {"model": ErroHTTPResponse},
        500: {"model": ErroHTTPResponse},
        503: {"model": ErroHTTPResponse},
        504: {"model": ErroHTTPResponse}
    },
    summary="Processar Texto com LLM Local",
    tags=["Inteligência Artificial (LLM)"]
)
def processar_texto_llm(payload: ProcessamentoTextoRequest):
    """
    Executa análise de sentimento local via DistilBERT com validação de entrada,
    controle de tempo de execução e tratamento específico de exceções HTTP.
    """
    # 1. Verifico disponibilidade do modelo na memória local (503)
    if classificador_llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço de IA indisponível: o modelo LLM local não foi carregado na memória do servidor."
        )

    # 2. Valido qualidade semântica mínima do texto de entrada (400)
    texto_limpo = payload.texto.strip()
    if len(texto_limpo) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Texto insuficiente: forneça pelo menos 5 caracteres válidos (espaços em branco são desconsiderados)."
        )

    if not any(c.isalpha() for c in texto_limpo):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conteúdo inválido: o texto deve conter palavras/letras para análise semântica, não apenas números ou símbolos."
        )

    # 3. Executo a inferência monitorando tempo e exceções internas (500 / 504)
    try:
        inicio = time.perf_counter()
        resultado = classificador_llm(texto_limpo[:512])[0]
        duracao_seg = time.perf_counter() - inicio

        if duracao_seg > TEMPO_MAXIMO_INFERENCIA_SEG:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Tempo limite excedido: a inferência levou {duracao_seg:.2f}s (máximo permitido: {TEMPO_MAXIMO_INFERENCIA_SEG}s)."
            )

        mapa_rotulos = {
            "positive": "Positivo",
            "neutral": "Neutro",
            "negative": "Negativo"
        }
        rotulo_original = resultado.get("label", "").lower()
        sentimento_pt = mapa_rotulos.get(rotulo_original, rotulo_original.capitalize())
        confianca = round(float(resultado.get("score", 0.0)), 4)

        return ProcessamentoTextoResponse(
            texto_analisado=texto_limpo,
            modelo_utilizado=NOME_MODELO_LLM,
            tarefa="Classificação de Sentimento ESG",
            sentimento=sentimento_pt,
            confianca=confianca,
            tempo_processamento_ms=round(duracao_seg * 1000, 2)
        )

    except HTTPException:
        raise
    except Exception as erro_inferencia:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno durante a inferência do modelo local: {str(erro_inferencia)}"
        )