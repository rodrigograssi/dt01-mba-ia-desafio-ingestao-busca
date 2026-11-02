import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

for key in ["OPENROUTER_API_KEY", "MODEL_EMBEDDING", "MODEL_LLM", "PG_URL", "PG_VECTOR_COLLECTION"]:
    if not os.getenv(key):
        raise RuntimeError(f"Variavel de ambiente {key} nao esta definida")

MODELO_EMBEDDING = os.getenv("MODEL_EMBEDDING")
COLLECTION = os.getenv("PG_VECTOR_COLLECTION")


system_prompt = """Você é um assistente que responde perguntas sobre o conteúdo de PDFs ingeridos no sistema.

REGRAS:
1. Responda usando EXCLUSIVAMENTE as informações do CONTEXTO fornecido.
2. Não use conhecimento externo, mesmo que você saiba a resposta.
3. Nunca invente informações.
4. Não faça suposições, extrapolações nem cálculos que não estejam no CONTEXTO
   (ex.: o CONTEXTO fala de 2023 e a pergunta é sobre 2024).
5. Responda de forma clara, concisa e direta, sem preâmbulos como "Com base no contexto...".
6. Se a resposta não estiver explicitamente no CONTEXTO, responda exatamente:
   "Não tenho informações necessárias para responder sua pergunta."

Os exemplos abaixo são apenas ilustrativos do formato.
NÃO use as informações deles para responder.

Exemplo 1 (a resposta está no CONTEXTO):
CONTEXTO: "A padaria Pão Dourado vende 300 pães por dia."
PERGUNTA: Quantos pães a padaria vende por dia?
RESPOSTA: A padaria vende 300 pães por dia.

Exemplo 2 (a resposta NÃO está no CONTEXTO):
CONTEXTO: "A padaria Pão Dourado vende 300 pães por dia."
PERGUNTA: Qual o preço do pão?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

Exemplo 3 (conhecimento geral, mas fora do CONTEXTO):
CONTEXTO: "A padaria Pão Dourado vende 300 pães por dia."
PERGUNTA: Qual a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta."""

user_prompt = """CONTEXTO:
{contexto}

PERGUNTA: {pergunta}"""

embedding_model = OpenAIEmbeddings(
    model=MODELO_EMBEDDING,  # no OpenRouter: "openai/text-embedding-3-small"
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    check_embedding_ctx_length=False,
)


llm = ChatOpenAI(
    model=os.getenv("MODEL_LLM"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)

store = PGVector(
    embeddings=embedding_model,
    connection=os.getenv("PG_URL"),
    collection_name=COLLECTION,
    use_jsonb=True,
)

chat_prompt = ChatPromptTemplate([
    ("system",system_prompt),
    ("user",user_prompt)
])

chain = chat_prompt | llm

while True:
    try:
        pergunta = input("Digite sua pergunta: ").strip()
        
    except (KeyboardInterrupt, EOFError):
        print("\nEncerrando o chat.")
        break
    
    if not pergunta:
        continue

    if pergunta.lower() in ["sair", "exit", "quit"]:
        print("Encerrando o chat.")
        break

    try:

        result = store.similarity_search_with_score(query=pergunta, k=5)
#        for doc, score in result:
#             print(f"[{score:.3f}] pág {doc.metadata.get('pagina')}: {doc.page_content[:80]}")

        contexto = "\n\n---\n\n".join(doc.page_content for doc, score in result)

        resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta})
        print(f"RESPOSTA: {resposta.content}")

    except Exception as e:
        print(f"Erro ao processar a pergunta: {e}")
        continue    