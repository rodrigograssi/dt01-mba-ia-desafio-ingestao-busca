import os
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

for key in ["OPENROUTER_API_KEY", "MODEL_EMBEDDING", "PG_URL", "PG_VECTOR_COLLECTION"]:
    if not os.getenv(key):
        raise RuntimeError(f"Variavel de ambiente {key} nao esta definida")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MODELO_EMBEDDING = os.getenv("MODEL_EMBEDDING")
COLLECTION = os.getenv("PG_VECTOR_COLLECTION")

ROOT_DIR = Path(__file__).resolve().parent.parent  # src/ -> raiz do projeto
pdf_path = ROOT_DIR / "arq.pdf"
arquivo = pdf_path.name


def hash_ingestao(pdf_path: Path) -> str:
    """Hash do conteudo do PDF + configuracao da ingestao.
    Se o arquivo OU a configuracao mudar, o hash muda e o arquivo e reprocessado."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while bloco := f.read(8192):
            h.update(bloco)
    config = {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "modelo": MODELO_EMBEDDING}
    h.update(json.dumps(config, sort_keys=True).encode())
    return h.hexdigest()


# --- 1. Curadoria: o arquivo ja foi processado com este hash? -----------------
engine = create_engine(os.getenv("PG_URL"))
novo_hash = hash_ingestao(pdf_path)

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ingestao_arquivos (
            arquivo       TEXT PRIMARY KEY,
            hash          TEXT NOT NULL,
            total_chunks  INT,
            processado_em TIMESTAMPTZ DEFAULT now()
        )"""))
    hash_atual = conn.execute(
        text("SELECT hash FROM ingestao_arquivos WHERE arquivo = :a"), {"a": arquivo}
    ).scalar()

if hash_atual == novo_hash:
    print(f"{arquivo} sem alteracoes (hash {novo_hash[:12]}), nada a fazer.")
    raise SystemExit(0)

print(f"{arquivo}: {'novo arquivo' if hash_atual is None else 'arquivo alterado'}, processando...")

# --- 2. Carrega, quebra em chunks e enriquece os metadados --------------------
docs = PyPDFLoader(file_path=str(pdf_path)).load()

splits = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=False,
).split_documents(docs)

if not splits:
    raise RuntimeError("Nao foi possivel gerar os chunks do documento")

agora = datetime.now(timezone.utc).isoformat()

enriched_docs = []
for i, d in enumerate(splits):
    meta = {k: v for k, v in d.metadata.items() if v not in ("", None)}
    meta.update({
        "arquivo": arquivo,
        "hash_arquivo": novo_hash,
        "inserido_em": agora,
        "chunk": i,
        "pagina": d.metadata.get("page", 0) + 1,  # PyPDF comeca a contar em 0
    })
    enriched_docs.append(Document(page_content=d.page_content, metadata=meta))

# Prefixo com o hash: a versao nova nunca sobrescreve a antiga durante a troca
ids = [f"{novo_hash[:16]}-{i}" for i in range(len(enriched_docs))]

# --- 3. Insere a versao nova --------------------------------------------------
embedding_model = OpenAIEmbeddings(
    model=MODELO_EMBEDDING,  # no OpenRouter: "openai/text-embedding-3-small"
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    check_embedding_ctx_length=False,
)

store = PGVector(
    embeddings=embedding_model,
    connection=os.getenv("PG_URL"),
    collection_name=COLLECTION,
    use_jsonb=True,
)

store.add_documents(documents=enriched_docs, ids=ids)

# --- 4. Apaga a versao antiga e registra o sucesso (mesma transacao) ----------
with engine.begin() as conn:
    removidos = conn.execute(text("""
        DELETE FROM langchain_pg_embedding e
        USING langchain_pg_collection c
        WHERE c.uuid = e.collection_id
          AND c.name = :col
          AND e.cmetadata->>'arquivo' = :a
          AND (e.cmetadata->>'hash_arquivo' IS DISTINCT FROM :h)
    """), {"col": COLLECTION, "a": arquivo, "h": novo_hash}).rowcount

    conn.execute(text("""
        INSERT INTO ingestao_arquivos (arquivo, hash, total_chunks, processado_em)
        VALUES (:a, :h, :n, now())
        ON CONFLICT (arquivo) DO UPDATE
           SET hash = EXCLUDED.hash,
               total_chunks = EXCLUDED.total_chunks,
               processado_em = now()
    """), {"a": arquivo, "h": novo_hash, "n": len(enriched_docs)})

print(f"{len(enriched_docs)} chunks inseridos, {removidos} chunks da versao antiga removidos.")
