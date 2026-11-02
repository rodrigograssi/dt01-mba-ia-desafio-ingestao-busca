# Ingestão de PDF e Busca Semântica

Sistema RAG simples em Python: um PDF é quebrado em trechos (chunks), transformado em vetores (embeddings) e gravado no PostgreSQL com **pgvector**. Depois, um chat via linha de comando responde perguntas **usando apenas o conteúdo do PDF**.

```
arq.pdf ──► 01-ingestor-pdf.py ──► Postgres (pgvector) ──► 02-chat.py ──► resposta no terminal
```

## Tecnologias

- Python 3 + LangChain
- PostgreSQL 16 + pgvector (via Docker)
- OpenRouter para embeddings e LLM

## Estrutura

```
.
├── src/
│   ├── 01-ingestor-pdf.py   # lê o PDF, gera os embeddings e grava no banco
│   └── 02-chat.py           # chat de perguntas e respostas no terminal
├── arq.pdf                  # documento a ser ingerido (LGPD - Lei 13.709/2018)
├── docker-compose.yml       # Postgres + pgvector
├── requirements.txt         # dependências Python
└── .env.example             # modelo das variáveis de ambiente
```

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Uma chave de API do [OpenRouter](https://openrouter.ai/settings/keys)

## Instalação

**1. Crie e ative o ambiente virtual**

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Instale as dependências**

```bash
pip install -r requirements.txt
```

**3. Configure as variáveis de ambiente**

Copie o modelo e preencha com seus valores:

```bash
cp .env.example .env        # Linux / macOS
copy .env.example .env      # Windows
```

| Variável | Descrição | Exemplo |
|---|---|---|
| `OPENROUTER_API_KEY` | Chave de API do OpenRouter | `sk-or-v1-...` |
| `MODEL_EMBEDDING` | Modelo de embedding (com prefixo do fornecedor) | `openai/text-embedding-3-small` |
| `MODEL_LLM` | Modelo de chat que gera as respostas | `openai/gpt-4o-mini` |
| `PG_URL` | String de conexão com o Postgres | `postgresql+psycopg://postgres:postgres@localhost:5432/semantica` |
| `PG_VECTOR_COLLECTION` | Nome da coleção dos chunks | `documentos` |

> A ingestão e o chat precisam usar o **mesmo** `MODEL_EMBEDDING`. Vetores de modelos diferentes não são comparáveis.

**4. Suba o banco de dados**

```bash
docker compose up -d
docker compose ps        # aguarde o status "healthy"
```

> Se a porta 5432 já estiver em uso por outro Postgres, altere para `"5433:5432"` no `docker-compose.yml` e ajuste a porta no `PG_URL`.

## Uso

### 1. Ingerir o PDF

```bash
python src/01-ingestor-pdf.py
```

O script:

1. Calcula um hash do PDF + configuração (tamanho do chunk, overlap e modelo).
2. Se esse hash já foi processado, encerra sem fazer nada.
3. Caso contrário, quebra o PDF em chunks de 500 caracteres (overlap de 100), gera os embeddings e grava no banco.
4. Remove os chunks da versão anterior do arquivo e registra o hash na tabela `ingestao_arquivos`.

Saída esperada:

```
# primeira execução
arq.pdf: novo arquivo, processando...
N chunks inseridos, 0 chunks da versao antiga removidos.

# execuções seguintes, sem alterar o PDF
arq.pdf sem alteracoes (hash xxxxxxxxxxxx), nada a fazer.
```

A ingestão é **idempotente**: pode ser executada várias vezes sem duplicar dados. Se o PDF ou a configuração mudar, o arquivo é reprocessado automaticamente.

### 2. Fazer perguntas

```bash
python src/02-chat.py
```

Digite a pergunta e pressione Enter. Para sair, digite `sair`, `exit` ou `quit` (ou pressione Ctrl+C).

Exemplo (saída real com o `arq.pdf` deste repositório):

```
Digite sua pergunta: Qual o valor máximo da multa por infração?
RESPOSTA: O valor máximo da multa por infração é de R$ 50.000.000,00 (cinquenta milhões de reais).

Digite sua pergunta: O que é dado pessoal sensível?
RESPOSTA: Dado pessoal sensível é dado pessoal sobre origem racial ou étnica, convicção religiosa, opinião política, filiação a sindicato ou a organização de caráter religioso, filosófico ou político, dado referente à saúde ou à vida sexual, dado genético ou biométrico, quando vinculado a uma pessoa natural.

Digite sua pergunta: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

Digite sua pergunta: Quem ganhou a Copa do Mundo de 2002?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

### Sobre o documento de exemplo

O `arq.pdf` é o texto compilado da **Lei Geral de Proteção de Dados (Lei nº 13.709/2018)**, obtido no site do Planalto. Textos de lei são de domínio público (Lei nº 9.610/1998, art. 8º, IV).

Para usar outro documento, substitua o `arq.pdf` e rode a ingestão novamente: o hash muda e os chunks antigos são substituídos automaticamente. Gere o PDF com **"Salvar como PDF"** do navegador; PDFs feitos de imagens (como os do "Microsoft Print to PDF" ou documentos escaneados) não têm texto extraível.

O modelo responde **somente** com base nos trechos do PDF recuperados pela busca semântica. Perguntas fora do conteúdo do documento recebem a mensagem padrão de recusa.

## Como funciona

**Ingestão** (`01-ingestor-pdf.py`)
`PyPDFLoader` lê o PDF → `RecursiveCharacterTextSplitter` divide em chunks → cada chunk recebe metadados (`arquivo`, `pagina`, `chunk`, `hash_arquivo`, `inserido_em`) → `PGVector` gera os embeddings e grava no banco.

**Chat** (`02-chat.py`)
A pergunta vira embedding → `similarity_search_with_score` recupera os 5 chunks mais próximos → os chunks são montados como CONTEXTO em um `ChatPromptTemplate` (system prompt com as regras + user prompt com contexto e pergunta) → o LLM (`temperature=0`) gera a resposta.

## Consultando o banco

```bash
docker exec -it pgvector-db psql -U postgres -d semantica
```

```sql
-- arquivos já ingeridos
SELECT * FROM ingestao_arquivos;

-- quantidade de chunks por coleção
SELECT c.name, count(*)
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON c.uuid = e.collection_id
GROUP BY c.name;
```

## Problemas comuns

| Erro | Causa provável |
|---|---|
| `Variavel de ambiente X nao esta definida` | Falta a variável no `.env` |
| `401 - Missing Authentication header` | `OPENROUTER_API_KEY` inválida, com aspas ou espaços |
| `connection timeout expired` | Container parado ou porta 5432 ocupada; rode `docker compose ps` |
| `FileNotFoundError: arq.pdf` | O PDF não está no caminho esperado pelo ingestor |
