"""
Strict RAG Engine for OLED Assistant
Aligned with notebooks/OLED_assistant_v3_final.ipynb
"""
import math
import os

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config
from utils import logger


def create_llm(model_name: str, temperature: float):
    """
    Create Mistral LLM via Ollama local server.
    
    Args:
        model_name: Ollama model name (e.g., "mistral-nemo")
        temperature: Response diversity (0.0 = deterministic, 1.0 = creative)
    
    Returns:
        ChatOpenAI: LLM instance connected to Ollama server
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=config.LLM_BASE_URL,
        api_key="ollama",  # Required parameter for ChatOpenAI (Ollama ignores it)
    )


def create_embeddings():

    try:
        import torch

        # Device selection: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Using device for embeddings: {device}")

        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,   # cosine similarity
                "batch_size": config.EMBEDDING_BATCH_SIZE,
            },
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        raise


def get_vectorstore(embeddings):
    """
    Deploy assumes DB already exists; if missing, fail fast with a clear message.
    """
    if os.path.exists(config.DB_PATH) and os.listdir(config.DB_PATH):
        vectorstore = Chroma(
            persist_directory=config.DB_PATH,
            embedding_function=embeddings,
        )
        return vectorstore

    raise FileNotFoundError(
        f"ChromaDB not found or empty at: {config.DB_PATH}. "
        f"Please copy the persisted DB folder to this path before running the app."
    )


class StrictRAGAssistant:
    """
    Strict RAG System: Answers questions ONLY based on provided documents.
    Returns 'No Answer' if information is missing or question is off-topic.
    """
    
    def __init__(
        self,
        vectorstore,
        llm_model,
        relevance_threshold,
        top_k,
        temperature,
        sigmoid_midpoint,
        sigmoid_steepness,
    ):
        """
        Signature aligned with OLED_assistant_v3_final.ipynb.
        """
        self.vectorstore = vectorstore
        self.relevance_threshold = relevance_threshold
        self.top_k = top_k
        self.sigmoid_midpoint = sigmoid_midpoint
        self.sigmoid_steepness = sigmoid_steepness

        # Create Mistral LLM (using Ollama local server)
        self.llm = create_llm(model_name=llm_model, temperature=temperature)

        # Strict RAG prompt
        rag_prompt_template = """You are an OLED Display Technical Assistant.
Answer the question using the provided context documents as your PRIMARY source.

RULES:
1. Always read the Context carefully and base your answer as much as possible on the Context.
2. If the Context contains partial but relevant information, you MAY use your own OLED/physics knowledge to fill in missing logical steps.
3. ONLY when the Context is clearly irrelevant or provides almost no signal, say: "Information not found in the provided OLED documents."
4. Never contradict the facts given in the Context.
5. Do NOT hallucinate specific numbers, experimental conditions, or paper titles that are not supported by the Context.

Context: {context}

Question: {question}

Answer:"""
        
        self.rag_prompt = PromptTemplate(
            template=rag_prompt_template,
            input_variables=["context", "question"]
        )
        
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
            chain_type_kwargs={"prompt": self.rag_prompt}
        )

    def get_relevance_score(self, query):
        """Calculate relevance score using sigmoid transformation."""
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        if not docs_with_scores:
            return 0.0
        
        # Chroma returns distance (lower is better). Convert to similarity.
        scores = [1.0 / (1.0 + score) for _, score in docs_with_scores]
        avg_score = sum(scores) / len(scores)
        
        # Sigmoid transformation
        sigmoid_score = 1 / (1 + math.exp(-self.sigmoid_steepness * (avg_score - self.sigmoid_midpoint)))
        return sigmoid_score

    def query(self, question):
        """Process query through Strict RAG logic."""
        relevance_score = self.get_relevance_score(question)
        
        result = {
            "answer": None,
            "mode": None,
            "relevance_score": relevance_score,
            "retrieved_docs": []
        }
        
        # Check relevance threshold
        if relevance_score >= self.relevance_threshold:
            logger.info(f"✅ High relevance ({relevance_score:.3f}). Executing RAG.")
            result["mode"] = "RAG"
            
            # Get documents for display
            docs = self.vectorstore.similarity_search(question, k=self.top_k)
            result["retrieved_docs"] = docs
            
            # Execute Chain
            try:
                rag_response = self.rag_chain.invoke({"query": question})["result"]
                result["answer"] = rag_response
                
                # Check for "Information not found" response from LLM
                if "Information not found" in rag_response or ("provided context" in rag_response and "does not contain" in rag_response):
                    result["mode"] = "NO_ANSWER_IN_DOCS"
                    result["answer"] = "No Answer: The relevant content is not found in RAG documents."
                    logger.info("❌ Documents found but LLM could not find answer in context.")
                    
            except Exception as e:
                logger.error(f"RAG Chain execution failed: {str(e)}")
                result["answer"] = "Error processing request."
                
        else:
            logger.info(f"🚫 Low relevance ({relevance_score:.3f}). Rejecting.")
            result["mode"] = "OFF_TOPIC"
            result["answer"] = "No Answer: The question is not related to OLED display or relevant documents are not available."
            
        return result
