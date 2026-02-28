"""
Strict RAG Engine for OLED Assistant
Aligned with notebooks/OLED_assistant_v3_final.ipynb
"""
import math

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import config
from document_pipeline import create_embeddings_model, get_or_create_vectorstore
from utils import logger


def create_llm(model_name: str, temperature: float):
    """
    Create OpenAI-compatible chat model for cloud deployment.
    
    Args:
        model_name: OpenAI model name (e.g., "gpt-4o-mini")
        temperature: Response diversity (0.0 = deterministic, 1.0 = creative)
    
    Returns:
        ChatOpenAI: LLM instance using OPENAI_API_KEY from environment
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )


def create_embeddings():
    try:
        # Keep this wrapper for backward compatibility with app.py import.
        return create_embeddings_model()
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        raise


def get_vectorstore(embeddings):
    """
    Keep a stable interface while delegating lifecycle logic to document_pipeline.
    """
    return get_or_create_vectorstore(
        embeddings=embeddings,
        docs_folder=config.DOCS_FOLDER,
        persist_directory=config.DB_PATH,
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

        # Create cloud LLM (OpenAI API)
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
        
        # Chroma returns L2 distance (lower is better). Convert to similarity.
        # For normalized embeddings, L2 distance relates to cosine similarity:
        #   sim = 1 - (d^2)/2  (then clamp to [0, 1])
        scores = []
        for _, d in docs_with_scores:
            sim = 1.0 - (float(d) * float(d)) / 2.0
            scores.append(max(0.0, min(1.0, sim)))
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
