from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FastEmbedEmbeddings(Embeddings):
    """Embeddings locales con fastembed (ONNX, sin GPU, sin API key).

    Se usa en reemplazo de langchain_community.embeddings.FastEmbedEmbeddings
    porque ese presentaba un error con pydantic 2 (el modelo quedaba en None).
    """

    def __init__(self, model_name=DEFAULT_MODEL, cache_dir=None):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text):
        return next(iter(self.model.embed([text]))).tolist()
