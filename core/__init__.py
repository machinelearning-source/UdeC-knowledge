import os
import platform
import tempfile

from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def default_cache_dir():
    """Return the folder used to cache the embedding model.

    Priority: FASTEMBED_CACHE_DIR env var, then a project-local folder
    (persists between builds/runs on Linux hosts such as Render), and the
    OS temp folder on Windows where project paths may exceed 260 chars.
    """
    env_dir = os.environ.get("FASTEMBED_CACHE_DIR")
    if env_dir:
        return env_dir
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if platform.system() == "Windows":
        return os.path.join(tempfile.gettempdir(), "fastembed_cache")
    return os.path.join(project_root, ".fastembed_cache")


class FastEmbedEmbeddings(Embeddings):
    """Embeddings locales con fastembed (ONNX, sin GPU, sin API key).

    Se usa en reemplazo de langchain_community.embeddings.FastEmbedEmbeddings
    porque ese presentaba un error con pydantic 2 (el modelo quedaba en None).
    """

    def __init__(self, model_name=DEFAULT_MODEL, cache_dir=None, threads=1):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text):
        return next(iter(self.model.embed([text]))).tolist()
