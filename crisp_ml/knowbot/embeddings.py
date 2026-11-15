import os
import platform
import tempfile

import numpy as np
from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Slim mode, so that the app fits in 512 MB of RAM (measured: about 740 MB without it).
# 1. The ONNX model stores its 250,002 x 384 vocabulary table in float16 (183 MB, 83% of
#    the file) and onnxruntime keeps a float32 copy of it. The table is moved out of the
#    graph into a memory-mapped .npy file, and only the rows of the tokens of each question
#    are read; the rest of the network is unchanged.
# 2. The tokenizers library needs about 250 MB for this 250,000-piece vocabulary. The
#    original SentencePiece model of the same tokenizer needs a few MB and gives the same
#    token ids (checked on the 100 questions, the 290 fragments and edge cases).
# The vectors are identical to the ones of the full model.
WORD_TABLE = "embeddings.word_embeddings.weight"
WORD_LOOKUP_OUTPUT = "/embeddings/word_embeddings/Gather_output_0"
SENTENCEPIECE_FILE = "sentencepiece.bpe.model"
MAX_TOKENS = 128  # truncation length of the model's tokenizer.json, as fastembed applies it
BOS_ID, PAD_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


def default_cache_dir():
    """Return the folder used to cache the embedding model.

    Priority: FASTEMBED_CACHE_DIR env var, then a project-local folder
    (persists between builds and runs on Linux hosts and containers), and the
    OS temp folder on Windows where project paths may exceed 260 chars.
    """
    env_dir = os.environ.get("FASTEMBED_CACHE_DIR")
    if env_dir:
        return env_dir
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if platform.system() == "Windows":
        return os.path.join(tempfile.gettempdir(), "fastembed_cache")
    return os.path.join(project_root, ".fastembed_cache")


def build_slim_model(model_path, slim_dir):
    """Split the ONNX model into the encoder graph and the vocabulary table (run once)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    os.makedirs(slim_dir, exist_ok=True)
    model = onnx.load(model_path)
    graph = model.graph
    table = next(t for t in graph.initializer if t.name == WORD_TABLE)
    np.save(os.path.join(slim_dir, "word_embeddings.npy"), numpy_helper.to_array(table))
    graph.initializer.remove(table)
    lookup = next(n for n in graph.node if n.op_type == "Gather" and WORD_TABLE in n.input)
    graph.node.remove(lookup)
    graph.input.append(helper.make_tensor_value_info(
        WORD_LOOKUP_OUTPUT, TensorProto.FLOAT16, ["batch_size", "sequence_length", table.dims[1]]))
    onnx.save(model, os.path.join(slim_dir, "encoder.onnx"))


class SlimSession:
    """Drop-in replacement of the onnxruntime session used by fastembed."""

    def __init__(self, slim_dir, threads=1):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        self.table = np.load(os.path.join(slim_dir, "word_embeddings.npy"), mmap_mode="r")
        self.session = ort.InferenceSession(os.path.join(slim_dir, "encoder.onnx"),
                                            sess_options=options, providers=["CPUExecutionProvider"])

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, output_names, feeds):
        feeds = dict(feeds)
        feeds[WORD_LOOKUP_OUTPUT] = np.asarray(self.table[feeds["input_ids"]])
        return self.session.run(output_names, feeds)


class _Encoding:
    def __init__(self, ids, length):
        self.ids = ids + [PAD_ID] * (length - len(ids))
        self.attention_mask = [1] * len(ids) + [0] * (length - len(ids))


class SentencePieceTokenizer:
    """Same token ids as the model's tokenizer.json (XLM-RoBERTa ids: SentencePiece id + 1)."""

    def __init__(self, model_file):
        import sentencepiece

        self.processor = sentencepiece.SentencePieceProcessor(model_file=model_file)

    def ids(self, text):
        pieces = [UNK_ID if i == 0 else i + 1 for i in self.processor.encode(text)]
        return [BOS_ID] + pieces[: MAX_TOKENS - 2] + [EOS_ID]

    def encode_batch(self, texts):
        batch = [self.ids(text) for text in texts]
        length = max(len(ids) for ids in batch)
        return [_Encoding(ids, length) for ids in batch]


class FastEmbedEmbeddings(Embeddings):
    """Local embeddings with fastembed (ONNX, no GPU, no API key).

    Used instead of langchain_community.embeddings.FastEmbedEmbeddings
    because that class failed with pydantic 2 (the model ended up as None).
    """

    def __init__(self, model_name=DEFAULT_MODEL, cache_dir=None, threads=1, slim=True):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads, lazy_load=slim)
        if slim:
            inner = self.model.model
            model_path = os.path.join(str(inner._model_dir), inner.model_description.model_file)
            slim_dir = os.path.join(cache_dir or default_cache_dir(), "slim-" + model_name.replace("/", "--"))
            if not os.path.exists(os.path.join(slim_dir, "encoder.onnx")):
                build_slim_model(model_path, slim_dir)
            from huggingface_hub import hf_hub_download

            spm_file = hf_hub_download(model_name, SENTENCEPIECE_FILE, cache_dir=cache_dir or default_cache_dir())
            inner.tokenizer = SentencePieceTokenizer(spm_file)
            inner.model = SlimSession(slim_dir, threads=threads)

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text):
        return next(iter(self.model.embed([text]))).tolist()
