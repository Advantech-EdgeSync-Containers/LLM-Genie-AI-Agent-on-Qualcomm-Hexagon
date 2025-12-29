import os
from genie_llm_conf import GenieLLM
from langchain.callbacks.base import BaseCallbackHandler

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy-key")

def to_int(value):
    return int(value) if value and value.isdigit() else None


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def get_genie_llm(callback_handler=None):
    class MyLoggingHandler(BaseCallbackHandler):
        def on_llm_start(self, *args, **kwargs):
            print("\n=== MLC Request Log Start ===",flush=True)

        def on_llm_end(self, *args, **kwargs):
            print("\n=== MLC Request Log End ===",flush=True)

    return GenieLLM(
        callback_handler=callback_handler or MyLoggingHandler()
    )
