from engine.tts_engine import TTSEngine
from engine.chatterbox.tts_engine import ChatterboxTTSEngine
from engine.mira.tts_engine import MiraTTSEngine


def get_available_model_engines() -> dict[TTSEngine]:
    """
    Returns a list of available model engines.
    """
    return {"chatterbox": ChatterboxTTSEngine, "mira": MiraTTSEngine}
