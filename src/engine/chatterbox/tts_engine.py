import utils
import config
import numpy as np
from typing import Optional
from engine.tts_engine import TTSEngine


class ChatterboxTTSEngine(TTSEngine):
    def __init__(self):
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(config.DEVICE)

    def generate_audio(
        self,
        text: str,
        voice: str,
        output_format: str,
        speed: float = 1.0,
        chunk_size: int = 250,
        seed: int = 0,
        params: dict = {},
    ) -> Optional[bytes]:
        if seed != 0:
            utils.set_seed(seed)  # For reproducibility

        voice_file = config.VOICES_DIR + f"{voice}.wav"

        all_audio_data = []

        chunks = utils.chunk_text_by_sentences(text, chunk_size)
        sample_rate = self.model.sr

        # split in chunks
        for chunk in chunks:
            print(f"Generating audio for chunk: {chunk}")

            # Generate the waveform
            audio_tensor = self.model.generate(
                chunk,
                audio_prompt_path=voice_file,
                # exaggeration=params.get("exaggeration", None),
                temperature=params.get("temperature", 0.8),
                # cfg_weight=params.get("cfg_weight", None),
                # **(
                #     {"language_id": language_id}
                #     if model_name == "Chatterbox-Multilingual"
                #     else {}
                # ),
            )

            # adjust speed
            tensor_tuple = utils.apply_speed_factor(audio_tensor, sample_rate, speed)
            audio_tensor = tensor_tuple[0]

            audio_data = audio_tensor.squeeze(0).numpy()
            audio_data = np.clip(audio_data, -1.0, 1.0)  # Clip to prevent saturation
            audio_data = (audio_data * 32767).astype(np.int16)
            all_audio_data.append(audio_data)

        all_audio_data = np.concatenate(all_audio_data)
        bytes_object = utils.encode_audio(all_audio_data, sample_rate, output_format)

        return bytes_object
