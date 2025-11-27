from .whisperx_node import WhisperXNode

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "WhisperXNode": WhisperXNode,
}

# Perbaikan: Isi dictionary ini agar nama node muncul cantik di UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "WhisperXNode": "WhisperX Transcription"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
