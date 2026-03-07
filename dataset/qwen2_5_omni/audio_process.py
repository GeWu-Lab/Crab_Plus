import audioread
import av
import librosa
import numpy as np


def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if use_audio_in_video and ele["type"] =="audio":
                    if "audio" in ele:
                        path = ele["audio"]                         
                        audio= librosa.load(path, sr=16000)[0]
                        audio_segments = sample_10_audio_segments(audio, sr=16000)
                        audios.append(audio_segments)
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                elif ele["type"] == "audio":
                    if "audio" in ele:
                        path = ele["audio"]                         
                        audio = librosa.load(path, sr=16000)[0]
                        audios.append(audio)
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
    if len(audios) == 0:
        audios = None
    return audios
    

        