import base64
from io import BytesIO

import audioread
import av
import librosa
import numpy as np


SAMPLE_RATE=16000
def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True

# def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
#     audios = []
#     if isinstance(conversations[0], dict):
#         conversations = [conversations]
#     for conversation in conversations:
#         for message in conversation:
#             if not isinstance(message["content"], list):
#                 continue
#             for ele in message["content"]:
#                 if ele["type"] == "audio":
#                     if "audio" in ele or "audio_url" in ele:
#                         path = ele.get("audio", ele.get("audio_url"))
#                         audio = librosa.load(path, sr=16000)[0]
#                         audios.append(audio)
#                     else:
#                         raise ValueError("Unknown audio {}".format(ele))
#                 elif use_audio_in_video and ele["type"] == "video":
#                     if "video" in ele or "video_url" in ele:
#                         path = ele.get("video", ele.get("video_url"))
#                         audio_start = ele.get("video_start", 0.0)
#                         audio_end = ele.get("video_end", None)
#                         assert _check_if_video_has_audio(
#                             path
#                         ), "Video must has audio track when use_audio_in_video=True"
#                         if path.startswith("http://") or path.startswith("https://"):
#                             data = audioread.ffdec.FFmpegAudioFile(path)
#                         elif path.startswith("file://"):
#                             data = path[len("file://") :]
#                         else:
#                             data = path
#                     else:
#                         raise ValueError("Unknown video {}".format(ele))
#                 else:
#                     continue
#     if len(audios) == 0:
#         audios = None
#     return audios


def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio, sr = librosa.load(path, sr=16000, mono=True)

                        processed_audio = process_audio_by_task(audio, sr, ele)
                        audios.append(processed_audio)
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        if path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue
    if len(audios) == 0:
        audios = None
    return audios


def process_audio_by_task(audio: np.ndarray, sr: int, ele: dict) -> np.ndarray:

    if len(audio) < sr:
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)

    if 'task' not in ele:
        return audio
    
    task = ele['task']
    length = len(audio)

    if task == 'avqa':
        tot = 60
        nums_per_second = int(length / tot)
        indices = [i for i in range(0, 60, 6)]
        segments = []
        for indice in indices:
            start_time = max(0, indice - 0.5)
            end_time = min(tot, indice + 1.5)
            audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
            
            if indice - 0.5 < 0:
                sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                audio_seg = np.concatenate((sil, audio_seg), axis=0)
            if indice + 1.5 > tot:
                sil = np.zeros(2 * nums_per_second - len(audio_seg), dtype=float)
                audio_seg = np.concatenate((audio_seg, sil), axis=0)
            segments.append(audio_seg)
        processed_audio = np.concatenate(segments, axis=0)

    elif task in ['avqa_thu', 'ks', 'ucf', 'avvp', 'ave', 'a2v', 'v2a', 'unav', 'avcap']:
        tot = 10
        nums_per_second = int(length / tot)
        
        segments = []
        for i in range(tot):
            start_time = max(0, i)
            end_time = min(tot, i + 1)
            audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
            
            if len(audio_seg) < 1 * nums_per_second:
                sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                audio_seg = np.concatenate((audio_seg, sil), axis=0)
            
            segments.append(audio_seg)
        processed_audio = np.concatenate(segments, axis=0)

    elif task in ['mer24', 'meld', 'dfew', 'mafw', 'cremad']:
        tot = 4
        samples_per_second = sr
        
        segments = []
        for i in range(tot):
            start_sample = i * samples_per_second
            end_sample = (i + 1) * samples_per_second
            audio_seg = audio[start_sample:end_sample] if len(audio) > end_sample else audio[start_sample:]
            
            if len(audio_seg) < samples_per_second:
                padding_needed = samples_per_second - len(audio_seg)
                silence_padding = np.zeros(padding_needed, dtype=float)
                audio_seg = np.concatenate((audio_seg, silence_padding))
            
            segments.append(audio_seg)
        processed_audio = np.concatenate(segments, axis=0)

    elif task == 'arig':
        tot = 5
        nums_per_second = int(length / tot)
        
        segments = []
        for i in range(tot):
            start_time = max(0, i)
            end_time = min(tot, i + 1)
            audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
            
            if len(audio_seg) < 1 * nums_per_second:
                sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                audio_seg = np.concatenate((audio_seg, sil), axis=0)
            
        
            segments.append(audio_seg)
        processed_audio = np.concatenate(segments, axis=0)

    elif task in ['s4', 'ms3']:
        tot = 5
        nums_per_second = int(length / tot)
        i = ele.get('idx', 0)
        
        start_time = max(0, i)
        end_time = min(tot, i + 1)
        start_sample = int(start_time * nums_per_second)
        end_sample = int(nums_per_second * end_time)
        audio_seg = audio[start_sample:end_sample]
        
        if len(audio_seg) < nums_per_second:
            sil = np.zeros(nums_per_second - len(audio_seg), dtype=float)
            audio_seg = np.concatenate((audio_seg, sil), axis=0)
        elif len(audio_seg) > nums_per_second:
            audio_seg = audio_seg[:nums_per_second]
        
        processed_audio = audio_seg

    elif task == 'ref_avs':
        tot = 10
        nums_per_second = int(length / tot)
        i = ele.get('idx', 0)
        
        start_time = max(0, i)
        end_time = min(tot, i + 1)
        start_sample = int(start_time * nums_per_second)
        end_sample = int(nums_per_second * end_time)
        audio_seg = audio[start_sample:end_sample]
        
        if len(audio_seg) < nums_per_second:
            sil = np.zeros(nums_per_second - len(audio_seg), dtype=float)
            audio_seg = np.concatenate((audio_seg, sil), axis=0)
        elif len(audio_seg) > nums_per_second:
            audio_seg = audio_seg[:nums_per_second]
        
        processed_audio = audio_seg

    else:
        processed_audio = audio
    
    return processed_audio