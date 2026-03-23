import json,os,random
from os.path import join,exists
import numpy as np
import torch
import audioread
import av, csv
import librosa
from torch.utils.data import Dataset
from dataclasses import dataclass
import transformers
from typing import Sequence,Dict
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen2VLImageProcessor, WhisperFeatureExtractor, Qwen2TokenizerFast
from transformers.feature_extraction_utils import BatchFeature
from models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from dataset.qwen_omni_utils import process_mm_info

ALL_TASKS = [
    'a2v', 'v2a', 'ks', 'ucf', 'meld', 'mer24', 
    'cremad', 'mafw', 'dfew', 'avqa', 'avqa_thu', 
    'ave', 'unav', 'avvp', 'ms3', 's4', 
    'ref_avs', 'arig', 'avcap'
]

TASK_PATH_REGISTRY = {
    task: {
        'train': f'AVUIE_2/{task}/train.json',
        'test': f'AVUIE_2/{task}/test.json'
    }
    for task in ALL_TASKS
}

class OmniDataset(Dataset):
    def __init__(
        self,
        data_args,
        tokenizer: Qwen2TokenizerFast,
        audio_processor: WhisperFeatureExtractor,
        vision_processor: Qwen2VLImageProcessor,
        mm_processor: Qwen2_5OmniProcessor,
        mode='train',
        **kwargs
    ):
        super().__init__()

        self.samples = []
        self.mode = mode
        self.mm_processor = mm_processor
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.vision_processor = vision_processor
        self.tot = 0
        self.data_root = getattr(data_args, 'data_root', '')

        self.task_loaders = {
            'a2v': self.add_a2v_samples,
            'v2a': self.add_v2a_samples,
            'avqa': self.add_avqa_samples,
            'avqa_thu': self.add_avqa_thu_samples,
            'ave': self.add_ave_samples,
            'unav': self.add_unav_samples,
            'avvp': self.add_avvp_samples,
            'ms3': self.add_ms3_samples,
            's4': self.add_s4_samples,
            'ref_avs': self.add_ref_avs_samples,
            'arig': self.add_arig_samples,
            'avcap': self.add_avcap_samples,
            'meld': self.add_meld_samples,
            'mer24': self.add_mer24_samples,
            'mafw': self.add_mafw_samples,
            'dfew': self.add_dfew_samples,
            'cremad': self.add_cremad_samples,
            'ks': self.add_ks_samples,
            'ucf': self.add_ucf_samples,
        }

        for task_name, loader_func in self.task_loaders.items():
            arg_name = f"{task_name}_task"
            is_enabled = getattr(data_args, arg_name, kwargs.get(arg_name, False))

            if is_enabled:
                print(f"Loading task: {task_name}...")
                if hasattr(loader_func, '__call__'):
                    loader_func()
                else:
                    print(f"Warning: Method for {task_name} is mapped but not implemented.")

        print(f'Total {self.mode} sample nums: {self.tot}')

    def _get_media_path(self, task_name, media_type, filename):
        if not filename: return None
        if media_type:
            return os.path.join(self.data_root, 'AVUIE_2', task_name, media_type, filename)
        else:
            return os.path.join(self.data_root, 'AVUIE_2', task_name, filename)

    def read_label(self, label_path):
        with open(label_path, 'r') as f:
            label = f.read()
        return label

    def _get_task_path(self, task_name):
        if task_name not in TASK_PATH_REGISTRY:
            print(f"Warning: Task {task_name} not explicitly in registry. Using default pattern.")
            relative_path = f"AVUIE_2/{task_name}/{self.mode}.json"
        else:
            relative_path = TASK_PATH_REGISTRY[task_name].get(self.mode)
        
        if not relative_path:
            print(f"Warning: No path config found for task {task_name} in mode {self.mode}, skipping.")
            return None
            
        full_path = os.path.join(self.data_root, relative_path)
        return full_path

    def add_a2v_samples(self):
        task_name = 'a2v'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
            
        for sample in samples:
            video1_path = self._get_media_path(task_name, 'video', sample['video1_path'])
            video2_path = self._get_media_path(task_name, 'video', sample['video2_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            correct_video = str(sample['correct_video'])
            audio_event = sample['audio_event']
            video1_event = sample['video1_event']
            video2_event = sample['video2_event']
            output = sample['label']

            instruction = f"Please listen to the audio and indicate which of the two videos matches what you hear."

            user_content_list = []
            user_content_list.append({'type':'text','text':"These are two videos and an audio:"})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"a2v"})
            user_content_list.append({'type':'video','video':video1_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'video','video':video2_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'text','text':instruction})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'task_name':'a2v'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'a2v sample nums: {tot}')
        self.tot += tot

    def add_v2a_samples(self):
        task_name = 'v2a'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio1_path = self._get_media_path(task_name, 'audio', sample['audio1_path'])
            audio2_path = self._get_media_path(task_name, 'audio', sample['audio2_path'])
            correct_audio = str(sample['correct_audio'])
            audio1_event = sample['audio1_event']
            video_event = sample['video_event']
            audio2_event = sample['audio2_event']
            output = sample['label']

            instruction = f"Please view the video and indicate which of the two audio clips matches what you see."

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and two audio clips:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio1_path,'task':"v2a"})
            user_content_list.append({'type':'audio','audio':audio2_path,'task':"v2a"})
            user_content_list.append({'type':'text','text':instruction})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'task_name':'v2a'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'v2a sample nums: {tot}')
        self.tot += tot

    def add_ks_samples(self):
        task_name = 'ks'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            action = sample['action']
            output = sample['label']
            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"ks"})
            user_content_list.append({'type':'text','text':"Please identify the action taking place in the video."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'task_name':'ks'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'ks sample nums: {tot}')
        self.tot += tot

    def add_ucf_samples(self):
        task_name = 'ucf'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            output = sample['new_label']
            action = sample['label'].lower()

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"ucf"})
            user_content_list.append({'type':'text','text':"Please identify the action taking place in the video."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'task_name':'ucf'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'ucf sample nums: {tot}')
        self.tot += tot

    def add_meld_samples(self):
        task_name = 'meld'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            output = sample['label']
            emotion = sample['Emotion']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':4})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"meld"})
            user_content_list.append({'type':'text','text':"Please judge the person's emotion at the moment."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'task_name':'meld'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'meld sample nums: {tot}')
        self.tot += tot

    def add_mer24_samples(self):
        task_name = 'mer24'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            output = sample['label']
            emotion = sample['discrete']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':4})
            user_content_list.append({'type':'audio','audio':audio_path,"task":"mer24"})
            user_content_list.append({'type':'text','text':"Please judge the person's emotion at the moment."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'task_name':'mer24'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'mer24 sample nums: {tot}')
        self.tot += tot

    def add_cremad_samples(self):
        task_name = 'cremad'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            output = sample['label']
            emotion = sample['emotion']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':4})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"cremad"})
            user_content_list.append({'type':'text','text':"Please judge the person's emotion at the moment."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'task_name':'cremad'
                }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'cremad sample nums: {tot}')
        self.tot += tot

    def add_mafw_samples(self):
        task_name = 'mafw'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            output = sample['label']
            emotion = sample['emotion']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':4})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"mafw"})
            user_content_list.append({'type':'text','text':"Please judge the person's emotion at the moment."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'task_name':'mafw'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'mafw sample nums: {tot}')
        self.tot += tot

    def add_dfew_samples(self):
        task_name = 'dfew'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            output = sample['label']
            emotion = sample['emotion']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':4})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"dfew"})
            user_content_list.append({'type':'text','text':"Please judge the person's emotion at the moment."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'task_name':'dfew'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'dfew sample nums: {tot}')
        self.tot += tot

    def add_avqa_samples(self):
        task_name = 'avqa'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            video_id = sample['video_id']
            question_id = sample['question_id']
            _type = sample['type']
            question = sample['question']
            answer = sample['answer']
            output = sample['label']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"avqa"})
            user_content_list.append({'type':'text','text':f"Please answer this question: {question}"})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'vid':video_id,
                'qid':question_id,
                'type':_type,
                'task_name':'avqa'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'avqa sample nums: {tot}')
        self.tot += tot

    def add_avqa_thu_samples(self):
        task_name = 'avqa_thu'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            _id = sample['id']
            name = sample['video_name']
            video_id = sample['video_id']
            relation = sample['question_relation']
            _type = sample['question_type']
            multi_choice = sample['multi_choice']
            question = sample['question_text']
            answer = sample['answer']
            output = sample['label']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"avqa_thu"})
            user_content_list.append({'type':'text','text':f"Please answer the question based on the options: {question}, options:{multi_choice}"})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'vid':video_id,
                'name':name,
                'task_name':'avqa_thu'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'avqa_thu sample nums: {tot}')
        self.tot += tot

    def add_ave_samples(self):
        task_name = 'ave'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            event = sample['event']
            vid = sample['vid']
            start_time = sample['start_time']
            end_time = sample['end_time']
            output = sample['label_content']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"ave"})
            user_content_list.append({'type':'text','text':"Please describe the events and time range that occurred in the video."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'task_name':'ave'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'ave sample nums: {tot}')
        self.tot += tot

    def add_unav_samples(self):
        task_name = 'unav'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            vid = sample['vid']
            output = sample['label_content']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"unav"})
            user_content_list.append({'type':'text','text':"Please describe the events and time range that occurred in the video."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'task_name':'unav'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'unav sample nums: {tot}')
        self.tot += tot

    def add_avvp_samples(self):
        task_name = 'avvp'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            filename = sample['filename']
            vid = sample['vid']
            event = sample['event']
            output = sample['label_content']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"avvp"})
            user_content_list.append({'type':'text','text':"Please determine the events that occur based on the visual and audio information in the video, as well as the start and end times of these events."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'task_name':'avvp'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'avvp sample nums: {tot}')
        self.tot += tot

    def add_ms3_samples(self):
        task_name = 'ms3'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            visual_path = self._get_media_path(task_name, None, sample['image_path'])
            audio_path = self._get_media_path(task_name, None, sample['audio_path'])
            vid = sample['vid']
            uid = sample['uid']
            a_obj = sample['a_obj']
            frame_id = sample['frame_idx']
            output = sample['label_content']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is an image and an audio:"})
            user_content_list.append({'type':'image','image':visual_path,'resized_height':224,'resized_width':224})
            user_content_list.append({'type':'audio','audio':audio_path,'task':'ms3','idx':frame_id})
            user_content_list.append({'type':'text','text':"Please identify the category of the object producing the sound in the video frame. Then, provide the bounding box and three pixel points of the object that is making the sound within the frame."})
            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'a_obj':a_obj,
                'idx':frame_id,
                'tot':5,
                'task_name':'ms3'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'ms3_box sample nums: {tot}')
        self.tot += tot

    def add_s4_samples(self):
        task_name = 's4'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            visual_path = self._get_media_path(task_name, None, sample['image_path'])
            audio_path = self._get_media_path(task_name, None, sample['audio_path'])
            vid = sample['vid']
            uid = sample['uid']
            a_obj = sample['a_obj']
            
            output = sample['label_content']
            user_content_list = []
            user_content_list.append({'type':'text','text':"This is an image and an audio:"})
            user_content_list.append({'type':'image','image':visual_path,'resized_height':224,'resized_width':224})
            user_content_list.append({'type':'audio','audio':audio_path,'task':'s4','idx':0})
            user_content_list.append({'type':'text','text':"Please identify the category of the object producing the sound in the video frame. Then, provide the bounding box and three pixel points of the object that is making the sound within the frame."})
            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'a_obj':a_obj,
                'idx':0,
                'task_name':'s4'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f's4 sample nums: {tot}')
        self.tot += tot

    def add_arig_samples(self,):
        task_name = 'arig'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            frame_path = self._get_media_path(task_name, None, sample['image_path'])
            audio_path = self._get_media_path(task_name, None, sample['audio_path'])
            a_obj = sample['a_obj']
            idx = int(frame_path.split('/')[-1][:-4])
            output = sample['label_content']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is an image and an audio:"})
            user_content_list.append({'type':'image','image':frame_path,'resized_height':224,'resized_width':224})
            user_content_list.append({'type':'audio','audio':audio_path, 'task':'arig'})
            user_content_list.append({'type':'text','text':"Please recognize the category of object that makes the sound and then output its bounding box."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'idx':idx,
                'tot':5,
                'task_name':'arig'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
    
        print(f'audio referred image grounding sample nums: {tot}')
        self.tot += tot

    def add_avcap_samples(self):
        task_name = 'avcap'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            video_id = sample['video_id']
            desc = sample['label_content']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':'avcap'})
            user_content_list.append({'type':'text','text':"Please describe this video and audio."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':desc}
            ]
            metadata = {
                'task_name':'avcap'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        self.tot += tot
        print(f'avcap sample nums: {tot}')

    def add_ref_avs_samples(self):
        task_name = 'ref_avs'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            image_path = self._get_media_path(task_name, None, sample['image_path'])
            audio_path = self._get_media_path(task_name, None, sample['audio_path'])
            vid = sample['vid']
            fid = sample['fid']
            uid = sample['uid']
            idx = sample['idx']
            description = sample['description']
            output = sample['output']

            instruction_text = f'''Please analyze the video frame and audio to find the object described as <des>{description.lower()}</des>. Answer with "true" or "false" whether the object exists. If it exists, respond with a sentence identifying the object's name, its bounding box, and three pixel points; if not, state that the object is absent.'''

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is an image and an audio:"})
            user_content_list.append({'type':'image','image':image_path,'resized_height':224,'resized_width':224})
            user_content_list.append({'type':'audio','audio':audio_path,'task':'ref_avs','idx':idx})
            user_content_list.append({'type':'text','text':instruction_text})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list},
                {'role':'assistant','content':output}
            ]
            metadata = {
                'vid':vid,
                'uid':uid,
                'fid':fid,
                'idx':idx,
                'task_name':'ref_avs'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        self.tot += tot
        print(f'ref_avs sample nums: {tot}')

    def read_label(self,label_path):
        with open(label_path,'r') as f:
            label = f.read()
        return label

    def extract_mm_info(self, conv):
        mm_infos = []
        for message in conv:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if 'image' in ele or 'audio' in ele or 'video' in ele:
                        mm_infos.append(ele)
        return mm_infos

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            conv = sample['conv']
            metadata = sample.get('metadata', {})
            task_name = metadata.get('task_name', '')

            audios_inputs, image_inputs, video_inputs = process_mm_info(conv, use_audio_in_video=False)

            need_concat_video = task_name in ['a2v']
            need_concat_audio = task_name in ['v2a']
        
            if need_concat_video and video_inputs is not None and len(video_inputs) == 2:
                video_inputs = torch.cat(video_inputs, dim=0)
        
            if need_concat_audio and audios_inputs is not None and len(audios_inputs) == 2:
                tensor1 = torch.from_numpy(audios_inputs[0])
                tensor2 = torch.from_numpy(audios_inputs[1])
                audios_inputs = torch.cat([tensor1, tensor2], dim=0)

            text = self.mm_processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
        
            AUDIO_PLACEHOLDER = "<|audio_bos|><|AUDIO|><|audio_eos|>"
            VIDEO_PLACEHOLDER = "<|vision_bos|><|VIDEO|><|vision_eos|>"

            if need_concat_audio:
                num_audio_tokens = text.count(AUDIO_PLACEHOLDER)
                if num_audio_tokens > 1:
                    text = text.replace(AUDIO_PLACEHOLDER, '', num_audio_tokens - 1)
        
            if need_concat_video:
                num_video_tokens = text.count(VIDEO_PLACEHOLDER)
                if num_video_tokens > 1:
                    text = text.replace(VIDEO_PLACEHOLDER, '', num_video_tokens - 1)
                    
            inputs = self.mm_processor(text=text, audio=audios_inputs, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True , use_audio_in_video = False)
            inputs["use_audio_in_video"] = False
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)

            labels = torch.full_like(input_ids, -100)
        
            assistant_start_token = "<|im_start|>assistant"
            assistant_end_token = "<|im_end|>"
        
            assistant_start_ids = self.tokenizer.encode(assistant_start_token, add_special_tokens=False)
            assistant_end_ids = self.tokenizer.encode(assistant_end_token, add_special_tokens=False)

            assistant_positions = []
            for i in range(len(input_ids) - len(assistant_start_ids) + 1):
                if torch.all(input_ids[i:i+len(assistant_start_ids)] == torch.tensor(assistant_start_ids)):
                    assistant_positions.append(i)
        
            if len(assistant_positions) > 0:
                start_pos = assistant_positions[0] + len(assistant_start_ids)
            
            end_pos = None
            for i in range(start_pos, len(input_ids) - len(assistant_end_ids) + 1):
                if torch.all(input_ids[i:i+len(assistant_end_ids)] == torch.tensor(assistant_end_ids)):
                    end_pos = i
                    break
                    
            if end_pos is not None:
                labels[start_pos:end_pos+len(assistant_end_ids)] = input_ids[start_pos:end_pos+len(assistant_end_ids)]
        
            data = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "use_audio_in_video": False,
            }

            if image_inputs is not None:
                data["pixel_values"] = inputs["pixel_values"]
                data["image_grid_thw"] = inputs["image_grid_thw"]
            if audios_inputs is not None:
                data["input_features"] = inputs["input_features"]
                data["feature_attention_mask"] = inputs["feature_attention_mask"]
            if video_inputs is not None:
                data["pixel_values_videos"] = inputs["pixel_values_videos"]
                data["video_grid_thw"] = inputs["video_grid_thw"]
                data["video_second_per_grid"] = inputs["video_second_per_grid"]
            return data
            
        except Exception as e:
            print(f"ERROR processing sample {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

@dataclass
class DataCollatorForOmniDataset:
    mm_processor: Qwen2_5OmniProcessor
    mode: str = 'train'
    
    def __init__(self, mm_processor, **kwargs):
        self.mm_processor = mm_processor
        for key, value in kwargs.items():
            setattr(self, key, value)

    def gather_list(self, batch, key):
        return [item[key] for item in batch if key in item and item[key] is not None]
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if not instances:
            print("Warning: No instances provided to DataCollator")
            return {}
            
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        batch_metadata = []
        use_audio_in_video = any(instance.get('use_audio_in_video', False) for instance in instances)
        has_images = any('pixel_values' in instance for instance in instances)
        has_videos = any('pixel_values_videos' in instance for instance in instances)
        has_audio = any('input_features' in instance for instance in instances)

        if has_images:
            batch_pixel_values = []
            batch_image_grid_thw = []
        
        if has_videos:
            batch_pixel_values_videos = []
            batch_video_grid_thw = []
            batch_video_second_per_grid = []
        
        if has_audio:
            batch_input_features = []
            batch_feature_attention_mask = []

        for i, instance in enumerate(instances):
            
            batch_input_ids.append(instance['input_ids'])
            batch_labels.append(instance['labels'])
            batch_attention_mask.append(instance['attention_mask'])
            batch_metadata.append(instance.get('metadata', {}))

            if has_images and 'pixel_values' in instance:
                batch_pixel_values.append(instance['pixel_values'])
                if 'image_grid_thw' in instance:
                    batch_image_grid_thw.append(instance['image_grid_thw'])

            if has_videos and 'pixel_values_videos' in instance:
                batch_pixel_values_videos.append(instance['pixel_values_videos'])
                if 'video_grid_thw' in instance:
                    batch_video_grid_thw.append(instance['video_grid_thw'])
                if 'video_second_per_grid' in instance:
                    batch_video_second_per_grid.append(instance['video_second_per_grid'])

            if has_audio and 'input_features' in instance:
                batch_input_features.append(instance['input_features'])
                if 'feature_attention_mask' in instance:
                    batch_feature_attention_mask.append(instance['feature_attention_mask'])

        try:
            input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.mm_processor.tokenizer.pad_token_id)
            labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
            attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)

            batch = {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
                'use_audio_in_video': use_audio_in_video
            }
            
            if has_images:
                batch_pixel_values = self.gather_list(instances, 'pixel_values')
                if batch_pixel_values:
                    batch['pixel_values'] = torch.cat(batch_pixel_values, dim=0)
                
                batch_image_grid_thw = self.gather_list(instances, 'image_grid_thw')
                if batch_image_grid_thw:
                    batch['image_grid_thw'] = torch.cat(batch_image_grid_thw, dim=0)

            if has_videos:
                batch_pixel_values_videos = self.gather_list(instances, 'pixel_values_videos')
                if batch_pixel_values_videos:
                    batch['pixel_values_videos'] = torch.cat(batch_pixel_values_videos, dim=0)
                
                batch_video_grid_thw = self.gather_list(instances, 'video_grid_thw')
                if batch_video_grid_thw:
                    batch['video_grid_thw'] = torch.cat(batch_video_grid_thw, dim=0)
                
                batch_video_second_per_grid = self.gather_list(instances, 'video_second_per_grid')
                if batch_video_second_per_grid:
                    batch['video_second_per_grid'] = torch.tensor(batch_video_second_per_grid)

            if has_audio:
                batch_input_features = self.gather_list(instances, 'input_features')
                if batch_input_features:
                    batch['input_features'] = torch.cat(batch_input_features, dim=0)
                
                batch_feature_attention_mask = self.gather_list(instances, 'feature_attention_mask')
                if batch_feature_attention_mask:
                    batch['feature_attention_mask'] = torch.cat(batch_feature_attention_mask, dim=0)
            
            return batch
            
        except Exception as e:
            print(f"Error in DataCollator: {str(e)}")
            raise e

class OmniTestDataset(Dataset):
    def __init__(
        self,
        data_args,
        tokenizer: Qwen2TokenizerFast,
        audio_processor: WhisperFeatureExtractor,
        vision_processor: Qwen2VLImageProcessor,
        mm_processor: Qwen2_5OmniProcessor,
        mode='test',
        **kwargs
    ):
        super().__init__()

        self.samples = []
        self.mode = mode
        self.mm_processor = mm_processor
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.vision_processor = vision_processor
        self.tot = 0
        self.data_root = getattr(data_args, 'data_root', '')

        self.task_loaders = {
            'a2v': self.add_a2v_samples,
            'v2a': self.add_v2a_samples,
            'avqa': self.add_avqa_samples,
            'avqa_thu': self.add_avqa_thu_samples,
            'ave': self.add_ave_samples,
            'avvp': self.add_avvp_samples,
            'ms3': self.add_ms3_samples,
            's4': self.add_s4_samples,
            'ref_avs': self.add_ref_avs_samples,
            'arig': self.add_arig_samples,
            'meld': self.add_meld_samples,
            'mafw': self.add_mafw_samples,
            'dfew': self.add_dfew_samples,
            'cremad': self.add_cremad_samples,
            'ks': self.add_ks_samples,
            'ucf': self.add_ucf_samples,
        }

        for task_name, loader_func in self.task_loaders.items():
            arg_name = f"{task_name}_task"
            is_enabled = getattr(data_args, arg_name, kwargs.get(arg_name, False))

            if is_enabled:
                print(f"Loading task: {task_name}...")
                if hasattr(loader_func, '__call__'):
                    loader_func()
                else:
                    print(f"Warning: Method for {task_name} is mapped but not implemented.")

        print(f'Total {self.mode} sample nums: {self.tot}')

    def _get_media_path(self, task_name, media_type, filename):
        if not filename: return None
        if media_type:
            return os.path.join(self.data_root, 'AVUIE_2', task_name, media_type, filename)
        else:
            return os.path.join(self.data_root, 'AVUIE_2', task_name, filename)

    def read_label(self, label_path):
        with open(label_path, 'r') as f:
            label = f.read()
        return label

    def _get_task_path(self, task_name):
        if task_name not in TASK_PATH_REGISTRY:
            print(f"Warning: Task {task_name} not explicitly in registry. Using default pattern.")
            relative_path = f"AVUIE_2/{task_name}/{self.mode}.json"
        else:
            relative_path = TASK_PATH_REGISTRY[task_name].get(self.mode)
        
        if not relative_path:
            print(f"Warning: No path config found for task {task_name} in mode {self.mode}, skipping.")
            return None
            
        full_path = os.path.join(self.data_root, relative_path)
        return full_path

    def add_a2v_samples(self):
        task_name = 'a2v'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video1_path = self._get_media_path(task_name, 'video', sample['video1_path'])
            video2_path = self._get_media_path(task_name, 'video', sample['video2_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])

            correct_video = sample['correct_video']
            if str(correct_video) == "1":
                output = "one"
            else:
                output = "two"
            output = correct_video
            label = f"<answer>{output}</answer>."

            instruction = f'Please listen to the audio and indicate which of the two videos matches what you hear.'

            user_content_list = []
            user_content_list.append({'type':'text','text':"These are two videos and an audio:"})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"a2v"})
            user_content_list.append({'type':'video','video':video1_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'video','video':video2_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'text','text':instruction})

            conv = [
                    {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                    {'role':'user','content':user_content_list}
                ]
            metadata = {
                'task_name':'a2v',
                'output': label
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'{self.mode}, a2v sample nums: {tot}')
        self.tot += tot

    def add_v2a_samples(self):
        task_name = 'v2a'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio1_path = self._get_media_path(task_name, 'audio', sample['audio1_path'])
            audio2_path = self._get_media_path(task_name, 'audio', sample['audio2_path'])

            correct_audio = sample['correct_audio']
            if str(correct_audio) == "1":
                output = "one"
            else:
                output = "two"
            output = correct_audio
            label = f"<answer>{output}</answer>."

            instruction = f'Please view the video and indicate which of the two audio clips matches what you see.'

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and two audio clips:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio1_path,'task':"v2a"})
            user_content_list.append({'type':'audio','audio':audio2_path,'task':"v2a"})
            user_content_list.append({'type':'text','text':instruction})

            conv = [
                    {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                    {'role':'user','content':user_content_list}
                ]
            metadata = {
                'task_name':'v2a',
                'output':label
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'{self.mode}, v2a sample nums: {tot}')
        self.tot += tot

    def add_ks_samples(self):
        task_name = 'ks'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            action = sample['action']
            output = f"<answer>{action}</answer>"

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"ks"})
            user_content_list.append({'type':'text','text':"Please identify the action taking place in the video."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'task_name':'ks',
                'output':output
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'ks sample nums: {tot}')
        self.tot += tot

    def add_ucf_samples(self):
        task_name = 'ucf'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            action = sample['label']
            action = action.lower()
            output = f"<answer>{action}</answer>"

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"ucf"})
            user_content_list.append({'type':'text','text':"Please identify the action taking place in the video."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'task_name':'ucf',
                'audio_path':audio_path,
                'video_path':video_path,
                'output':output
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'ucf sample nums: {tot}')
        self.tot += tot

    def add_meld_samples(self):
        task_name = 'meld'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            emotion = sample['Emotion']
            output = f"<answer>{emotion}</answer>."

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':4})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"meld"})
            user_content_list.append({'type':'text','text':"Please judge the person's emotion at the moment."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'task_name':'meld',
                'output': output
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'meld sample nums: {tot}')
        self.tot += tot

    def add_cremad_samples(self):
        task_name = 'cremad'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            emotion = sample['emotion']
            output = f"<answer>{emotion}</answer>."

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':4})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"cremad"})
            user_content_list.append({'type':'text','text':"Please judge the person's emotion at the moment."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'task_name':'cremad',
                'output':output
                }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'cremad sample nums: {tot}')
        self.tot += tot

    def add_mafw_samples(self):
        task_name = 'mafw'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            emotion = sample['emotion']
            output = f"<answer>{emotion}</answer>."

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':4})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"mafw"})
            user_content_list.append({'type':'text','text':"Please judge the person's emotion at the moment."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'task_name':'mafw',
                'output':output
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'mafw sample nums: {tot}')
        self.tot += tot

    def add_dfew_samples(self):
        task_name = 'dfew'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            emotion = sample['emotion']
            output = f"<answer>{emotion}</answer>."

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':4})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"dfew"})
            user_content_list.append({'type':'text','text':"Please judge the person's emotion at the moment."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'task_name':'dfew',
                'output':output
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'dfew sample nums: {tot}')
        self.tot += tot

    def add_avqa_samples(self):
        task_name = 'avqa'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            video_id = sample['video_id']
            question_id = sample['question_id']
            _type = sample['type']
            question = sample['question']
            answer = sample['answer']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"avqa"})
            user_content_list.append({'type':'text','text':f"Please answer this question: {question}"})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'vid':video_id,
                'qid':question_id,
                'type':_type,
                'task_name':'avqa',
                'output':answer
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'avqa sample nums: {tot}')
        self.tot += tot

    def add_avqa_thu_samples(self):
        task_name = 'avqa_thu'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            _id = sample['id']
            name = sample['video_name']
            video_id = sample['video_id']
            relation = sample['question_relation']
            _type = sample['question_type']
            multi_choice = sample['multi_choice']
            question = sample['question_text']
            output = sample['answer']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"avqa_thu"})
            user_content_list.append({'type':'text','text':f"Please answer the question based on the options: {question}, options:{multi_choice}"})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'vid':video_id,
                'instruction':f"Please answer the question based on the options: {question}, options:{multi_choice}",
                'name':name,
                'task_name':'avqa_thu',
                'type':_type,
                'relation':relation,
                'output':output
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'avqa_thu sample nums: {tot}')
        self.tot += tot

    def add_ave_samples(self):
        task_name = 'ave'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            event = sample['event']
            vid = sample['vid']
            start_time = sample['start_time']
            end_time = sample['end_time']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"ave"})
            user_content_list.append({'type':'text','text':"Please describe the events and time range that occurred in the video."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'task_name':'ave',
                'output': f'event:{event} start_time:{start_time} end_time:{end_time}'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'ave sample nums: {tot}')
        self.tot += tot

    def add_avvp_samples(self):
        task_name = 'avvp'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            video_path = self._get_media_path(task_name, 'video', sample['video_path'])
            audio_path = self._get_media_path(task_name, 'audio', sample['audio_path'])
            filename = sample['filename']
            vid = sample['vid']
            event = sample['event']

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is a video and an audio:"})
            user_content_list.append({'type':'video','video':video_path,'resized_height':224,'resized_width':224,'nframes':10})
            user_content_list.append({'type':'audio','audio':audio_path,'task':"avvp"})
            user_content_list.append({'type':'text','text':"Please determine the events that occur based on the visual and audio information in the video, as well as the start and end times of these events."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'task_name':'avvp',
                'output': event
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'avvp sample nums: {tot}')
        self.tot += tot

    def add_ms3_samples(self):
        task_name = 'ms3'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            visual_path = self._get_media_path(task_name, None, sample['visual_path'])
            audio_path = self._get_media_path(task_name, None, sample['audio_path'])
            vid = sample['vid']
            uid = sample['uid']
            a_obj = sample['a_obj']
            idx = sample['frame_idx']
            user_content_list = []
            user_content_list.append({'type':'text','text':"This is an image and an audio:"})
            user_content_list.append({'type':'image','image':visual_path,'resized_height':224,'resized_width':224})
            user_content_list.append({'type':'audio','audio':audio_path,'task':'ms3','idx':idx})
            user_content_list.append({'type':'text','text':"Please identify the category of the object producing the sound in the video frame. Then, provide the bounding box and three pixel points of the object that is making the sound within the frame."})
            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'a_obj':a_obj,
                'image_path':visual_path,
                'idx':idx,
                'tot':5,
                'task_name':'ms3',
                'output':f'The object making the sound in the video is {a_obj}.'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f'ms3 sample nums: {tot}')
        self.tot += tot

    def add_s4_samples(self):
        task_name = 's4'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            visual_path = self._get_media_path(task_name, None, sample['visual_path'])
            audio_path = self._get_media_path(task_name, None, sample['audio_path'])
            vid = sample['vid']
            uid = sample['uid']
            a_obj = sample['a_obj']
            idx = sample['frame_idx']
            user_content_list = []
            user_content_list.append({'type':'text','text':"This is an image and an audio:"})
            user_content_list.append({'type':'image','image':visual_path,'resized_height':224,'resized_width':224})
            user_content_list.append({'type':'audio','audio':audio_path,'task':'s4','idx':idx})
            user_content_list.append({'type':'text','text':"Please identify the category of the object producing the sound in the video frame. Then, provide the bounding box and three pixel points of the object that is making the sound within the frame."})
            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'a_obj':a_obj,
                'idx':0,
                'image_path':visual_path,
                'task_name':'s4',
                'output':f'The object making the sound in the video is {a_obj}.'
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
        print(f's4 sample nums: {tot}')
        self.tot += tot

    def add_arig_samples(self,):
        task_name = 'arig'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            frame_path = self._get_media_path(task_name, None, sample['visual_path'])
            audio_path = self._get_media_path(task_name, None, sample['audio_path'])
            top_left = sample['top_left']
            bottom_right = sample['bottom_right']
            a_obj = sample['a_obj']
            x1, y1 = top_left
            x2, y2 = bottom_right
            idx = sample['frame_idx']

            output = f'The sounding object is {a_obj}. Its coordinates are <obj>({x1},{y1})({x2},{y2})</obj>.'

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is an image and an audio:"})
            user_content_list.append({'type':'image','image':frame_path,'resized_height':224,'resized_width':224})
            user_content_list.append({'type':'audio','audio':audio_path, 'task':'arig'})
            user_content_list.append({'type':'text','text':"Please recognize the category of object that makes the sound and then output its bounding box."})

            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'frame_path':frame_path,
                'idx':idx,
                'tot':5,
                'task_name':'arig',
                'output': output
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1
    
        print(f'audio referred image grounding sample nums: {tot}')
        self.tot += tot

    def add_ref_avs_samples(self):
        task_name = 'ref_avs'
        json_path = self._get_task_path(task_name)
        if json_path is None: return
        
        tot = 0
        if not os.path.exists(json_path):
            print(f"Skipping {task_name}: File not found {json_path}")
            return
            
        with open(json_path, 'r') as f:
            samples = json.load(f)
        for sample in samples:
            image_path = self._get_media_path(task_name, None, sample['visual_path'])
            audio_path = self._get_media_path(task_name, None, sample['audio_path'])
            exp = sample['object']
            split = sample['split']
            vid = sample['vid']
            uid = sample['uid']
            fid = sample['fid']
            idx = sample['frame_idx']
            instruction = f'''Please analyze the video frame and audio to find the object described as <des>{exp.lower()}</des>. Answer with "true" or "false" whether the object exists. If it exists, respond with a sentence identifying the object's name, its bounding box, and three pixel points; if not, state that the object is absent.'''
            output = f'{exp}'

            user_content_list = []
            user_content_list.append({'type':'text','text':"This is an image and an audio:"})
            user_content_list.append({'type':'image','image':image_path,'resized_height':224,'resized_width':224})
            user_content_list.append({'type':'audio','audio':audio_path,'task':'ref_avs','idx':idx})
            user_content_list.append({'type':'text','text':instruction})
            conv = [
                {'role':'system','content':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'},
                {'role':'user','content':user_content_list}
            ]
            metadata = {
                'vid':vid,
                'idx':idx,
                'uid':uid,
                'fid':fid,
                'split': split,
                'image_path':image_path,
                'task_name':'ref_avs',
                'output':output
            }
            self.samples.append(
                {
                    'conv':conv,
                    'metadata': metadata
                }
            )
            tot += 1

        self.tot += tot
        print(f'ref_avs sample nums: {tot}')

    def extract_mm_info(self, conv):
        mm_infos = []
        for message in conv:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if 'image' in ele or 'audio' in ele or 'video' in ele:
                        mm_infos.append(ele)
        
        return mm_infos

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            conv = sample['conv']
            metadata = sample.get('metadata', {})
            task_name = metadata.get('task_name', '')

            audios_inputs, image_inputs, video_inputs = process_mm_info(conv, use_audio_in_video=False)

            need_concat_video = task_name in ['a2v']
            need_concat_audio = task_name in ['v2a']

            if need_concat_video and video_inputs is not None and len(video_inputs) == 2:
                video_inputs = torch.cat(video_inputs, dim=0)

            if need_concat_audio and audios_inputs is not None and len(audios_inputs) == 2:
                tensor1 = torch.from_numpy(audios_inputs[0])
                tensor2 = torch.from_numpy(audios_inputs[1])
                audios_inputs = torch.cat([tensor1, tensor2], dim=0)

            text = self.mm_processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    
            AUDIO_PLACEHOLDER = "<|audio_bos|><|AUDIO|><|audio_eos|>"
            VIDEO_PLACEHOLDER = "<|vision_bos|><|VIDEO|><|vision_eos|>"

            if need_concat_audio:
                num_audio_tokens = text.count(AUDIO_PLACEHOLDER)
                if num_audio_tokens > 1:
                    text = text.replace(AUDIO_PLACEHOLDER, '', num_audio_tokens - 1)

            if need_concat_video:
                num_video_tokens = text.count(VIDEO_PLACEHOLDER)
                if num_video_tokens > 1:
                    text = text.replace(VIDEO_PLACEHOLDER, '', num_video_tokens - 1)

            inputs = self.mm_processor(
                text=text, 
                audio=audios_inputs, 
                images=image_inputs, 
                videos=video_inputs, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=False
            )

            inputs["use_audio_in_video"] = False
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)

            data = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_audio_in_video": False,
                "metadata": metadata,
            }

            if image_inputs is not None:
                data["pixel_values"] = inputs["pixel_values"]
                data["image_grid_thw"] = inputs["image_grid_thw"]
            if audios_inputs is not None:
                data["input_features"] = inputs["input_features"]
                data["feature_attention_mask"] = inputs["feature_attention_mask"]
            if video_inputs is not None:
                data["pixel_values_videos"] = inputs["pixel_values_videos"]
                data["video_grid_thw"] = inputs["video_grid_thw"]
                data["video_second_per_grid"] = inputs["video_second_per_grid"]

            return data
        
        except Exception as e:
            print(f"ERROR processing test sample {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e


@dataclass
class DataCollatorForOmniTestDataset:
    mm_processor: Qwen2_5OmniProcessor
    mode: str = 'test'
    
    def __init__(self, mm_processor, **kwargs):
        self.mm_processor = mm_processor
        for key, value in kwargs.items():
            setattr(self, key, value)

    def gather_list(self, batch, key):
        return [item[key] for item in batch if key in item and item[key] is not None]
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if not instances:
            print("Warning: No instances provided to DataCollator")
            return {}
            
        batch_input_ids = []
        batch_attention_mask = []
        batch_metadata = []
        use_audio_in_video = any(instance.get('use_audio_in_video', False) for instance in instances)
        has_images = any('pixel_values' in instance for instance in instances)
        has_videos = any('pixel_values_videos' in instance for instance in instances)
        has_audio = any('input_features' in instance for instance in instances)

        if has_images:
            batch_pixel_values = []
            batch_image_grid_thw = []
        
        if has_videos:
            batch_pixel_values_videos = []
            batch_video_grid_thw = []
            batch_video_second_per_grid = []
        
        if has_audio:
            batch_input_features = []
            batch_feature_attention_mask = []

        for i, instance in enumerate(instances):
            batch_input_ids.append(instance['input_ids'])
            batch_attention_mask.append(instance['attention_mask'])
            batch_metadata.append(instance.get('metadata', {}))

            if has_images and 'pixel_values' in instance:
                batch_pixel_values.append(instance['pixel_values'])
                if 'image_grid_thw' in instance:
                    batch_image_grid_thw.append(instance['image_grid_thw'])

            if has_videos and 'pixel_values_videos' in instance:
                batch_pixel_values_videos.append(instance['pixel_values_videos'])
                if 'video_grid_thw' in instance:
                    batch_video_grid_thw.append(instance['video_grid_thw'])
                if 'video_second_per_grid' in instance:
                    batch_video_second_per_grid.append(instance['video_second_per_grid'])

            if has_audio and 'input_features' in instance:
                batch_input_features.append(instance['input_features'])
                if 'feature_attention_mask' in instance:
                    batch_feature_attention_mask.append(instance['feature_attention_mask'])

        try:
            input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.mm_processor.tokenizer.pad_token_id)
            attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)

            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'use_audio_in_video': use_audio_in_video,
                'metadata': batch_metadata,
            }
            
            if has_images:
                batch_pixel_values = self.gather_list(instances, 'pixel_values')
                if batch_pixel_values:
                    batch['pixel_values'] = torch.cat(batch_pixel_values, dim=0)
                
                batch_image_grid_thw = self.gather_list(instances, 'image_grid_thw')
                if batch_image_grid_thw:
                    batch['image_grid_thw'] = torch.cat(batch_image_grid_thw, dim=0)

            if has_videos:
                batch_pixel_values_videos = self.gather_list(instances, 'pixel_values_videos')
                if batch_pixel_values_videos:
                    batch['pixel_values_videos'] = torch.cat(batch_pixel_values_videos, dim=0)
                
                batch_video_grid_thw = self.gather_list(instances, 'video_grid_thw')
                if batch_video_grid_thw:
                    batch['video_grid_thw'] = torch.cat(batch_video_grid_thw, dim=0)
                
                batch_video_second_per_grid = self.gather_list(instances, 'video_second_per_grid')
                if batch_video_second_per_grid:
                    batch['video_second_per_grid'] = torch.tensor(batch_video_second_per_grid)

            if has_audio:
                batch_input_features = self.gather_list(instances, 'input_features')
                if batch_input_features:
                    batch['input_features'] = torch.cat(batch_input_features, dim=0)
                
                batch_feature_attention_mask = self.gather_list(instances, 'feature_attention_mask')
                if batch_feature_attention_mask:
                    batch['feature_attention_mask'] = torch.cat(batch_feature_attention_mask, dim=0)
            return batch
            
        except Exception as e:
            print(f"Error in Test DataCollator: {str(e)}")
            raise e