import gradio as gr
from pathlib import Path
from modules import script_callbacks,scripts, devices, sd_hijack
from modules.sd_models import *
from modules.timer import Timer
from modules.modelloader import load_file_from_url
from tqdm import tqdm
from copy import deepcopy
import gc
import safetensors.torch
import os
import torch

extension_name = 'Reload model as inpaint'

EXTENSION_ROOT = scripts.basedir()
ext2abs = lambda x: os.path.join(EXTENSION_ROOT,x)

ARCHITECTURES = {
    'V1-inpaint': {
        #Which element, shape index, and dimension to ID the architecture by
        'id' : ['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight',1,768],
        'type_id' : ['model.diffusion_model.input_blocks.0.0.weight',1,4],
        'filename' : 'models/V1_5inpaintdifference.safetensors',
        'url' : 'https://huggingface.co/groinge/inpaintdifferencemerge/resolve/main/V1_5inpaintdifference.safetensors'
    }
}

class script(scripts.Script):
        def __init__(self) -> None:
            super().__init__()

        def title(self):
            return extension_name

        def show(self,is_img2img):
            if is_img2img:
                return scripts.AlwaysVisible
        
        def ui(self,is_img2img):
            with gr.Accordion(label=extension_name,open=False):
                with gr.Row():
                    button = gr.Button(value = 'Convert',scale=3,variant='primary')
                    use_cuda = gr.Checkbox(value = True, label="Use cuda for merging")
                with gr.Row():
                    display = gr.Textbox(label="",value="",interactive=False,max_lines=1)
                
                button.click(fn=event,inputs=use_cuda,outputs=display)

def event(use_cuda):
    message = convert(use_cuda)
    print(extension_name+':  '+message)
    return message

def convert(use_cuda):
    device = 'cuda' if use_cuda else 'cpu'

    if not model_data.sd_model: return "No checkpoint loaded."
    sd_0, checkpoint_info, arch, message = grab_active_model_sd(device)
    if not sd_0: sd_hijack.model_hijack.hijack(model_data.sd_model); return message
    fake_cp_info = deepcopy(checkpoint_info)

    sd_1 = load_inpaint_statedict(arch, device)
    sd_1 = sd_1.get('state_dict') or sd_1
    
    for k1 in tqdm(list(sd_1.keys()),desc='Merging models'):
        k0 = 'model.diffusion_model.' + k1

        a = list(sd_0[k0].shape)
        b = list(sd_1[k1].shape)

        if a != b and a[0:1] + a[2:] == b[0:1] + b[2:]:
            sd_1[k1][:, 0:4, :, :] = sd_1[k1][:, 0:4, :, :] + sd_0[k0]
            sd_0[k0] = sd_1[k1]
        else:
            sd_0[k0] = sd_1[k1] + sd_0[k0]

    fake_cp_info.name = f"temp-{checkpoint_info.name}-inpainting"

    del sd_1
    load_model(checkpoint_info=fake_cp_info, already_loaded_state_dict=sd_0)
    del sd_0

    gc.collect()
    devices.torch_gc()
    return 'Successfully converted model.'

def load_inpaint_statedict(arch,device):
    if os.path.isfile(ext2abs(arch['filename'])):
        return safetensors.torch.load_file(ext2abs(arch['filename']), device=device)
    else:
        gr.Info('Downloading model (1.7 GB)...')
        file_path = load_file_from_url(arch['url'],model_dir=ext2abs('models'))
        return safetensors.torch.load_file(file_path, device=device)
        

def grab_active_model_sd(device):
    checkpoint_info = model_data.sd_model.sd_checkpoint_info

    model_data.sd_model.to(torch.device(device))
    sd_hijack.model_hijack.undo_hijack(model_data.sd_model)

    sd_0 = deepcopy(model_data.sd_model.state_dict())

    for info in ARCHITECTURES.values():
        elname, idindex, iddim = info['id']
        elem = sd_0.get(elname)
        if list(elem) and list(elem.shape)[idindex] == iddim:
            elname, idindex, iddim = info['type_id']
            elem = sd_0.get(elname)
            if list(elem) and list(elem.shape)[idindex] == iddim:
                arch = info
                break
            else:
                del sd_0; return None,None,None,'Loaded checkpoint is already an inpainting or pix-to-pix model'
    else:
        del sd_0; return None,None,None,'Unsupported model architecture, this extension only supports V1.5 checkpoints'
    
    model_data.sd_model = None
    gc.collect()
    devices.torch_gc()
    return sd_0,checkpoint_info,arch,None

    