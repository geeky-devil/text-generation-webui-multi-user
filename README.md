# Text generation web UI mod for multi-user mode

A simple mod for running the ui for mutliple users.

## Screenshots

| ![image](https://github.com/geeky-devil/text-generation-webui-multi-user/assets/73377915/1ffbc7ac-c9c5-4abb-8e05-5d4f63c9fa81)|![image](https://github.com/geeky-devil/text-generation-webui-multi-user/assets/73377915/46176c96-f411-4d81-8261-cc7285d76400)|
|:---:|:---:|
### Log
![image](https://github.com/geeky-devil/text-generation-webui-multi-user/assets/73377915/eaba1b7c-792b-4794-b262-c033b93270f7)
## Features

* Single Interface : chat.
* Save chats in multi-user mode, chats saved under
  ```logs/chat/character/username/```
* Logout button to terminate the ui
* logger info indicating ui termination

## How to install the mod
1) Clone or [download](https://github.com/oobabooga/text-generation-webui/archive/refs/heads/main.zip) the repository.
2) create the backup of these files : ui_chat.py , server.py , shared.py, chat.py ( location given in next point).
3) if this is your first time just follow the orignal method given and then replace the files as follows
   * ui_chat,shared,chat in modules
   * server in main dir
   * main.css in css
4) Once replaced run the server.py with custom arguments like
   ```
   python server.py --character Carl --settings settings-template.yaml --multi-user --username goof --listen-port 4000
   ```
6) Once the ui loads you can then terminate the session using logout button.

# NOTE
  * The following modification are only made to work with ```--multi-user``` mode, to run the orignal ui, replace the original files.
  * The files used here are bit older than the latest repo ( about 1 month), updated files have new features so this mod wouldnt work with the latest files.
  * Have added the old version in repo.




#### Basic settings

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `-h`, `--help`                             | show this help message and exit |
| `--multi-user`                             | Multi-user mode. Chat histories are not saved or automatically loaded. WARNING: this is likely not safe for sharing publicly. |
| `--character CHARACTER`                    | The name of the character to load in chat mode by default. |
| `--model MODEL`                            | Name of the model to load by default. |
| `--lora LORA [LORA ...]`                   | The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces. |
| `--model-dir MODEL_DIR`                    | Path to directory with all the models. |
| `--lora-dir LORA_DIR`                      | Path to directory with all the loras. |
| `--model-menu`                             | Show a model menu in the terminal when the web UI is first launched. |
| `--settings SETTINGS_FILE`                 | Load the default interface settings from this yaml file. See `settings-template.yaml` for an example. If you create a file called `settings.yaml`, this file will be loaded by default without the need to use the `--settings` flag. |
| `--extensions EXTENSIONS [EXTENSIONS ...]` | The list of extensions to load. If you want to load more than one extension, write the names separated by spaces. |
| `--verbose`                                | Print the prompts to the terminal. |
| `--chat-buttons`                           | Show buttons on the chat tab instead of a hover menu. |

#### Model loader

| Flag                                       | Description |
|--------------------------------------------|-------------|
| `--loader LOADER`                          | Choose the model loader manually, otherwise, it will get autodetected. Valid options: Transformers, llama.cpp, llamacpp_HF, ExLlamav2_HF, ExLlamav2, AutoGPTQ, AutoAWQ, GPTQ-for-LLaMa, QuIP#. |

#### Accelerate/transformers

| Flag                                        | Description |
|---------------------------------------------|-------------|
| `--cpu`                                     | Use the CPU to generate text. Warning: Training on CPU is extremely slow. |
| `--auto-devices`                            | Automatically split the model across the available GPU(s) and CPU. |
|  `--gpu-memory GPU_MEMORY [GPU_MEMORY ...]` | Maximum GPU memory in GiB to be allocated per GPU. Example: --gpu-memory 10 for a single GPU, --gpu-memory 10 5 for two GPUs. You can also set values in MiB like --gpu-memory 3500MiB. |
| `--cpu-memory CPU_MEMORY`                   | Maximum CPU memory in GiB to allocate for offloaded weights. Same as above. |
| `--disk`                                    | If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk. |
| `--disk-cache-dir DISK_CACHE_DIR`           | Directory to save the disk cache to. Defaults to "cache". |
| `--load-in-8bit`                            | Load the model with 8-bit precision (using bitsandbytes). |
| `--bf16`                                    | Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU. |
| `--no-cache`                                | Set `use_cache` to `False` while generating text. This reduces VRAM usage slightly, but it comes at a performance cost. |
| `--trust-remote-code`                       | Set `trust_remote_code=True` while loading the model. Necessary for some models. |
| `--no_use_fast`                             | Set use_fast=False while loading the tokenizer (it's True by default). Use this if you have any problems related to use_fast. |
| `--use_flash_attention_2`                   | Set use_flash_attention_2=True while loading the model. |

#### bitsandbytes 4-bit

⚠️  Requires minimum compute of 7.0 on Windows at the moment.

| Flag                                        | Description |
|---------------------------------------------|-------------|
| `--load-in-4bit`                            | Load the model with 4-bit precision (using bitsandbytes). |
| `--use_double_quant`                        | use_double_quant for 4-bit. |
| `--compute_dtype COMPUTE_DTYPE`             | compute dtype for 4-bit. Valid options: bfloat16, float16, float32. |
| `--quant_type QUANT_TYPE`                   | quant_type for 4-bit. Valid options: nf4, fp4. |

#### llama.cpp

| Flag        | Description |
|-------------|-------------|
| `--tensorcores`  | Use llama-cpp-python compiled with tensor cores support. This increases performance on RTX cards. NVIDIA only. |
| `--n_ctx N_CTX` | Size of the prompt context. |
| `--threads` | Number of threads to use. |
| `--threads-batch THREADS_BATCH` | Number of threads to use for batches/prompt processing. |
| `--no_mul_mat_q` | Disable the mulmat kernels. |
| `--n_batch` | Maximum number of prompt tokens to batch together when calling llama_eval. |
| `--no-mmap`   | Prevent mmap from being used. |
| `--mlock`     | Force the system to keep the model in RAM. |
| `--n-gpu-layers N_GPU_LAYERS` | Number of layers to offload to the GPU. |
| `--tensor_split TENSOR_SPLIT`       | Split the model across multiple GPUs. Comma-separated list of proportions. Example: 18,17. |
| `--numa`      | Activate NUMA task allocation for llama.cpp. |
| `--logits_all`| Needs to be set for perplexity evaluation to work. Otherwise, ignore it, as it makes prompt processing slower. |
| `--no_offload_kqv` | Do not offload the K, Q, V to the GPU. This saves VRAM but reduces the performance. |
| `--cache-capacity CACHE_CAPACITY`   | Maximum cache capacity (llama-cpp-python). Examples: 2000MiB, 2GiB. When provided without units, bytes will be assumed. |
| `--row_split`                               | Split the model by rows across GPUs. This may improve multi-gpu performance. |
| `--streaming-llm`                           | Activate StreamingLLM to avoid re-evaluating the entire prompt when old messages are removed. |
| `--attention-sink-size ATTENTION_SINK_SIZE` | StreamingLLM: number of sink tokens. Only used if the trimmed prompt doesn't share a prefix with the old prompt. |

#### ExLlamav2

| Flag             | Description |
|------------------|-------------|
|`--gpu-split`     | Comma-separated list of VRAM (in GB) to use per GPU device for model layers. Example: 20,7,7. |
|`--max_seq_len MAX_SEQ_LEN`           | Maximum sequence length. |
|`--cfg-cache`                         | ExLlamav2_HF: Create an additional cache for CFG negative prompts. Necessary to use CFG with that loader. |
|`--no_flash_attn`                     | Force flash-attention to not be used. |
|`--cache_8bit`                        | Use 8-bit cache to save VRAM. |
|`--cache_4bit`                        | Use Q4 cache to save VRAM. |
|`--num_experts_per_token NUM_EXPERTS_PER_TOKEN` |  Number of experts to use for generation. Applies to MoE models like Mixtral. |

#### AutoGPTQ

| Flag             | Description |
|------------------|-------------|
| `--triton`                     | Use triton. |
| `--no_inject_fused_attention`  | Disable the use of fused attention, which will use less VRAM at the cost of slower inference. |
| `--no_inject_fused_mlp`        | Triton mode only: disable the use of fused MLP, which will use less VRAM at the cost of slower inference. |
| `--no_use_cuda_fp16`           | This can make models faster on some systems. |
| `--desc_act`                   | For models that don't have a quantize_config.json, this parameter is used to define whether to set desc_act or not in BaseQuantizeConfig. |
| `--disable_exllama`            | Disable ExLlama kernel, which can improve inference speed on some systems. |
| `--disable_exllamav2`          | Disable ExLlamav2 kernel. |

#### GPTQ-for-LLaMa

| Flag                      | Description |
|---------------------------|-------------|
| `--wbits WBITS`           | Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported. |
| `--model_type MODEL_TYPE` | Model type of pre-quantized model. Currently LLaMA, OPT, and GPT-J are supported. |
| `--groupsize GROUPSIZE`   | Group size. |
| `--pre_layer PRE_LAYER [PRE_LAYER ...]`  | The number of layers to allocate to the GPU. Setting this parameter enables CPU offloading for 4-bit models. For multi-gpu, write the numbers separated by spaces, eg `--pre_layer 30 60`. |
| `--checkpoint CHECKPOINT` | The path to the quantized checkpoint file. If not specified, it will be automatically detected. |
| `--monkey-patch`          | Apply the monkey patch for using LoRAs with quantized models. |

#### HQQ

| Flag        | Description |
|-------------|-------------|
| `--hqq-backend` | Backend for the HQQ loader. Valid options: PYTORCH, PYTORCH_COMPILE, ATEN. |

#### DeepSpeed

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--deepspeed`                         | Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration. |
| `--nvme-offload-dir NVME_OFFLOAD_DIR` | DeepSpeed: Directory to use for ZeRO-3 NVME offloading. |
| `--local_rank LOCAL_RANK`             | DeepSpeed: Optional argument for distributed setups. |

#### RoPE (for llama.cpp, ExLlamaV2, and transformers)

| Flag             | Description |
|------------------|-------------|
| `--alpha_value ALPHA_VALUE`           | Positional embeddings alpha factor for NTK RoPE scaling. Use either this or `compress_pos_emb`, not both. |
| `--rope_freq_base ROPE_FREQ_BASE`     | If greater than 0, will be used instead of alpha_value. Those two are related by `rope_freq_base = 10000 * alpha_value ^ (64 / 63)`. |
| `--compress_pos_emb COMPRESS_POS_EMB` | Positional embeddings compression factor. Should be set to `(context length) / (model's original context length)`. Equal to `1/rope_freq_scale`. |

#### Gradio

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--listen`                            | Make the web UI reachable from your local network. |
| `--listen-port LISTEN_PORT`           | The listening port that the server will use. |
| `--listen-host LISTEN_HOST`           | The hostname that the server will use. |
| `--share`                             | Create a public URL. This is useful for running the web UI on Google Colab or similar. |
| `--auto-launch`                       | Open the web UI in the default browser upon launch. |
| `--gradio-auth USER:PWD`              | Set Gradio authentication password in the format "username:password". Multiple credentials can also be supplied with "u1:p1,u2:p2,u3:p3". |
| `--gradio-auth-path GRADIO_AUTH_PATH` | Set the Gradio authentication file path. The file should contain one or more user:password pairs in the same format as above. |
| `--ssl-keyfile SSL_KEYFILE`           | The path to the SSL certificate key file. |
| `--ssl-certfile SSL_CERTFILE`         | The path to the SSL certificate cert file. |

#### API

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--api`                               | Enable the API extension. |
| `--public-api`                        | Create a public URL for the API using Cloudfare. |
| `--public-api-id PUBLIC_API_ID`       | Tunnel ID for named Cloudflare Tunnel. Use together with public-api option. |
| `--api-port API_PORT`                 | The listening port for the API. |
| `--api-key API_KEY`                   | API authentication key. |
| `--admin-key ADMIN_KEY`               | API authentication key for admin tasks like loading and unloading models. If not set, will be the same as --api-key. |
| `--nowebui`                           | Do not launch the Gradio UI. Useful for launching the API in standalone mode. |

#### Multimodal

| Flag                                  | Description |
|---------------------------------------|-------------|
| `--multimodal-pipeline PIPELINE`      | The multimodal pipeline to use. Examples: `llava-7b`, `llava-13b`. |

</details>

## Documentation

https://github.com/oobabooga/text-generation-webui/wiki

## Downloading models

Models should be placed in the folder `text-generation-webui/models`. They are usually downloaded from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads).

* GGUF models are a single file and should be placed directly into `models`. Example:

```
text-generation-webui
└── models
    └── llama-2-13b-chat.Q4_K_M.gguf
```

* The remaining model types (like 16-bit transformers models and GPTQ models) are made of several files and must be placed in a subfolder. Example:

```
text-generation-webui
├── models
│   ├── lmsys_vicuna-33b-v1.3
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── pytorch_model-00001-of-00007.bin
│   │   ├── pytorch_model-00002-of-00007.bin
│   │   ├── pytorch_model-00003-of-00007.bin
│   │   ├── pytorch_model-00004-of-00007.bin
│   │   ├── pytorch_model-00005-of-00007.bin
│   │   ├── pytorch_model-00006-of-00007.bin
│   │   ├── pytorch_model-00007-of-00007.bin
│   │   ├── pytorch_model.bin.index.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.model
```

In both cases, you can use the "Model" tab of the UI to download the model from Hugging Face automatically. It is also possible to download it via the command-line with 

```
python download-model.py organization/model
```

Run `python download-model.py --help` to see all the options.

## Google Colab notebook

https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb

## Contributing

If you would like to contribute to the project, check out the [Contributing guidelines](https://github.com/oobabooga/text-generation-webui/wiki/Contributing-guidelines).

## Community

* Subreddit: https://www.reddit.com/r/oobabooga/
* Discord: https://discord.gg/jwZCF2dPQN

## Acknowledgment

In August 2023, [Andreessen Horowitz](https://a16z.com/) (a16z) provided a generous grant to encourage and support my independent work on this project. I am **extremely** grateful for their trust and recognition.
