**Rodando uma IA Localmente com GPU NVIDIA de 12GB e CUDA**

## 1. Requisitos Iniciais

Antes de configurar a IA localmente, certifique-se de ter os seguintes requisitos atendidos:
- **GPU NVIDIA compatível** (mínimo 12GB de VRAM)
- **Drivers da NVIDIA atualizados**
- **CUDA instalado e configurado corretamente**
- **Python 3.9 ou superior**
- **PyTorch ou TensorFlow com suporte a CUDA**
- **Modelos de IA otimizados para execução local**

---

## 2. Instalando os Drivers da NVIDIA e CUDA

### 2.1 Verificar compatibilidade da GPU

Execute:
```bash
nvidia-smi
```
Isso listará as informações da sua GPU, incluindo compatibilidade com CUDA.

### 2.2 Instalar CUDA Toolkit e cuDNN

Instale as versões mais recentes diretamente do site oficial da NVIDIA:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### 2.2.1 CUDA Toolkit Installer

Instruções:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

Instale os drivers CUDA.  Escolha entre o Open Source, e o proprietário.

```
sudo apt-get install -y nvidia-open
```

Ou o proprietário:

```
sudo apt-get install -y cuda-drivers
```

### 2.2.2 Instale NVIDIA CUDA® Deep Neural Network library (cuDNN) 

No meu caso, escolhi o frontend, por facilidade de uso e maior abstração.

Consulte:
- [cuDNN](https://developer.nvidia.com/cudnn)

Para a instalação, siga as instruções do repo:

- [REPO_cuDNN](https://github.com/NVIDIA/cudnn-frontend)

Após instalar, verifique a versão do CUDA:
```bash
nvcc --version
```

---

## 3. Configurando o Ambiente Python

### 3.1 Criar um ambiente virtual

```bash
python3 -m venv ia_local
source ia_local/bin/activate
```

### 3.2 Instalar PyTorch ou TensorFlow com suporte a CUDA

#### PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
#### TensorFlow

```bash
pip install tensorflow==2.15.0
```
Verifique se o PyTorch detecta a GPU:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```
Se retornar `True`, sua GPU está funcionando corretamente.

---

## 4. Baixando e Executando um Modelo de IA

Escolha um modelo conforme sua necessidade:

### 4.1 Modelos de IA Populares

- **LLMs (Modelos de Linguagem):** Llama 2, Mistral, Phi-2, GPTQ
- **Visão Computacional:** YOLOv8, Stable Diffusion
- **Voz e Áudio:** Whisper, FastSpeech2

### 4.2 Baixando um Modelo

Exemplo com **Llama 2** via `llama-cpp-python`:
```bash
pip install llama-cpp-python
```
Depois, faça o download do modelo compatível no formato GGUF:
```bash
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf -P models/
```
E execute:
```python
from llama_cpp import Llama
llm = Llama(model_path="models/llama-2-7b.Q4_K_M.gguf")
print(llm("Qual é a capital do Brasil?"))
```

---

## 5. Considerações Finais

- Ajuste a **quantidade de memória VRAM utilizada** com parâmetros como `max_seq_len`.
- Use **inferência quantizada** para rodar modelos grandes em GPUs com menos memória.
- Considere usar frameworks como **vLLM** e **Text Generation Web UI** para interfaces amigáveis.


