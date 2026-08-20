# Local Models Setup

This project runs entirely locally. No cloud APIs are required.

## 1. Local LLM: Ollama + Llama 3.1

### Installation

1. **Install Ollama:** https://ollama.com/download
   - Windows: Download and run the installer
   - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
   - macOS: Download from https://ollama.com/download

2. **Pull the Llama 3.1 model:**
   ```bash
   ollama pull llama3.1
   ```

3. **Start Ollama server:**
   ```bash
   ollama serve
   ```
   The server runs on `http://localhost:11434` by default.

4. **Verify it works:**
   ```bash
   ollama list
   ```
   You should see `llama3.1` in the list.

### Alternative Models

You can use any Ollama-compatible model:
```bash
ollama pull mistral        # Mistral 7B
ollama pull gemma          # Google Gemma
ollama pull phi3           # Microsoft Phi-3
```

Set the model in environment variable or `.env` file:
```bash
OLLAMA_MODEL=mistral
```

### Configuration

Environment variables (or `.env` file):
```bash
OLLAMA_BASE_URL=http://localhost:11434/v1   # default
OLLAMA_MODEL=llama3.1                        # default
```

## 2. Local Image Generation: Stable Diffusion

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install diffusers transformers accelerate torch pillow
   ```

2. **Download model weights (automatic on first use):**
   The first time you run image generation, Stable Diffusion v1.5 will be
   downloaded from Hugging Face (~4 GB). This happens automatically.

   To pre-download:
   ```python
   from diffusers import StableDiffusionPipeline
   pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5")
   ```

### GPU vs CPU

- **With NVIDIA GPU:** Model runs on GPU (much faster, ~2-5 seconds per image)
- **Without GPU:** Model runs on CPU (slower, ~30-120 seconds per image)

The system automatically detects and uses GPU if available.

### Alternative Models

You can use other Stable Diffusion models by changing the `model_id`:
```python
StableDiffusionGenerator(model_id="stabilityai/stable-diffusion-2-1")
```

### Disk Space

- Stable Diffusion v1.5: ~4 GB
- Llama 3.1 (8B): ~4.7 GB
- Total: ~9 GB

## 3. No Cloud APIs Required

This project does NOT use:
- OpenAI API
- Google Gemini
- Anthropic Claude
- Any other cloud AI service

All AI inference runs on your local machine.
