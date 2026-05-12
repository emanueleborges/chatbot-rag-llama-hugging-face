# image_gen.py
import httpx
import json
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Dict
import os

class ImageGenerator:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.sd_url = os.getenv("SD_API_URL", "http://127.0.0.1:7860")
        
    async def enhance_prompt(self, simple_prompt: str) -> Dict[str, str]:
        """Usa Ollama para melhorar o prompt para imagem"""
        
        system = """Você é um especialista em prompts para geração de imagens.
        Receba uma ideia simples e gere um prompt detalhado e uma negative prompt.
        Retorne APENAS JSON no formato: {"positive": "...", "negative": "..."}"""
        
        prompt = f"Transforme esta ideia em um prompt detalhado para Stable Diffusion: {simple_prompt}"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.ollama_url,
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "format": "json"
                }
            )
            
            data = response.json()
            try:
                result = json.loads(data["response"])
                return result
            except:
                # Fallback
                return {
                    "positive": f"A beautiful detailed illustration of {simple_prompt}, high quality, 4k, detailed",
                    "negative": "blurry, low quality, distorted, ugly"
                }
    
    async def generate_image(
        self,
        prompt: str,
        enhance: bool = True,
        steps: int = 25,
        width: int = 512,
        height: int = 512
    ) -> Optional[bytes]:
        """Gera imagem via Stable Diffusion API"""
        
        # Opcional: melhora o prompt com Ollama
        final_prompt = prompt
        negative_prompt = "blurry, low quality, distorted"
        
        if enhance:
            enhanced = await self.enhance_prompt(prompt)
            final_prompt = enhanced.get("positive", prompt)
            negative_prompt = enhanced.get("negative", negative_prompt)
        
        # Payload para Stable Diffusion
        payload = {
            "prompt": final_prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "width": width,
            "height": height,
            "sampler_index": "DPM++ 2M Karras",
            "cfg_scale": 7,
            "seed": -1,
            "batch_size": 1
        }
        
        async with httpx.AsyncClient() as client:
            try:
                # Envia para SD API
                response = await client.post(
                    f"{self.sd_url}/sdapi/v1/txt2img",
                    json=payload,
                    timeout=120.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Decodifica imagem base64
                    image_data = base64.b64decode(data["images"][0])
                    return image_data
                    
            except Exception as e:
                print(f"Erro na geração de imagem: {e}")
                return None
            
        return None
    
    def save_image(self, image_bytes: bytes, filename: str) -> str:
        """Salva imagem em disco"""
        os.makedirs("generated_images", exist_ok=True)
        filepath = f"generated_images/{filename}"
        
        image = Image.open(BytesIO(image_bytes))
        image.save(filepath)
        
        return filepath