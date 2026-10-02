import logging
from pathlib import Path
from typing import List
from pypdf import PdfReader
import PIL.Image
from google import genai
import base64
import io
from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

class DocumentParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Instanciamos el cliente de Gemini para usar su capacidad de Visión/OCR
        self.genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def extract_text(self, file_path: Path) -> str:
        """Enruta dinámicamente la extracción dependiendo del tipo de archivo."""
        ext = file_path.suffix.lower()
        if ext == '.pdf':
            return self._extract_from_pdf(file_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            return self._extract_from_image(file_path)
        elif ext == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Formato no soportado por el sistema: {ext}")

    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extrae texto de un PDF usando procesamiento local."""
        try:
            reader = PdfReader(str(file_path))
            text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
            if not text.strip():
                raise ValueError("El PDF está vacío o requiere OCR (es un scan sin capa de texto).")
            return text
        except Exception as e:
            logger.error(f"Error procesando PDF {file_path.name}: {e}")
            raise

    def _extract_from_txt(self, file_path: Path) -> str:
        """Extrae texto plano de un archivo .txt."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            if not text.strip():
                raise ValueError("El archivo de texto está vacío.")
            return text
        except Exception as e:
            logger.error(f"Error leyendo archivo de texto {file_path.name}: {e}")
            raise

    def _extract_from_image(self, file_path: Path) -> str:
        """Extrae texto usando Gemini con fallback a OpenAI en caso de fallo."""
        prompt = (
            "Eres un sistema de OCR experto. Extrae todo el texto visible en esta imagen "
            "de forma estructurada y precisa. Respeta el orden lógico de lectura. "
            "No inventes información, devuelve solo el texto extraído."
        )

        # Intento 1: Gemini (Principal)
        try:
            logger.info(f"Intentando OCR con Gemini 2.5 Flash para {file_path.name}...")
            image = PIL.Image.open(file_path)
            response = self.genai_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, image]
            )
            return response.text
        except Exception as e:
            logger.warning(f"Fallo en Gemini ({e}). Iniciando Fallback a OpenAI...")

            # Intento 2: OpenAI Fallback (GPT-4o-mini)
            if not settings.OPENAI_API_KEY:
                raise RuntimeError("Falló Gemini y no hay OPENAI_API_KEY configurada para el fallback.")
            
            return self._fallback_openai_vision(file_path, prompt)

    def _fallback_openai_vision(self, file_path: Path, prompt: str) -> str:
        """Ejecuta la llamada a OpenAI transformando la imagen a Base64."""
        try:
            logger.info("Ejecutando fallback con OpenAI (gpt-4o-mini)...")
            openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error crítico: Ambos proveedores fallaron para {file_path.name}. Detalle: {e}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        """Divide el texto en chunks con solapamiento estricto."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap

        return chunks