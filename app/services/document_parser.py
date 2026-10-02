import logging
from pathlib import Path
from typing import List
from pypdf import PdfReader
import PIL.Image
from google import genai
import io
import pymupdf

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
        """Extrae texto nativo o aplica OCR si el PDF es un escaneo."""
        try:
            reader = PdfReader(str(file_path))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            text = "\n".join(text_parts)
            if text.strip():
                return text

            logger.info(
                f"El PDF '{file_path.name}' no tiene capa de texto. Iniciando OCR..."
            )
            return self._ocr_pdf_with_vision(file_path)
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

    def _ocr_pdf_with_vision(self, file_path: Path) -> str:
        """Renderiza hasta cinco páginas y las procesa con la cascada Gemini."""
        extracted_pages = []

        with pymupdf.open(file_path) as document:
            max_pages = min(len(document), 5)
            for page_number in range(max_pages):
                logger.info(
                    f"Aplicando OCR a la página {page_number + 1} de {file_path.name}..."
                )
                page = document.load_page(page_number)
                pixmap = page.get_pixmap(dpi=150)
                image_bytes = pixmap.tobytes("png")

                with PIL.Image.open(io.BytesIO(image_bytes)) as image:
                    page_text = self._run_vision_cascade(
                        image,
                        f"Página {page_number + 1} del archivo {file_path.name}",
                    )
                extracted_pages.append(page_text)

        return "\n\n".join(extracted_pages)

    def _extract_from_image(self, file_path: Path) -> str:
        """Extrae texto de una imagen usando la cascada de modelos Gemini."""
        prompt = (
            "Eres un sistema de OCR experto. Extrae todo el texto visible en esta imagen "
            "de forma estructurada y precisa. Respeta el orden lógico de lectura. "
            "No inventes información, devuelve solo el texto extraído."
        )

        with PIL.Image.open(file_path) as image:
            return self._run_vision_cascade(image, file_path.name, prompt)

    def _run_vision_cascade(
        self,
        image: PIL.Image.Image,
        source_name: str,
        prompt: str | None = None,
    ) -> str:
        """Prueba los modelos Gemini configurados en orden para una imagen."""
        if prompt is None:
            prompt = (
                "Eres un sistema de OCR experto. Extrae todo el texto visible en esta imagen "
                "de forma estructurada y precisa. Respeta el orden lógico de lectura. "
                "No inventes información, devuelve solo el texto extraído."
            )

        gemini_models = [
            settings.GEMINI_VISION_MODEL,
            *settings.GEMINI_VISION_FALLBACK_MODELS,
        ]
        gemini_models = list(dict.fromkeys(gemini_models))
        last_gemini_error = None

        # Prueba los modelos Gemini en orden antes de activar el proveedor alternativo.
        for model_name in gemini_models:
            try:
                logger.info(f"Intentando OCR con {model_name} para {source_name}...")
                response = self.genai_client.models.generate_content(
                    model=model_name,
                    contents=[prompt, image]
                )
                return response.text
            except Exception as e:
                last_gemini_error = e
                logger.warning(f"Fallo en Gemini {model_name}: {e}")

        # En el futuro se puede añadir aquí otro proveedor multimodal como fallback.
        raise RuntimeError(
            f"Fallaron todos los modelos Gemini configurados para {source_name}."
        ) from last_gemini_error

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