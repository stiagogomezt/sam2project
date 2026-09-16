"""
Motor de IA basado en Segment Anything Model 2 (SAM 2).
Soporta selección de modelos (tiny, small, large), aceleración CUDA con fallback a CPU y caching de embeddings.
"""
import os
import time
import urllib.request
from typing import Tuple, Dict, Any, Optional
import numpy as np
import streamlit as st
import torch
from config import SAM2_MODELS, DEFAULT_SAM2_MODEL_KEY

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    build_sam2 = None
    SAM2ImagePredictor = None

class SAM2Engine:
    """Clase envolvente para el predictor de SAM 2."""
    
    def __init__(self, model_key: str = DEFAULT_SAM2_MODEL_KEY):
        if build_sam2 is None or SAM2ImagePredictor is None:
            raise ImportError("El paquete 'sam2' no está instalado en el entorno.")
            
        if model_key not in SAM2_MODELS:
            model_key = DEFAULT_SAM2_MODEL_KEY
            
        self.model_info = SAM2_MODELS[model_key]
        self.model_key = model_key
        
        # Selección inteligente de dispositivo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
        
        checkpoint_path = self.model_info["checkpoint"]
        config_path = self.model_info["config"]
        
        # Descarga de pesos si no existen localmente
        if not os.path.exists(checkpoint_path):
            print(f"Descargando pesos de {model_key}...")
            urllib.request.urlretrieve(self.model_info["url"], checkpoint_path)
            
        # Construcción del modelo
        print(f"Cargando SAM 2 ({model_key}) en {self.device}...")
        sam2_model = build_sam2(config_path, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.current_image_id: Optional[str] = None
        print(f"SAM 2 inicializado exitosamente en {self.device_name}.")
        
    def set_image(self, image_array: np.ndarray, image_id: Optional[str] = None) -> float:
        """
        Precalcula y cachea los image embeddings en el modelo.
        Devuelve el tiempo transcurrido en segundos.
        """
        start_time = time.time()
        self.predictor.set_image(image_array)
        self.current_image_id = image_id
        elapsed = time.time() - start_time
        return elapsed

    def predict_point(
        self,
        col: int,
        row: int,
        multimask_output: bool = True
    ) -> Dict[str, Any]:
        """
        Ejecuta la predicción para un punto dado (col=X, row=Y) sobre la imagen pre-cargada.
        
        Returns:
            Dict con:
                - 'best_mask': Máscara binaria booleana (H, W) del mejor candidato.
                - 'all_masks': Array de máscaras candidatas (N, H, W).
                - 'scores': Puntuaciones de confianza de cada máscara.
                - 'best_score': Puntuación más alta.
                - 'inference_time': Tiempo en segundos.
                - 'device': Dispositivo utilizado.
        """
        if self.predictor is None:
            raise RuntimeError("El predictor de SAM 2 no está inicializado.")
            
        start_time = time.time()
        
        point_coords = np.array([[col, row]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32) # 1 = Foreground
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output,
        )
        
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(bool)
        best_score = float(scores[best_idx])
        
        elapsed = time.time() - start_time
        
        return {
            "best_mask": best_mask,
            "all_masks": masks,
            "scores": [float(s) for s in scores],
            "best_score": best_score,
            "best_idx": best_idx,
            "inference_time": elapsed,
            "device": self.device_name,
            "clicked_point": (col, row)
        }

@st.cache_resource(show_spinner="Inicializando modelo SAM 2 en memoria...")
def get_sam2_engine(model_key: str = DEFAULT_SAM2_MODEL_KEY) -> SAM2Engine:
    """Carga y cachea la instancia del motor SAM 2."""
    return SAM2Engine(model_key=model_key)
