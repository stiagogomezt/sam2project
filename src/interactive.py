"""
Sesión interactiva de segmentación de UN objeto (independiente de Streamlit).

Modela la secuencia de prompts que el usuario va dando sobre la imagen y la reproduce con SAM 2:

- Puntos positivos (label=1) y negativos (label=0), acumulables.
- Una caja (xmin, ymin, xmax, ymax), introducida con dos clics (esquinas opuestas).
- Refinamiento iterativo oficial de SAM: desde el segundo prompt se entrega como `mask_input` el
  logit de baja resolución (256x256) de la candidata elegida en el paso anterior.
- Salida multimáscara (3 candidatas) solo cuando el prompt es ambiguo: un único punto sin caja.
  Con varios puntos o con caja, SAM recomienda una sola salida.
- Deshacer = reproducir la secuencia sin el último evento (la inferencia en CPU es determinista y
  el decodificador cuesta ~0,1 s, por lo que no se guardan máscaras intermedias en memoria).

Qué es oficial de SAM 2 y qué es de esta aplicación: los tipos de prompt, `mask_input`,
`multimask_output` y el IoU predicho son del modelo; el historial, deshacer, la regla de cuándo
pedir 3 candidatas y la elección manual de candidata son lógica de esta aplicación.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PromptEvent:
    kind: str  # "point" | "box"
    x: float = 0.0
    y: float = 0.0
    label: int = 1
    box: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "x": self.x, "y": self.y, "label": self.label,
                "box": list(self.box) if self.box else None}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PromptEvent":
        return PromptEvent(kind=d["kind"], x=float(d.get("x", 0)), y=float(d.get("y", 0)),
                           label=int(d.get("label", 1)), box=tuple(d["box"]) if d.get("box") else None)


@dataclass
class InteractiveSegmenter:
    refine_with_logits: bool = True
    events: List[PromptEvent] = field(default_factory=list)
    selections: List[int] = field(default_factory=list)  # candidata elegida tras cada evento
    pending_corner: Optional[Tuple[float, float]] = None
    result: Optional[Dict[str, Any]] = None  # salida de SAM2Engine.predict del último evento

    # ------------------------------------------------------------------ consultas
    @property
    def is_empty(self) -> bool:
        return not self.events and self.pending_corner is None

    @property
    def selected_idx(self) -> int:
        return self.selections[-1] if self.selections else 0

    @property
    def candidates(self) -> list:
        return self.result["candidates"] if self.result else []

    @property
    def selected_candidate(self):
        c = self.candidates
        return c[self.selected_idx] if c else None

    def prompts_upto(self, n: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """(points, labels, box) acumulados de los n primeros eventos."""
        pts = [(e.x, e.y) for e in self.events[:n] if e.kind == "point"]
        lbl = [e.label for e in self.events[:n] if e.kind == "point"]
        boxes = [e.box for e in self.events[:n] if e.kind == "box"]
        points = np.array(pts, dtype=np.float32) if pts else None
        labels = np.array(lbl, dtype=np.int32) if lbl else None
        box = np.array(boxes[-1], dtype=np.float32) if boxes else None
        return points, labels, box

    def summary(self) -> Dict[str, int]:
        return {
            "positivos": sum(1 for e in self.events if e.kind == "point" and e.label == 1),
            "negativos": sum(1 for e in self.events if e.kind == "point" and e.label == 0),
            "caja": int(any(e.kind == "box" for e in self.events)),
        }

    # ------------------------------------------------------------------ ejecución
    def _run_step(self, engine, emb, i: int, prev_logits: Optional[np.ndarray]) -> Dict[str, Any]:
        points, labels, box = self.prompts_upto(i + 1)
        n_points = 0 if points is None else len(points)
        multimask = box is None and n_points == 1
        mask_logits = prev_logits if (self.refine_with_logits and i > 0) else None
        return engine.predict(emb, points=points, labels=labels, box=box,
                              mask_logits=mask_logits, multimask=multimask)

    def _replay(self, engine, emb) -> None:
        """Recalcula toda la secuencia (tras deshacer o al reabrir un objeto)."""
        self.result = None
        prev_logits = None
        for i in range(len(self.events)):
            self.result = self._run_step(engine, emb, i, prev_logits)
            n = len(self.result["candidates"])
            # La selección guardada solo es válida si ese paso produjo esa candidata
            if i >= len(self.selections) or self.selections[i] >= n:
                sel = self.result["best_idx"]
                if i < len(self.selections):
                    self.selections[i] = sel
                else:
                    self.selections.append(sel)
            prev_logits = self.result["candidates"][self.selections[i]].logits
        del self.selections[len(self.events):]

    def add_event(self, engine, emb, event: PromptEvent) -> Dict[str, Any]:
        if event.kind == "box" and event.box is not None:
            x0, y0, x1, y1 = event.box
            event.box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            if event.box[2] - event.box[0] < 2 or event.box[3] - event.box[1] < 2:
                raise ValueError("La caja es demasiado pequeña (mínimo 2 px por lado).")
        prev_logits = self.selected_candidate.logits if self.selected_candidate is not None else None
        self.events.append(event)
        self.result = self._run_step(engine, emb, len(self.events) - 1, prev_logits)
        self.selections.append(self.result["best_idx"])
        return self.result

    def add_point(self, engine, emb, x: float, y: float, positive: bool = True) -> Dict[str, Any]:
        return self.add_event(engine, emb, PromptEvent("point", x=x, y=y, label=1 if positive else 0))

    def click_box_corner(self, engine, emb, x: float, y: float) -> Optional[Dict[str, Any]]:
        """Primer clic: guarda la esquina. Segundo clic: crea la caja y ejecuta SAM 2."""
        if self.pending_corner is None:
            self.pending_corner = (x, y)
            return None
        x0, y0 = self.pending_corner
        self.pending_corner = None
        return self.add_event(engine, emb, PromptEvent("box", box=(x0, y0, x, y)))

    def select(self, idx: int) -> None:
        if not self.candidates or not (0 <= idx < len(self.candidates)):
            raise IndexError("Candidata inexistente")
        self.selections[-1] = idx

    def undo(self, engine, emb) -> None:
        if self.pending_corner is not None:
            self.pending_corner = None
            return
        if not self.events:
            return
        self.events.pop()
        self.selections.pop()
        self._replay(engine, emb)

    def clear(self) -> None:
        self.events.clear()
        self.selections.clear()
        self.pending_corner = None
        self.result = None

    # ------------------------------------------------------------------ persistencia
    def to_dict(self) -> Dict[str, Any]:
        return {"events": [e.to_dict() for e in self.events], "selections": list(self.selections),
                "refine_with_logits": self.refine_with_logits}

    @staticmethod
    def from_dict(d: Dict[str, Any], engine=None, emb=None) -> "InteractiveSegmenter":
        seg = InteractiveSegmenter(refine_with_logits=bool(d.get("refine_with_logits", True)))
        seg.events = [PromptEvent.from_dict(e) for e in d.get("events", [])]
        seg.selections = [int(s) for s in d.get("selections", [])]
        if engine is not None and emb is not None and seg.events:
            seg._replay(engine, emb)
        return seg
