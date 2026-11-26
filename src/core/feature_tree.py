"""[HITO 1] Árbol Binario de Búsqueda (BST) para indexación por puntaje de Data Drift.

Informe de complejidad Big-O
----------------------------
Notación: n = número de vectores almacenados, u = número de nodos (drift_score
distintos, u <= n), h = altura del árbol, k = número de vectores devueltos.
En un BST sin balanceo, h = O(log u) en el caso promedio (inserciones en orden
aleatorio) y h = O(u) en el peor caso (inserciones ya ordenadas -> árbol degenerado
en lista enlazada).

| Operación            | Tiempo promedio     | Tiempo peor caso | Espacio extra          |
|----------------------|---------------------|------------------|------------------------|
| insert(record)       | O(log u + d)        | O(u + d)         | O(1) (iterativo)       |
| search(drift_score)  | O(log u)            | O(u)             | O(d) (copia de la lista)|
| range_query(a, b)    | O(log u + k)        | O(u + k)         | O(k + h) (resultado + pila de recursión) |
| inorder_traversal()  | O(n)                | O(n)             | O(n + h)               |

donde d = cantidad de vectores que ya comparten el mismo drift_score (para
verificar si el `feature_id` existe y actualizarlo en lugar de duplicarlo; en la
práctica d es pequeño, O(1)).

`range_query` aplica poda: no desciende a la izquierda si `node.drift_score <= min_drift`
ni a la derecha si `node.drift_score >= max_drift`, por lo que solo visita los nodos del
camino hacia los límites más los nodos dentro del rango -> O(h + k).

Espacio total de la estructura: O(n) (u nodos + n referencias a vectores).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config.settings import FeatureVector


def _validate_drift(value: float, name: str = "drift_score") -> float:
    """Valida que `value` sea numérico (no bool) y esté dentro de [0.0, 1.0]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} debe ser numérico; se recibió {value!r}.")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} debe estar en [0.0, 1.0]; se recibió {value}.")
    return value


@dataclass
class BSTNode:
    """Nodo del Árbol Binario de Búsqueda indexado por `drift_score`.

    Attributes:
        drift_score: Clave numérica de ordenamiento en el rango [0.0, 1.0].
        records: Lista de vectores que comparten exactamente este `drift_score`.
        left: Subárbol izquierdo (`drift_score` menores).
        right: Subárbol derecho (`drift_score` mayores).
    """

    drift_score: float
    records: list[FeatureVector] = field(default_factory=list)
    left: Optional[BSTNode] = None
    right: Optional[BSTNode] = None


class DriftIndexBST:
    """Árbol Binario de Búsqueda que organiza vectores según su Drift Score [0.0, 1.0]."""

    def __init__(self) -> None:
        """Inicializa un árbol BST vacío."""
        self.root: Optional[BSTNode] = None
        self._size: int = 0

    def __len__(self) -> int:
        """Devuelve la cantidad total de vectores indexados en el árbol."""
        return self._size

    def insert(self, record: FeatureVector) -> None:
        """Inserta un registro `FeatureVector` en el árbol según su `drift_score`.

        Requisitos de evaluación (Casos Borde):
        - Validar que `record` sea instancia de `FeatureVector` (`TypeError` si no).
        - Validar que `0.0 <= record.drift_score <= 1.0` (`ValueError` si está fuera de rango).
        - Soportar múltiples vectores con el mismo `drift_score` almacenándolos en `node.records`.
        - Si el `feature_id` ya existe en ese nodo, actualizarlo sin incrementar `_size`.
        """
        if not isinstance(record, FeatureVector):
            raise TypeError("Solo se pueden insertar instancias de FeatureVector.")
        score = _validate_drift(record.drift_score)

        # Árbol vacío: el registro se convierte en la raíz.
        if self.root is None:
            self.root = BSTNode(drift_score=score, records=[record])
            self._size += 1
            return

        current = self.root
        while True:
            if score < current.drift_score:
                if current.left is None:
                    current.left = BSTNode(drift_score=score, records=[record])
                    self._size += 1
                    return
                current = current.left
            elif score > current.drift_score:
                if current.right is None:
                    current.right = BSTNode(drift_score=score, records=[record])
                    self._size += 1
                    return
                current = current.right
            else:
                # Mismo drift_score: actualizar si el feature_id ya existe, o agregar.
                for i, existing in enumerate(current.records):
                    if existing.feature_id == record.feature_id:
                        current.records[i] = record
                        return
                current.records.append(record)
                self._size += 1
                return

    def search(self, drift_score: float) -> list[FeatureVector]:
        """Busca todos los vectores que tengan exactamente el `drift_score` indicado.

        Requisitos de evaluación:
        - Validar que `drift_score` esté en `[0.0, 1.0]` (`ValueError` si no).
        - Retornar lista vacía `[]` si el árbol está vacío o el puntaje no existe.
        """
        score = _validate_drift(drift_score)
        current = self.root
        while current is not None:
            if score < current.drift_score:
                current = current.left
            elif score > current.drift_score:
                current = current.right
            else:
                return list(current.records)
        return []

    def range_query(self, min_drift: float, max_drift: float) -> list[FeatureVector]:
        """Extrae en orden ascendente todos los vectores cuyo drift esté en [min_drift, max_drift].

        Requisitos de evaluación:
        - Validar que `0.0 <= min_drift <= 1.0` y `0.0 <= max_drift <= 1.0`.
        - Lanzar `ValueError` si `min_drift > max_drift`.
        - Aplicar poda recursiva (no recorrer subárbol izquierdo si `node.drift_score <= min_drift`,
          ni subárbol derecho si `node.drift_score >= max_drift`) para lograr O(log n + k).
        """
        low = _validate_drift(min_drift, "min_drift")
        high = _validate_drift(max_drift, "max_drift")
        if low > high:
            raise ValueError("min_drift no puede ser mayor que max_drift.")

        result: list[FeatureVector] = []

        def _collect(node: Optional[BSTNode]) -> None:
            if node is None:
                return
            # Solo puede haber valores en el rango a la izquierda si node > low.
            if node.drift_score > low:
                _collect(node.left)
            if low <= node.drift_score <= high:
                result.extend(node.records)
            # Solo puede haber valores en el rango a la derecha si node < high.
            if node.drift_score < high:
                _collect(node.right)

        _collect(self.root)
        return result

    def inorder_traversal(self) -> list[FeatureVector]:
        """Devuelve todos los vectores almacenados ordenados ascendentemente por `drift_score`."""
        result: list[FeatureVector] = []

        def _inorder(node: Optional[BSTNode]) -> None:
            if node is None:
                return
            _inorder(node.left)
            result.extend(node.records)
            _inorder(node.right)

        _inorder(self.root)
        return result

    def clear(self) -> None:
        """Vacía el árbol por completo."""
        self.root = None
        self._size = 0


# Alias de compatibilidad requerido por el enunciado
BinarySearchTree = DriftIndexBST

