"""[HITO 1] Tabla Hash con resolución de colisiones por encadenamiento (Separate Chaining).

Informe de complejidad Big-O
----------------------------
Notación: n = registros almacenados, m = `capacity` (número de buckets),
alpha = n / m (factor de carga), L = longitud de la clave `feature_id`.

`_hash` usa un hash polinomial (estilo djb2: h = h * 31 + ord(c)) reducido módulo m,
determinista entre ejecuciones (a diferencia de `hash()` nativo, que está aleatorizado).

| Operación      | Tiempo promedio   | Tiempo peor caso | Espacio extra |
|----------------|-------------------|------------------|---------------|
| _hash(key)     | O(L)              | O(L)             | O(1)          |
| put(record)    | O(1 + alpha)      | O(n)             | O(1) (un HashNode nuevo) |
| search(id)     | O(1 + alpha)      | O(n)             | O(1)          |
| delete(id)     | O(1 + alpha)      | O(n)             | O(1)          |
| load_factor()  | O(1)              | O(1)             | O(1)          |

Bajo la hipótesis de dispersión uniforme, cada cadena tiene longitud esperada alpha;
si alpha se mantiene acotado (alpha = O(1)), put/search/delete son O(1) promedio
(L se considera constante para identificadores cortos). El peor caso O(n) ocurre
cuando todas las claves colisionan en un mismo bucket (p. ej. capacity = 1), y la
tabla degenera en una única lista enlazada.

Espacio total de la estructura: O(m + n) (arreglo de buckets + un nodo por registro).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.settings import DEFAULT_HASH_CAPACITY, FeatureVector


@dataclass
class HashNode:
    """Nodo de lista simplemente enlazada para un bucket de la Tabla Hash."""

    key: str
    value: FeatureVector
    next: Optional[HashNode] = None


class FeatureHashTable:
    """Tabla Hash con encadenamiento separado para indexar objetos `FeatureVector` por ID.

    IMPORTANTE: Prohibido delegar el almacenamiento a un `dict` nativo de Python.
    Debe manipularse la lista `self.buckets` y los nodos enlazados `HashNode`.
    """

    def __init__(self, capacity: int = DEFAULT_HASH_CAPACITY) -> None:
        """Inicializa la tabla hash con un número fijo de buckets.

        Raises:
            ValueError: Si `capacity <= 0` o no es un entero válido.
        """
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("La capacidad de la tabla hash debe ser un entero positivo.")
        self.capacity: int = capacity
        self.buckets: list[Optional[HashNode]] = [None] * self.capacity
        self._size: int = 0

    def __len__(self) -> int:
        """Devuelve la cantidad de registros únicos almacenados."""
        return self._size

    def _hash(self, key: str) -> int:
        """Calcula el índice del bucket en `[0, self.capacity - 1]` para la clave `key`.

        Requisitos de evaluación:
        - Validar que `key` sea un `str` no vacío (`ValueError` si está vacío o solo espacios).
        - Implementar una función de dispersión determinista módulo `self.capacity`.
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError("La clave debe ser una cadena no vacía.")
        hash_value = 5381
        for char in key:
            hash_value = (hash_value * 31 + ord(char)) & 0xFFFFFFFF
        return hash_value % self.capacity

    def put(self, record: FeatureVector) -> None:
        """Inserta o actualiza un `FeatureVector` en la tabla hash en O(1) promedio.

        Requisitos de evaluación (Casos Borde):
        - Validar que `record` sea instancia de `FeatureVector`.
        - Si `record.feature_id` ya existe en la lista enlazada del bucket, actualizar
          `current.value = record` sin incrementar `_size`.
        - Si hay colisión con otra clave distinta, enlazar un nuevo `HashNode` en el bucket.
        """
        if not isinstance(record, FeatureVector):
            raise TypeError("Solo se pueden insertar instancias de FeatureVector.")
        index = self._hash(record.feature_id)

        current = self.buckets[index]
        while current is not None:
            if current.key == record.feature_id:
                current.value = record  # Actualización: no cambia el tamaño.
                return
            current = current.next

        # Clave nueva: se inserta al inicio de la cadena en O(1).
        self.buckets[index] = HashNode(
            key=record.feature_id, value=record, next=self.buckets[index]
        )
        self._size += 1

    def search(self, feature_id: str) -> Optional[FeatureVector]:
        """Recupera un `FeatureVector` por su identificador en O(1) promedio.

        Returns:
            La instancia `FeatureVector` si se encuentra en la cadena del bucket, o `None`.
        """
        index = self._hash(feature_id)
        current = self.buckets[index]
        while current is not None:
            if current.key == feature_id:
                return current.value
            current = current.next
        return None

    def delete(self, feature_id: str) -> bool:
        """Elimina un registro de la lista enlazada del bucket correspondiente.

        Returns:
            `True` si el nodo fue encontrado y desvinculado; `False` si no existía.
        """
        index = self._hash(feature_id)
        previous: Optional[HashNode] = None
        current = self.buckets[index]
        while current is not None:
            if current.key == feature_id:
                if previous is None:
                    self.buckets[index] = current.next  # Era la cabeza de la cadena.
                else:
                    previous.next = current.next
                self._size -= 1
                return True
            previous = current
            current = current.next
        return False

    def load_factor(self) -> float:
        """Calcula el factor de carga actual `alpha = n / capacity`."""
        return self._size / self.capacity

    def clear(self) -> None:
        """Reinicia todos los buckets de la tabla hash."""
        self.buckets = [None] * self.capacity
        self._size = 0


# Alias de compatibilidad requerido por el enunciado
CustomHashTable = FeatureHashTable

