"""Easter eggs: the part of the program that has no reason to exist.

Rules the engine enforces so this stays fun instead of infuriating:

  * an egg never changes program state, only what is drawn;
  * an egg never fires on top of an error the user needs to read;
  * an egg never fires twice in quick succession;
  * every egg is discoverable, and discovery is remembered.
"""

from offset.eggs.engine import Egg, EggEngine, Reveal, Trigger
from offset.eggs.catalogue import CATALOGUE, build_engine

__all__ = ["CATALOGUE", "Egg", "EggEngine", "Reveal", "Trigger", "build_engine"]
