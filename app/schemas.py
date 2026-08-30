from typing import Optional

from pydantic import BaseModel, Field

# Upper bounds are the training-set maxima plus 20% headroom, so a reading a
# little beyond anything observed is still served while a physically
# impossible one is rejected. Regenerate after a dataset change with:
#
#     python -m src.compute_bounds
#
# Baked in as literals rather than read from the CSV at import time because
# .dockerignore excludes data/ - a schema that read it at startup would crash
# the container on boot.


class WaterInput(BaseModel):
    ph:              Optional[float] = Field(None, ge=0, le=14)  # scale definition
    Hardness:        Optional[float] = Field(None, ge=0, le=387.749)
    Solids:          Optional[float] = Field(None, ge=0, le=73472.6)
    Chloramines:     Optional[float] = Field(None, ge=0, le=15.7524)
    Sulfate:         Optional[float] = Field(None, ge=0, le=577.237)
    Conductivity:    Optional[float] = Field(None, ge=0, le=904.011)
    Organic_carbon:  Optional[float] = Field(None, ge=0, le=33.96)
    Trihalomethanes: Optional[float] = Field(None, ge=0, le=148.8)
    Turbidity:       Optional[float] = Field(None, ge=0, le=8.0868)

    model_config = {"extra": "forbid", "allow_inf_nan": False}
