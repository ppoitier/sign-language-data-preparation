from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


class SignLanguageSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    sign_language: str
    signer_id: Optional[str] = None

    label: Optional[str] = None
    label_id: Optional[int] = None

    poses: Optional[dict[str, np.ndarray]] = None
    video: Optional[bytes] = None
    video_path: Optional[str] = None

    # --- Continuous sample optional data ---

    annotations: Optional[dict[str, pd.DataFrame]] = None

    # --- Isolate sample optional data ---

    parent_sample_id: Optional[str] = None

    start_ms: Optional[int] = None
    end_ms: Optional[int] = None

    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    linguistic_metadata: Optional[dict[str, str]] = None
