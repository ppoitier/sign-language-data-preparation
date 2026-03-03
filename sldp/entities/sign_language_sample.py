from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel


class SignLanguageSample(BaseModel):
    id: str
    signer_id: str
    sign_language: str
    dataset: Optional[str] = None

    label: Optional[str] = None
    label_id: Optional[int] = None

    poses: Optional[dict[str, np.ndarray]] = None
    video: Optional[bytes] = None
    video_path: Optional[str] = None

    annotations: Optional[dict[str, pd.DataFrame]] = None
