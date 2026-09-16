from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path("/app/models/mushroom_catboost.cbm")

app = FastAPI(
    title="Mushroom Classification API",
    description="API for predicting whether a mushroom is edible or poisonous.",
    version="1.0.0",
)

model = CatBoostClassifier()
model.load_model(MODEL_PATH)


class MushroomInput(BaseModel):
    cap_diameter: float
    cap_shape: str
    cap_surface: str
    cap_color: str
    does_bruise_or_bleed: str
    gill_attachment: str
    gill_spacing: str
    gill_color: str
    stem_height: float
    stem_width: float
    stem_root: str
    stem_surface: str
    stem_color: str
    veil_type: str
    veil_color: str
    has_ring: str
    ring_type: str
    spore_print_color: str
    habitat: str
    season: str


@app.get("/")
def root():
    return {
        "message": "Mushroom Classification API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: MushroomInput):
    features = pd.DataFrame(
        [
            {
                "cap-diameter": data.cap_diameter,
                "cap-shape": data.cap_shape,
                "cap-surface": data.cap_surface,
                "cap-color": data.cap_color,
                "does-bruise-or-bleed": data.does_bruise_or_bleed,
                "gill-attachment": data.gill_attachment,
                "gill-spacing": data.gill_spacing,
                "gill-color": data.gill_color,
                "stem-height": data.stem_height,
                "stem-width": data.stem_width,
                "stem-root": data.stem_root,
                "stem-surface": data.stem_surface,
                "stem-color": data.stem_color,
                "veil-type": data.veil_type,
                "veil-color": data.veil_color,
                "has-ring": data.has_ring,
                "ring-type": data.ring_type,
                "spore-print-color": data.spore_print_color,
                "habitat": data.habitat,
                "season": data.season,
            }
        ]
    )

    prediction = int(
        model.predict(features).flatten()[0]
    )

    predicted_class = "p" if prediction == 1 else "e"
    label = "poisonous" if predicted_class == "p" else "edible"

    return {
        "class": predicted_class,
        "prediction": label,
    }