import requests
import streamlit as st


API_URL = "http://api:8000/predict"


st.set_page_config(
    page_title="Mushroom Classifier",
    page_icon="🍄",
)

st.title("Mushroom Classification")
st.write("Enter mushroom characteristics to predict whether it is edible or poisonous.")


st.header("Cap")

cap_diameter = st.number_input(
    "Cap diameter",
    min_value=0.0,
    value=5.5,
)

cap_shape = st.text_input("Cap shape", value="x")
cap_surface = st.text_input("Cap surface", value="s")
cap_color = st.text_input("Cap color", value="n")
does_bruise_or_bleed = st.text_input(
    "Does bruise or bleed",
    value="f",
)


st.header("Gill")

gill_attachment = st.text_input("Gill attachment", value="a")
gill_spacing = st.text_input("Gill spacing", value="c")
gill_color = st.text_input("Gill color", value="n")


st.header("Stem")

stem_height = st.number_input(
    "Stem height",
    min_value=0.0,
    value=5.8,
)

stem_width = st.number_input(
    "Stem width",
    min_value=0.0,
    value=10.0,
)

stem_root = st.text_input("Stem root", value="Unknown")
stem_surface = st.text_input("Stem surface", value="s")
stem_color = st.text_input("Stem color", value="n")


st.header("Other characteristics")

veil_type = st.text_input("Veil type", value="Unknown")
veil_color = st.text_input("Veil color", value="w")
has_ring = st.text_input("Has ring", value="t")
ring_type = st.text_input("Ring type", value="p")
spore_print_color = st.text_input(
    "Spore print color",
    value="Unknown",
)
habitat = st.text_input("Habitat", value="d")
season = st.text_input("Season", value="a")


if st.button("Predict", type="primary"):
    payload = {
        "cap_diameter": cap_diameter,
        "cap_shape": cap_shape,
        "cap_surface": cap_surface,
        "cap_color": cap_color,
        "does_bruise_or_bleed": does_bruise_or_bleed,
        "gill_attachment": gill_attachment,
        "gill_spacing": gill_spacing,
        "gill_color": gill_color,
        "stem_height": stem_height,
        "stem_width": stem_width,
        "stem_root": stem_root,
        "stem_surface": stem_surface,
        "stem_color": stem_color,
        "veil_type": veil_type,
        "veil_color": veil_color,
        "has_ring": has_ring,
        "ring_type": ring_type,
        "spore_print_color": spore_print_color,
        "habitat": habitat,
        "season": season,
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        result = response.json()

        if result["class"] == "p":
            st.error("The mushroom is predicted to be POISONOUS.")
        else:
            st.success("The mushroom is predicted to be EDIBLE.")

        st.write(f"Predicted class: `{result['class']}`")

    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")