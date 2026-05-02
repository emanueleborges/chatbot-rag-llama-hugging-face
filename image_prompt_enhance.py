"""Enriquece prompts para geração de imagem (modelos tendem ao inglês e confundem espécies)."""
from __future__ import annotations


def _is_dog_topic(low: str) -> bool:
    """Identifica pedidos sobre cães, inclusive 'raça calma' sem a palavra 'cachorro'."""
    explicit = (
        "cachorro",
        "cão",
        "cao",
        "dog",
        "canino",
        "canina",
        "filhote",
        "cachorrinho",
        "au au",
        "latido",
        "vira-lata",
        "vira lata",
        "labrador",
        "golden",
        "bulldog",
        "poodle",
        "husky",
        "pastor",
        "spitz",
        "beagle",
        "pug",
        "chihuahua",
        "rottweiler",
        "pit bull",
        "pitbull",
        "dachshund",
        "salsicha",
    )
    if any(w in low for w in explicit):
        return True
    if any(w in low for w in ("gato", "felino", "gatinho", "miau")):
        return False
    # Em PT, "raça calma/tranquila" em perguntas casuais costuma referir-se a cão
    if ("raça" in low or "raca" in low) and any(
        w in low for w in ("calma", "calmo", "calmos", "tranquil", "manso", "mansos", "dócil", "docil", "quieto")
    ):
        return True
    return False


def _is_cat_topic(low: str) -> bool:
    if any(w in low for w in ("gato", "felino", "cat", "gatinho", "miau")):
        return True
    return False


def prepare_image_prompts(user_prompt: str) -> tuple[str, str]:
    """
    Devolve (prompt_positivo_reforçado, negative_prompt).
    Ajuda a evitar troca de espécies (ex.: pediu cachorro e saiu cavalo ou gato).
    """
    raw = (user_prompt or "").strip() or "scene"
    low = raw.lower()

    negative: list[str] = [
        "blurry",
        "low quality",
        "distorted",
        "deformed",
        "watermark",
        "text",
        "signature",
    ]

    # Reforço por tema (PT + EN — SD foi treinado sobretudo em tags em inglês)
    if _is_dog_topic(low):
        negative.extend(
            [
                "horse",
                "equine",
                "pony",
                "mare",
                "stallion",
                "donkey",
                "mule",
                "zebra",
                "cow",
                "cattle",
                "pig",
                "sheep",
                "goat",
                "cat",
                "feline",
                "kitten",
                "bird",
                "parrot",
                "eagle",
                "reptile",
                "snake",
                "lion",
                "tiger",
                "bear",
                "rabbit",
                "hamster",
                "mouse",
                "wildlife",
            ]
        )
        breed_hint = ""
        if "raça" in low or "raca" in low or "breed" in low:
            breed_hint = (
                "dog breed, pedigree or mixed breed dog, breed traits visible, "
                "examples of calm breeds: golden retriever, labrador, basset hound, bulldog; "
                "must remain clearly a domestic dog, not another species"
            )
        suffix = (
            "dog, domestic dog, canine, mammal dog, pet dog, four legs, dog snout, dog ears, "
            "photorealistic, single dog as main subject, sharp focus, full body or portrait of dog only"
        )
        if breed_hint:
            positive = f"{raw}, {breed_hint}, {suffix}"
        else:
            positive = f"{raw}, {suffix}"
    elif _is_cat_topic(low):
        negative.extend(
            [
                "dog",
                "canine",
                "puppy",
                "wolf",
                "fox",
                "horse",
                "bird",
                "lion",
            ]
        )
        positive = f"{raw}, cat, domestic cat, feline, photorealistic, single cat, sharp focus"
    elif any(w in low for w in ("cavalo", "égua", "egua", "horse", "equino")):
        negative.extend(["dog", "canine", "cat", "feline"])
        positive = f"{raw}, horse, equine animal, photorealistic, single horse, sharp focus"
    elif any(w in low for w in ("pessoa", "homem", "mulher", "menino", "menina", "human")):
        negative.extend(["animal head", "horse face", "wrong species"])
        positive = f"{raw}, human, photorealistic, sharp focus"
    else:
        positive = f"{raw}, photorealistic, high detail, accurate subject"

    return positive, ", ".join(negative)
