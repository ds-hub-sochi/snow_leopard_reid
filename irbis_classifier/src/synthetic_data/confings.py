from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class SingleSpecieConfig:
    russian_label: str
    prompt: Sequence[str]
    guidance_scale: float
    number_of_images: int
    prompt_2: Sequence[str] = ('',)
    negative_prompt: Sequence[str] = ('',)
    negative_prompt_2: Sequence[str] = ('',)

    def __post_init__(self):
        n_prompts: int = len(self.prompt)

        self.prompt_2 = list(self.prompt_2) + [''] * (n_prompts - len(self.prompt_2))
        self.negative_prompt = list(self.negative_prompt) + [''] * (n_prompts - len(self.negative_prompt))
        self.negative_prompt_2 = list(self.negative_prompt_2) + [''] * (n_prompts - len(self.negative_prompt_2))


@dataclass
class GenerationConfig:
    species_confings: Sequence[SingleSpecieConfig]

    def __post_init__(self):
        for i, specied_config in enumerate(self.species_confings):
            self.species_confings[i] = SingleSpecieConfig(**specied_config)

    def __len__(self) -> int:
        return len(self.species_confings)
    
    def __getitem__(
        self,
        index: int,
    ) -> SingleSpecieConfig:
        return self.species_confings[index]
