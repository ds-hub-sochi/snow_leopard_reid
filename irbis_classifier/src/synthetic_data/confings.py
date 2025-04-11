from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class OneSpecieConfig:
    russian_label: str
    prompt: Sequence[str]
    guidance_scale: float
    number_of_images: int
    prompt_2: Sequence[str] = ('',)
    negative_prompt: Sequence[str] = ('',)
    negative_prompt_2: Sequence[str] = ('',)

    def __post_init__(self):
        self.prompt_2 = list(self.prompt_2) + [''] * (len(self.prompt) - len(self.prompt_2))
        self.negative_prompt = list(self.negative_prompt) + [''] * (len(self.prompt) - len(self.negative_prompt))
        self.negative_prompt_2 = list(self.negative_prompt_2) + [''] * (len(self.prompt) - len(self.negative_prompt_2))


@dataclass
class GenerationConfig:
    species_confings: Sequence[OneSpecieConfig]

    def __post_init__(self):
        for i, specied_config in enumerate(self.species_confings):
            self.species_confings[i] = OneSpecieConfig(**specied_config)


    def __len__(self) -> int:
        return len(self.species_confings)
    
    def __getitem__(
        self,
        index: int,
    ) -> OneSpecieConfig:
        return self.species_confings[index]
