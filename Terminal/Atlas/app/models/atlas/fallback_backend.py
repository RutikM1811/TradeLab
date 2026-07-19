from app.models.atlas.backend import AbstractInferenceBackend


class FallbackBackend(AbstractInferenceBackend):

    @property
    def name(self) -> str:
        return "fallback"

    def __init__(self, backends):
        self.backends = backends

    async def generate(self, prompt: str, **kwargs):
        last_exception = None

        for backend in self.backends:
            try:
                print(f"Trying {backend.name}")
                result = await backend.generate(prompt, **kwargs)
                print(f"{backend.name} succeeded")
                return result
            except Exception as ex:
                print(f"{backend.name} failed: {type(ex).__name__}: {ex}")
                last_exception = ex

        raise last_exception

    async def generate_from_conversation(
            self,
            conversation,
            system_prompt=None,
            **kwargs,
    ):
        last_exception = None

        for backend in self.backends:
            try:
                return await backend.generate_from_conversation(
                    conversation,
                    system_prompt,
                    **kwargs,
                )
            except Exception as ex:
                last_exception = ex

        raise last_exception