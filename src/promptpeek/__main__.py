import asyncio
import logging

import promptpeek.attack as attack
import promptpeek.attack_ref as attack_ref

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)


# async def main() -> None:
#     result = await attack_ref.promptpeek(prompt_hint="Imagine you are a")
#     print("\n=== Attack Result ===")
#     print(f"Reconstructed prompt: {result}")


# asyncio.run(main())

attack.main()
