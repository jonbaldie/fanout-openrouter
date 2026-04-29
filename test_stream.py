import asyncio
import httpx
import time


async def main():
    start = time.monotonic()
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://127.0.0.1:8000/api/v1/chat/completions",
            json={
                "model": "fanout/minimal",
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a 200 word essay on the Roman Empire.",
                    }
                ],
                "stream": True,
                "max_tokens": 400,
            },
            headers={
                "Authorization": "Bearer REDACTED_ROTATED_KEY"
            },
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    print(f"{(time.monotonic() - start):.3f}s: {line[:50]}...")


asyncio.run(main())
