#!/usr/bin/env python3
import subprocess, asyncio, websockets

PROC = subprocess.Popen(
    ["./target/release/uci_binary"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE, text=True, bufsize=1
)

CURRENT_WS = None  # track the one active client

async def broadcaster():
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, PROC.stdout.readline)
        if line and CURRENT_WS is not None:
            try:
                await CURRENT_WS.send(line.strip())
            except Exception as e:
                import sys
                sys.stderr.write(f"WS-SEND-ERR: {e}\n")

async def handler(ws, path):
    global CURRENT_WS
    CURRENT_WS = ws
    async for msg in ws:
        PROC.stdin.write(msg + "\n")
        PROC.stdin.flush()
    if CURRENT_WS is ws:
        CURRENT_WS = None

async def main():
    PROC.stdin.write("uci\n")   # sent exactly once, at server startup
    PROC.stdin.flush()
    asyncio.create_task(broadcaster())  # exactly one reader, ever
    async with websockets.serve(handler, "", 8765):
        await asyncio.Future()

asyncio.run(main())
