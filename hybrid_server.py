#!/usr/bin/env python3
import asyncio, websockets, subprocess, sys, time

LOG = open("/tmp/hybrid_io.log", "w", buffering=1)

def log_stdin(s):
    LOG.write(f"[{time.monotonic():.6f}] STDIN: {s!r}\n")

def log_stdout(s):
    LOG.write(f"[{time.monotonic():.6f}] STDOUT: {s!r}\n")

async def handler(ws, path):
    proc = subprocess.Popen(
        ["./target/release/uci_binary"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1
    )

    def stderr_reader():
        for err_line in iter(proc.stderr.readline, ''):
            LOG.write(f"[DIAG-STDERR] {err_line!r}\n")

    import threading
    threading.Thread(target=stderr_reader, daemon=True).start()

    loop = asyncio.get_event_loop()

    async def reader():
        while True:
            line = await loop.run_in_executor(None, proc.stdout.readline)
            LOG.write(f"[DIAG] readline returned: line={line!r} poll={proc.poll()}\n")
            if line:
                log_stdout(line)
                try:
                    await ws.send(line.strip())
                except Exception as e:
                    import websockets
                    exc_name = type(e).__name__
                    LOG.write(f"[DIAG-CONN-ERROR] {exc_name}: {e!r} poll={proc.poll()} stderr_readable={not proc.stderr.closed}\n")
                    sys.stderr.write(f"WS-SEND-ERR: {e}\n")

    asyncio.create_task(reader())

    async for msg in ws:
        log_stdin(msg)
        proc.stdin.write(msg + "\n")
        proc.stdin.flush()

start_server = websockets.serve(handler, "", 8765, ping_interval=None)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
