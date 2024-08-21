import asyncio
import argparse
import simpleobsws

# OBS WebSocket connection details
host = "192.168.112.188"
port = 4455
password = "@Umasenha@2016"  # Replace with your actual password

async def connect_obs():
    ws = simpleobsws.WebSocketClient(
        f"ws://{host}:{port}", password=password
    )
    try:
        await ws.connect()
        await ws.wait_until_identified()
        return ws
    except Exception as e:
        print(f"Error connecting to OBS: {e}")
        print("Please make sure OBS is running and the WebSocket server is enabled.")
        print(f"Check if the host ({host}), port ({port}), and password are correct.")
        return None

async def start_recording(ws):
    request = simpleobsws.Request('StartRecord')
    response = await ws.call(request)
    if response.ok():
        print("Recording started successfully")
    else:
        print(f"Failed to start recording: {response.error()}")

async def stop_recording(ws):
    request = simpleobsws.Request('StopRecord')
    response = await ws.call(request)
    if response.ok():
        print("Recording stopped successfully")
    else:
        print(f"Failed to stop recording: {response.error()}")

async def main():
    parser = argparse.ArgumentParser(description="Control OBS recording")
    parser.add_argument("action", choices=["start", "stop"], help="Action to perform (start or stop recording)")
    args = parser.parse_args()

    ws = await connect_obs()
    if ws is None:
        return  # Exit if connection failed

    try:
        if args.action == "start":
            await start_recording(ws)
        elif args.action == "stop":
            await stop_recording(ws)
    finally:
        if ws:
            await ws.disconnect()

if __name__ == "__main__":
    asyncio.run(main())