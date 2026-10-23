"""Small msgpack WebSocket server compatible with the existing Isaac client."""

import asyncio
import http
import logging
import time
import traceback
from typing import Any, Dict, Optional

import msgpack_numpy
import websockets.asyncio.server as websocket_server
import websockets.frames

msgpack_numpy.patch()
from msgpack_numpy import Packer, unpackb  # noqa: E402


LOGGER = logging.getLogger(__name__)


class WebsocketPolicyServer:
    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 5556,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            LOGGER.info("Policy server listening on ws://%s:%d", self.host, self.port)
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        packer = Packer()
        await websocket.send(packer.pack(self.metadata))
        previous_total_time = None

        while True:
            try:
                start_time = time.monotonic()
                observation = unpackb(await websocket.recv())
                infer_start = time.monotonic()
                response = self.policy.infer(observation)
                infer_ms = (time.monotonic() - infer_start) * 1000
                response["server_timing"] = {"infer_ms": infer_ms}
                if previous_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total_time * 1000
                await websocket.send(packer.pack(response))
                previous_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Policy inference failed",
                )
                raise


def _health_check(
    connection: websocket_server.ServerConnection,
    request: websocket_server.Request,
    ) -> Optional[websocket_server.Response]:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
