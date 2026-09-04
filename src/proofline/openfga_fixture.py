# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Provision the checked-in synthetic policy in a local OpenFGA server."""

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import yaml
from openfga_sdk import ClientConfiguration, OpenFgaClient

from proofline.authorization import OpenFgaAuthorizationAdapter


@dataclass(frozen=True, slots=True)
class ProvisionedOpenFga:
    """The adapter and store created for one isolated fixture run."""

    adapter: OpenFgaAuthorizationAdapter
    client: OpenFgaClient
    server_url: str
    store_id: str

    async def delete(self) -> None:
        """Close the SDK session and remove the temporary store."""

        try:
            await self.client.close()
        finally:
            _request(self.server_url, "DELETE", f"/stores/{self.store_id}")


async def provision_openfga(
    server_url: str, data_root: Path = Path("data/openfga")
) -> ProvisionedOpenFga:
    """Create a store, install the canonical model, and write synthetic tuples.

    The OpenFGA SDK owns an aiohttp client, so construct it while an event loop
    is running. The setup HTTP requests remain synchronous because provisioning
    only runs for this local demo and its integration tests.
    """

    store = _request(server_url, "POST", "/stores", {"name": "proofline-phase-1-5"})
    store_id = _required_string(store, "id")
    model = _request(
        server_url,
        "POST",
        f"/stores/{store_id}/authorization-models",
        json.loads((data_root / "model.json").read_text()),
    )
    model_id = _required_string(model, "authorization_model_id")
    tuples = yaml.safe_load((data_root / "tuples.yaml").read_text())["tuples"]
    _request(
        server_url,
        "POST",
        f"/stores/{store_id}/write",
        {"authorization_model_id": model_id, "writes": {"tuple_keys": tuples}},
    )
    client = OpenFgaClient(
        ClientConfiguration(
            api_url=server_url,
            store_id=store_id,
            authorization_model_id=model_id,
        )
    )
    return ProvisionedOpenFga(OpenFgaAuthorizationAdapter(client), client, server_url, store_id)


def _request(
    server_url: str, method: str, path: str, payload: object | None = None
) -> dict[str, object]:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        f"{server_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:  # noqa: S310 - caller chooses the local server
        body = response.read()
    return {} if not body else json.loads(body)


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"OpenFGA response did not include string {key!r}")
    return value
