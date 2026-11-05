"""Loads the robot's brain. Keep all model quirks in this file.

pi0.5 runs in openpi's own Python 3.11 environment as a websocket server
(scripts/serve_policy.py). This file is the client and runs in the LIBERO
Python 3.10 env. The preprocessing copies openpi/examples/libero/main.py
exactly (openpi commit 215abfb):
  * both images rotated 180 degrees, resized with padding to 224
  * state = eef_pos(3) + axis-angle(eef_quat)(3) + gripper_qpos(2) = 8 numbers
  * the model returns a 10-step chunk; we execute the first 5, then replan

Which checkpoint: see CHECKPOINTS below and PROGRESS.md. The default is
pi05_libero (trained on original LIBERO, never on LIBERO-Safety).
"""
import math
import time
import numpy as np

CHECKPOINTS = {
    # trained on obstacle-free original LIBERO; the public analogue of the
    # paper's "obstacle-free SFT" row (Table A.10: SR 18.7%, CR 60.3%)
    "pi05_libero": ("pi05_libero", "gs://openpi-assets/checkpoints/pi05_libero"),
    # fine-tuned on LIBERO-Safety (Table 6: CR ~3%); too safe to study
    "pi05_libero_safety": ("pi05_libero", "~/ckpt_finetuned"),
}


def _quat2axisangle(quat):
    """Copied from robosuite transform_utils, as openpi does."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def make_element(obs, instruction, resize=224):
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize))
    state = np.concatenate((obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]))
    return {"observation/image": img, "observation/wrist_image": wrist,
            "observation/state": state, "prompt": str(instruction)}


def _disable_ws_keepalive():
    """openpi's server runs policy.infer() inside its asyncio loop, so while the
    model JIT-compiles on the first call (30-120 s) it can't answer websocket
    keepalive pings. websockets>=14 sync clients ping every 20 s and drop the
    connection after a 20 s timeout ("1011 keepalive ping timeout"). Turn the
    client's pings off. Results are unaffected; this is transport only."""
    import websockets.sync.client as wsc
    if getattr(wsc.connect, "_vla_no_keepalive", False):
        return
    orig = wsc.connect

    def connect(*a, **k):
        k.setdefault("ping_interval", None)
        return orig(*a, **k)

    connect._vla_no_keepalive = True
    wsc.connect = connect


class Pi05Policy:
    """act(obs) -> (replan_steps, 7) array of actions to execute now.
    After each act(), self.last_chunk holds the full 10-step chunk and
    self.last_element the exact model input, for the monitor's use.
    sample(obs, n) draws n independent chunks for the same observation
    (flow matching resamples noise on every call), which is what a
    consistency monitor needs."""

    def __init__(self, host="127.0.0.1", port=8000, replan_steps=5):
        _disable_ws_keepalive()
        from openpi_client import websocket_client_policy
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.metadata = self.client.get_server_metadata()
        self.replan_steps = replan_steps
        self.instruction = ""
        self.last_chunk = None
        self.last_element = None
        self.infer_seconds = 0.0
        self.calls = 0

    def reset(self, instruction):
        self.instruction = instruction
        try:
            self.client.reset()
        except Exception:
            pass

    def _infer(self, element):
        t = time.perf_counter()
        out = self.client.infer(element)
        self.infer_seconds += time.perf_counter() - t
        self.calls += 1
        a = np.asarray(out["actions"], dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 7 or a.shape[0] < self.replan_steps:
            raise RuntimeError(f"unexpected action chunk shape {a.shape}")
        return a

    def act(self, obs):
        self.last_element = make_element(obs, self.instruction)
        self.last_chunk = self._infer(self.last_element)
        return self.last_chunk[: self.replan_steps]

    def sample(self, obs=None, n=4):
        el = self.last_element if obs is None else make_element(obs, self.instruction)
        return np.stack([self._infer(el) for _ in range(n)])


class DummyPolicy:
    """A stand-in for building the pipeline. It twitches randomly.
    NEVER report numbers from this."""

    def __init__(self, dim=7, horizon=5, seed=0):
        self.rng = np.random.default_rng(seed)
        self.dim, self.replan_steps = dim, horizon
        self.instruction = ""
        self.last_chunk = None

    def reset(self, instruction):
        self.instruction = instruction

    def act(self, obs):
        a = self.rng.normal(0, 0.05, size=(self.replan_steps * 2, self.dim))
        a[:, -1] = -1.0
        self.last_chunk = a
        return a[: self.replan_steps]

    def sample(self, obs=None, n=4):
        return np.stack([self.act(obs) for _ in range(n)])
