"""Cross-platform mic + system-audio capture."""

from __future__ import annotations

import platform
import sys
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sounddevice as sd

CHUNK = 1024
DTYPE = "int16"


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    channels: int
    sample_rate: float
    is_input: bool
    hostapi: str = ""
    is_loopback: bool = False

    def label(self) -> str:
        kind = "in" if self.is_input else "out"
        loop = " [loopback]" if self.is_loopback else ""
        return (
            f"[{self.index}] {self.name} "
            f"({kind}, {self.channels}ch, "
            f"{int(self.sample_rate)} Hz{loop})"
        )


def _hostapi_name(index: int) -> str:
    try:
        return str(sd.query_hostapis(index)["name"])
    except Exception:
        return ""


def list_sounddevice_devices() -> list[AudioDevice]:
    devices: list[AudioDevice] = []
    raw = sd.query_devices()

    for index, info in enumerate(raw):
        max_in = int(info["max_input_channels"])
        max_out = int(info["max_output_channels"])
        name = str(info["name"])
        rate = float(info["default_samplerate"])
        hostapi = _hostapi_name(int(info["hostapi"]))

        if max_in > 0:
            devices.append(
                AudioDevice(
                    index=index,
                    name=name,
                    channels=max_in,
                    sample_rate=rate,
                    is_input=True,
                    hostapi=hostapi,
                    is_loopback="loopback" in name.lower(),
                )
            )

        if max_out > 0 and max_in == 0:
            devices.append(
                AudioDevice(
                    index=index,
                    name=name,
                    channels=max_out,
                    sample_rate=rate,
                    is_input=False,
                    hostapi=hostapi,
                )
            )

    return devices


def list_windows_loopback_devices() -> list[AudioDevice]:
    """WASAPI loopback via PyAudioWPatch (Windows only)."""
    if sys.platform != "win32":
        return []

    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "pyaudiowpatch is required on Windows for system loopback. "
            "Install with: uv sync --extra win"
        ) from exc

    audio = pyaudio.PyAudio()
    devices: list[AudioDevice] = []

    try:
        wasapi = audio.get_host_api_info_by_type(
            pyaudio.paWASAPI
        )
        count = int(wasapi.get("deviceCount", 0))
        host_index = int(wasapi["index"])

        for i in range(count):
            info = audio.get_device_info_by_host_api_device_index(
                host_index,
                i,
            )
            if not info.get("isLoopbackDevice"):
                continue

            devices.append(
                AudioDevice(
                    index=int(info["index"]),
                    name=str(info["name"]),
                    channels=int(info["maxInputChannels"]) or 2,
                    sample_rate=float(info["defaultSampleRate"]),
                    is_input=True,
                    hostapi="Windows WASAPI",
                    is_loopback=True,
                )
            )
    finally:
        audio.terminate()

    return devices


def list_input_devices() -> list[AudioDevice]:
    devices = [
        d for d in list_sounddevice_devices() if d.is_input
    ]

    if sys.platform == "win32":
        seen = {d.index for d in devices}
        for loopback in list_windows_loopback_devices():
            if loopback.index not in seen:
                devices.append(loopback)

    return devices


def print_devices(devices: list[AudioDevice] | None = None) -> None:
    if devices is None:
        devices = list_input_devices()

    print()
    print("Input devices:")
    print("-" * 60)

    if not devices:
        print("(none found)")
        return

    for device in devices:
        print(device.label())


def _match_name(device: AudioDevice, needle: str) -> bool:
    return needle.lower() in device.name.lower()


def resolve_device(
    selector: str | int | None,
    *,
    role: str,
    candidates: list[AudioDevice],
) -> AudioDevice:
    if not candidates:
        raise RuntimeError(f"No input devices available for {role}")

    if selector is None:
        raise RuntimeError(
            f"No {role} device selected. "
            "Pass --mic/--system or use auto-detect."
        )

    if isinstance(selector, int) or (
        isinstance(selector, str) and selector.isdigit()
    ):
        index = int(selector)
        for device in candidates:
            if device.index == index:
                return device
        raise RuntimeError(
            f"{role} device index {index} not found. "
            "Use --list-devices."
        )

    matches = [
        d for d in candidates if _match_name(d, str(selector))
    ]

    if not matches:
        raise RuntimeError(
            f"No {role} device matching {selector!r}. "
            "Use --list-devices."
        )

    if len(matches) > 1:
        names = ", ".join(d.label() for d in matches)
        raise RuntimeError(
            f"Ambiguous {role} device {selector!r}: {names}"
        )

    return matches[0]


def auto_mic_device(
    devices: list[AudioDevice] | None = None,
) -> AudioDevice:
    if devices is None:
        devices = list_input_devices()

    non_virtual = [
        d
        for d in devices
        if d.is_input
        and not d.is_loopback
        and "blackhole" not in d.name.lower()
        and "soundflower" not in d.name.lower()
        and "cable" not in d.name.lower()
    ]

    if non_virtual:
        default = sd.default.device[0]
        if isinstance(default, int):
            for device in non_virtual:
                if device.index == default:
                    return device
        return non_virtual[0]

    raise RuntimeError(
        "Could not auto-detect microphone. "
        "Pass --mic INDEX_OR_NAME."
    )


def auto_system_device(
    devices: list[AudioDevice] | None = None,
) -> AudioDevice:
    if devices is None:
        devices = list_input_devices()

    system = platform.system()

    if system == "Darwin":
        for needle in ("BlackHole", "Soundflower", "Loopback"):
            matches = [
                d for d in devices if _match_name(d, needle)
            ]
            if matches:
                return matches[0]

        raise RuntimeError(
            "No virtual system-audio device found (expected BlackHole). "
            "Install BlackHole 2ch, create a Multi-Output Device, "
            "then re-run --list-devices or pass --system BlackHole."
        )

    if system == "Windows":
        loopbacks = [d for d in devices if d.is_loopback]
        if loopbacks:
            return loopbacks[0]

        raise RuntimeError(
            "No WASAPI loopback device found. "
            "Install win extras: uv sync --extra win "
            "and ensure a playback device exists."
        )

    raise RuntimeError(
        f"System-audio capture is not auto-configured on {system}. "
        "Pass --system INDEX_OR_NAME."
    )


class SoundDeviceRecorder:
    """Record one input device to a WAV file (streamed to disk)."""

    def __init__(
        self,
        device: AudioDevice,
        output_file: Path,
        name: str,
        *,
        channels: int | None = None,
        sample_rate: int | None = None,
    ):
        self.device = device
        self.output_file = Path(output_file)
        self.name = name
        self.channels = channels or min(device.channels, 2)
        self.sample_rate = int(
            sample_rate or device.sample_rate
        )

        self._stream: sd.RawInputStream | None = None
        self._wave: Any = None
        self._lock = threading.Lock()
        self.running = False

    def start(self) -> None:
        print()
        print(f"Starting {self.name}")
        print(f"  device      = {self.device.index}")
        print(f"  name        = {self.device.name}")
        print(f"  channels    = {self.channels}")
        print(f"  sample rate = {self.sample_rate}")
        print(f"  output      = {self.output_file}")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._wave = wave.open(str(self.output_file), "wb")
        self._wave.setnchannels(self.channels)
        self._wave.setsampwidth(2)
        self._wave.setframerate(self.sample_rate)

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=DTYPE,
            device=self.device.index,
            blocksize=CHUNK,
            callback=self._callback,
        )
        self.running = True
        self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[{self.name}] status: {status}")

        if not self.running or self._wave is None:
            return

        with self._lock:
            self._wave.writeframes(bytes(indata))

    def stop(self) -> None:
        if not self.running:
            return

        self.running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if self._wave is not None:
                self._wave.close()
                self._wave = None

        print(f"{self.name} saved: {self.output_file}")


class PyAudioWPatchRecorder:
    """Windows loopback/mic recorder using PyAudioWPatch."""

    def __init__(
        self,
        device: AudioDevice,
        output_file: Path,
        name: str,
        *,
        channels: int | None = None,
        sample_rate: int | None = None,
    ):
        import pyaudiowpatch as pyaudio

        self._pyaudio_mod = pyaudio
        self.device = device
        self.output_file = Path(output_file)
        self.name = name
        self.channels = channels or min(device.channels, 2)
        self.sample_rate = int(
            sample_rate or device.sample_rate
        )

        self._audio = None
        self._stream = None
        self._wave = None
        self._thread: threading.Thread | None = None
        self.running = False

    def start(self) -> None:
        pyaudio = self._pyaudio_mod

        print()
        print(f"Starting {self.name}")
        print(f"  device      = {self.device.index}")
        print(f"  name        = {self.device.name}")
        print(f"  channels    = {self.channels}")
        print(f"  sample rate = {self.sample_rate}")
        print(f"  output      = {self.output_file}")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._audio = pyaudio.PyAudio()
        self._wave = wave.open(str(self.output_file), "wb")
        self._wave.setnchannels(self.channels)
        self._wave.setsampwidth(
            self._audio.get_sample_size(pyaudio.paInt16)
        )
        self._wave.setframerate(self.sample_rate)

        self._stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device.index,
            frames_per_buffer=CHUNK,
        )

        self.running = True
        self._thread = threading.Thread(
            target=self._record,
            daemon=True,
        )
        self._thread.start()

    def _record(self) -> None:
        while self.running:
            try:
                data = self._stream.read(
                    CHUNK,
                    exception_on_overflow=False,
                )
                self._wave.writeframes(data)
            except Exception as exc:
                print(f"[{self.name}] ERROR: {exc}")
                break

    def stop(self) -> None:
        if not self.running:
            return

        self.running = False

        if self._thread is not None:
            self._thread.join()
            self._thread = None

        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._wave is not None:
            self._wave.close()
            self._wave = None

        if self._audio is not None:
            self._audio.terminate()
            self._audio = None

        print(f"{self.name} saved: {self.output_file}")


def create_recorder(
    device: AudioDevice,
    output_file: Path,
    name: str,
) -> SoundDeviceRecorder | PyAudioWPatchRecorder:
    if device.is_loopback and sys.platform == "win32":
        return PyAudioWPatchRecorder(
            device=device,
            output_file=output_file,
            name=name,
        )

    # Prefer sounddevice for normal inputs. On Windows, loopback-only
    # devices from PyAudioWPatch use the other recorder.
    try:
        return SoundDeviceRecorder(
            device=device,
            output_file=output_file,
            name=name,
        )
    except Exception:
        if sys.platform == "win32":
            return PyAudioWPatchRecorder(
                device=device,
                output_file=output_file,
                name=name,
            )
        raise

