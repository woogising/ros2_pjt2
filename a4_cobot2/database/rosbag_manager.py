# ============================================================
# database/rosbag_manager.py
# 역할:
#   - DBNode가 작업 시작/종료에 맞춰 실제 `ros2 bag record` 프로세스를
#     시작하고 정상 종료합니다.
#   - 기록이 정상 종료되고 metadata.yaml이 확인된 경우에만 성공으로 처리합니다.
# 주의:
#   - 이 파일은 독립 ROS2 노드가 아니라 DBNode가 사용하는 관리 클래스입니다.
#   - launch를 실행한 셸의 ROS_DOMAIN_ID/RMW 환경을 그대로 상속합니다.
# ============================================================
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, List, Optional


@dataclass
class RosbagStartResult:
    success: bool
    bag_path: Optional[str]
    started_at: Optional[str]
    message: str


@dataclass
class RosbagStopResult:
    success: bool
    bag_path: Optional[str]
    started_at: Optional[str]
    ended_at: Optional[str]
    metadata_path: Optional[str]
    message: str


class RosbagManager:
    """작업별 rosbag record subprocess를 하나만 관리합니다."""

    def __init__(
        self,
        base_dir: str,
        topics: List[str],
        storage_id: str = "sqlite3",
        startup_wait_sec: float = 0.7,
        stop_timeout_sec: float = 12.0,
        flush_delay_sec: float = 0.25,
        logger=None,
    ):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.topics = [str(topic).strip() for topic in topics if str(topic).strip()]
        self.storage_id = str(storage_id).strip()
        self.startup_wait_sec = max(0.0, float(startup_wait_sec))
        self.stop_timeout_sec = max(1.0, float(stop_timeout_sec))
        self.flush_delay_sec = max(0.0, float(flush_delay_sec))
        self.logger = logger

        self.process: Optional[subprocess.Popen] = None
        self.output_dir: Optional[Path] = None
        self.started_at: Optional[str] = None
        self.log_path: Optional[Path] = None
        self._log_file: Optional[IO[bytes]] = None

    def _now_iso(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _info(self, message: str):
        if self.logger is not None:
            self.logger.info(message)

    def _warn(self, message: str):
        if self.logger is not None:
            self.logger.warn(message)

    def _error(self, message: str):
        if self.logger is not None:
            self.logger.error(message)

    def is_recording(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _safe_name(self, value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value)
        return safe or datetime.now().strftime("bag_%Y%m%d_%H%M%S")

    def _make_unique_output_dir(self, run_id: str) -> Path:
        base_name = self._safe_name(run_id)
        candidate = self.base_dir / base_name
        suffix = 2
        while candidate.exists():
            candidate = self.base_dir / f"{base_name}_{suffix}"
            suffix += 1
        return candidate

    def _close_log_file(self):
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
        self._log_file = None

    def _read_log_tail(self, max_bytes: int = 4000) -> str:
        if self.log_path is None or not self.log_path.exists():
            return ""
        try:
            with self.log_path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - max_bytes), os.SEEK_SET)
                return stream.read().decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def _reset_session(self):
        self.process = None
        self.output_dir = None
        self.started_at = None
        self.log_path = None
        self._close_log_file()

    def start(self, run_id: str) -> RosbagStartResult:
        if self.is_recording():
            return RosbagStartResult(
                success=False,
                bag_path=str(self.output_dir) if self.output_dir else None,
                started_at=self.started_at,
                message="another rosbag recording is already running",
            )

        # 이전 프로세스가 비정상 종료된 상태라면 핸들을 먼저 정리합니다.
        if self.process is not None:
            self._close_log_file()
            self.process = None

        if not self.topics:
            return RosbagStartResult(False, None, None, "bag_topics is empty")

        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            logs_dir = self.base_dir / "_manager_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            self.output_dir = self._make_unique_output_dir(run_id)
            self.log_path = logs_dir / f"{self.output_dir.name}.log"
            self._log_file = self.log_path.open("wb")

            command = ["ros2", "bag", "record", "-o", str(self.output_dir)]
            if self.storage_id:
                command.extend(["-s", self.storage_id])
            command.extend(self.topics)

            self._info(f"Starting rosbag: {' '.join(command)}")

            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
            self.started_at = self._now_iso()

            if self.startup_wait_sec > 0:
                time.sleep(self.startup_wait_sec)

            return_code = self.process.poll()
            if return_code is not None:
                tail = self._read_log_tail()
                message = f"ros2 bag record exited during startup (code={return_code})"
                if tail:
                    message += f": {tail}"
                self._error(message)
                bag_path = str(self.output_dir)
                started_at = self.started_at
                self._reset_session()
                return RosbagStartResult(False, bag_path, started_at, message)

            self._info(f"Rosbag recording started: {self.output_dir}")
            return RosbagStartResult(
                success=True,
                bag_path=str(self.output_dir),
                started_at=self.started_at,
                message="rosbag recording started",
            )

        except FileNotFoundError:
            message = "`ros2` command was not found. Source ROS2 and install rosbag2."
            self._error(message)
            self._reset_session()
            return RosbagStartResult(False, None, None, message)
        except Exception as exc:
            message = f"failed to start rosbag recording: {exc}"
            self._error(message)
            self._reset_session()
            return RosbagStartResult(False, None, None, message)

    def stop(self) -> RosbagStopResult:
        if self.process is None:
            return RosbagStopResult(
                success=False,
                bag_path=None,
                started_at=None,
                ended_at=None,
                metadata_path=None,
                message="no rosbag session",
            )

        process = self.process
        output_dir = self.output_dir
        started_at = self.started_at

        try:
            if self.flush_delay_sec > 0 and process.poll() is None:
                # idle/final status가 recorder에 기록될 짧은 시간을 둡니다.
                time.sleep(self.flush_delay_sec)

            if process.poll() is None:
                self._info("Stopping rosbag with SIGINT...")
                os.killpg(process.pid, signal.SIGINT)

            try:
                process.wait(timeout=self.stop_timeout_sec)
            except subprocess.TimeoutExpired:
                self._warn("Rosbag did not stop after SIGINT. Sending SIGTERM.")
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._warn("Rosbag did not stop after SIGTERM. Sending SIGKILL.")
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3.0)

        except ProcessLookupError:
            # 이미 종료된 프로세스는 아래 파일 검증 단계에서 성공 여부를 판단합니다.
            pass
        except Exception as exc:
            self._warn(f"Error while stopping rosbag: {exc}")
        finally:
            ended_at = self._now_iso()
            self._close_log_file()

        metadata_path = output_dir / "metadata.yaml" if output_dir else None
        has_metadata = metadata_path is not None and metadata_path.is_file()

        # 기본 sqlite3 및 다른 storage plugin 결과를 함께 허용합니다.
        storage_files = []
        if output_dir is not None and output_dir.is_dir():
            for pattern in ("*.db3", "*.mcap"):
                storage_files.extend(output_dir.glob(pattern))

        success = bool(has_metadata and storage_files)
        log_tail = self._read_log_tail()

        if success:
            message = "rosbag recording stopped and files verified"
            self._info(f"Rosbag saved: {output_dir}")
        else:
            message = "rosbag files were not verified (metadata.yaml or storage file missing)"
            if log_tail:
                message += f": {log_tail}"
            self._warn(message)

        result = RosbagStopResult(
            success=success,
            bag_path=str(output_dir) if output_dir else None,
            started_at=started_at,
            ended_at=ended_at,
            metadata_path=str(metadata_path) if has_metadata else None,
            message=message,
        )

        self.process = None
        self.output_dir = None
        self.started_at = None
        self.log_path = None
        return result

    def close(self) -> RosbagStopResult:
        return self.stop()
