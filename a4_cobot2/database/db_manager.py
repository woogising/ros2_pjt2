# ============================================================
# database/db_manager.py
# 역할:
#   - SQLite DB 연결, 테이블 생성, INSERT/UPDATE/SELECT 함수를 담당합니다.
#   - db_node.py는 ROS2 topic을 받아 이 클래스의 함수만 호출합니다.
# 저장 범위:
#   - 기존 topic으로 외부에 이미 나오는 데이터만 저장합니다.
#   - service/action 내부 payload는 새 topic 없이 DB에서 볼 수 없으므로 저장하지 않습니다.
# ============================================================
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


class DBManager:
    # SQLite DB 파일을 열고 필요한 테이블을 준비하는 함수
    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # ROS2 callback 중 동시에 DB 접근이 생길 가능성을 고려해서 lock을 둡니다.
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        self._configure_database()
        self.create_tables()

    # SQLite 동작 옵션을 설정하는 함수
    def _configure_database(self):
        with self._lock:
            self.conn.execute('PRAGMA foreign_keys = ON;')
            self.conn.execute('PRAGMA journal_mode = WAL;')
            self.conn.execute('PRAGMA synchronous = NORMAL;')
            self.conn.commit()

    # 현재 프로젝트에서 사용할 DB 테이블들을 생성하는 함수
    def create_tables(self):
        schema_sql = """
        CREATE TABLE IF NOT EXISTS task_runs (
            run_id TEXT PRIMARY KEY,
            command TEXT,
            raw_text TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            final_status TEXT,
            success INTEGER DEFAULT 0,
            bag_path TEXT,
            memo TEXT
        );

        CREATE TABLE IF NOT EXISTS commands (
            command_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            timestamp TEXT NOT NULL,
            raw_text TEXT,
            parsed_command TEXT,
            FOREIGN KEY(run_id) REFERENCES task_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS task_status_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES task_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS user_notices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            timestamp TEXT NOT NULL,
            notice TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES task_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS safety_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            value TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES task_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS bag_records (
            bag_id TEXT PRIMARY KEY,
            run_id TEXT,
            bag_path TEXT NOT NULL,
            started_at TEXT,
            ended_at TEXT,
            topics TEXT,
            FOREIGN KEY(run_id) REFERENCES task_runs(run_id)
        );
        """

        with self._lock:
            self.conn.executescript(schema_sql)
            self.conn.commit()

    # 작업 1회 실행 기록을 생성하는 함수
    def create_task_run(
        self,
        run_id: str,
        command: str,
        raw_text: Optional[str],
        started_at: str,
        bag_path: Optional[str] = None,
        memo: Optional[str] = None,
    ):
        with self._lock:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO task_runs (
                    run_id, command, raw_text, started_at, bag_path, memo
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, command, raw_text, started_at, bag_path, memo),
            )
            self.conn.commit()

    # 작업 기록의 raw_text를 나중에 보정하는 함수
    def update_task_raw_text(self, run_id: str, raw_text: str):
        with self._lock:
            self.conn.execute(
                """
                UPDATE task_runs
                SET raw_text = COALESCE(raw_text, ?)
                WHERE run_id = ?
                """,
                (raw_text, run_id),
            )
            self.conn.commit()

    # 실제 rosbag 파일이 정상 생성된 뒤 task_runs에 경로를 연결하는 함수
    def update_task_bag_path(self, run_id: str, bag_path: Optional[str]):
        with self._lock:
            self.conn.execute(
                """
                UPDATE task_runs
                SET bag_path = ?
                WHERE run_id = ?
                """,
                (bag_path, run_id),
            )
            self.conn.commit()

    # 작업의 마지막 상태를 갱신하는 함수
    def update_task_status(self, run_id: str, final_status: str):
        with self._lock:
            self.conn.execute(
                """
                UPDATE task_runs
                SET final_status = ?
                WHERE run_id = ?
                """,
                (final_status, run_id),
            )
            self.conn.commit()

    # 작업 종료 시간과 성공 여부를 기록하는 함수
    def finish_task_run(
        self,
        run_id: str,
        ended_at: str,
        final_status: str,
        success: bool,
        memo: Optional[str] = None,
    ):
        with self._lock:
            self.conn.execute(
                """
                UPDATE task_runs
                SET ended_at = ?, final_status = ?, success = ?, memo = COALESCE(?, memo)
                WHERE run_id = ?
                """,
                (ended_at, final_status, int(success), memo, run_id),
            )
            self.conn.commit()

    # 명령 원문과 분류된 내부 명령어를 저장하는 함수
    def insert_command(
        self,
        run_id: Optional[str],
        timestamp: str,
        raw_text: Optional[str],
        parsed_command: str,
    ):
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO commands (run_id, timestamp, raw_text, parsed_command)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, timestamp, raw_text, parsed_command),
            )
            self.conn.commit()

    # /task_status 이벤트를 저장하는 함수
    def insert_task_status(
        self,
        run_id: Optional[str],
        timestamp: str,
        status: str,
    ):
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO task_status_events (run_id, timestamp, status)
                VALUES (?, ?, ?)
                """,
                (run_id, timestamp, status),
            )
            self.conn.commit()

    # /user_notice 안내 문장을 저장하는 함수
    def insert_user_notice(
        self,
        run_id: Optional[str],
        timestamp: str,
        notice: str,
    ):
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO user_notices (run_id, timestamp, notice)
                VALUES (?, ?, ?)
                """,
                (run_id, timestamp, notice),
            )
            self.conn.commit()

    # safety 관련 topic 이벤트를 저장하는 함수
    def insert_safety_event(
        self,
        run_id: Optional[str],
        timestamp: str,
        event_type: str,
        value: str,
    ):
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO safety_events (run_id, timestamp, event_type, value)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, timestamp, event_type, value),
            )
            self.conn.commit()

    # rosbag 기록 정보를 생성하거나 갱신하는 함수
    def upsert_bag_record(
        self,
        bag_id: str,
        run_id: Optional[str],
        bag_path: str,
        started_at: Optional[str],
        ended_at: Optional[str],
        topics: List[str],
    ):
        topics_json = json.dumps(topics, ensure_ascii=False)

        with self._lock:
            self.conn.execute(
                """
                INSERT INTO bag_records (bag_id, run_id, bag_path, started_at, ended_at, topics)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(bag_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    bag_path = excluded.bag_path,
                    started_at = COALESCE(bag_records.started_at, excluded.started_at),
                    ended_at = excluded.ended_at,
                    topics = excluded.topics
                """,
                (bag_id, run_id, bag_path, started_at, ended_at, topics_json),
            )
            self.conn.commit()

    # SQLite Row를 일반 dict로 바꾸는 함수
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row)

    # 최근 작업 목록을 UI에서 쓰기 좋은 형태로 반환하는 함수
    def get_recent_task_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT run_id, command, raw_text, started_at, ended_at,
                       final_status, success, bag_path, memo
                FROM task_runs
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    # 특정 작업의 기본 정보를 반환하는 함수
    def get_task_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT run_id, command, raw_text, started_at, ended_at,
                       final_status, success, bag_path, memo
                FROM task_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    # 특정 작업의 상태 타임라인을 반환하는 함수
    def get_status_timeline(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT timestamp, status
                FROM task_status_events
                WHERE run_id = ?
                ORDER BY timestamp ASC
                """,
                (run_id,),
            ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    # 특정 작업의 사용자 안내 문장 목록을 반환하는 함수
    def get_user_notices(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT timestamp, notice
                FROM user_notices
                WHERE run_id = ?
                ORDER BY timestamp ASC
                """,
                (run_id,),
            ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    # 특정 작업의 safety 이벤트 목록을 반환하는 함수
    def get_safety_events(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT timestamp, event_type, value
                FROM safety_events
                WHERE run_id = ?
                ORDER BY timestamp ASC
                """,
                (run_id,),
            ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    # 특정 작업의 rosbag 기록을 반환하는 함수
    def get_bag_records(self, run_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT bag_id, run_id, bag_path, started_at, ended_at, topics
                FROM bag_records
                WHERE run_id = ?
                ORDER BY started_at ASC
                """,
                (run_id,),
            ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    # 특정 작업 상세 화면에 필요한 데이터를 한 번에 반환하는 함수
    def get_task_run_detail(self, run_id: str) -> Optional[Dict[str, Any]]:
        run = self.get_task_run(run_id)

        if run is None:
            return None

        return {
            'run': run,
            'status_timeline': self.get_status_timeline(run_id),
            'user_notices': self.get_user_notices(run_id),
            'safety_events': self.get_safety_events(run_id),
            'bag_records': self.get_bag_records(run_id),
        }

    # DB 연결을 닫는 함수
    def close(self):
        with self._lock:
            self.conn.commit()
            self.conn.close()
