# ============================================================
# hmi/hmi_interface.py
# 역할:
#   - PyQt 기반 HMI 화면입니다.
#   - 기존 a4_cobot2 노드 구조와 호환되도록 /task_command,
#     /task_status, /user_notice, /safety_state와 연결됩니다.
#
# 현재 연결:
#   SCAN WORKSPACE  -> /task_command: check_workspace
#   START SORTING   -> /task_command: start_organize
#   RECHECK         -> /task_command: check_workspace
#   EMERGENCY STOP  -> /task_command: stop
#
# 현재 UI 표시 물건:
#   A Zone: 망치, 드라이버
#   B Zone: 볼트, 테이프
#   C Zone: 사과, 파인애플
#   D Zone: 포카리, 게토레이
#
# YOLO 클래스명 매핑:
#   hammer, screwdriver, bolt, tape,
#   green_apple, pineapple, pocari, gatorade
#
# 아직 연결 보류:
#   RESET, REPLAY VOICE, VOICE OFF
# ============================================================

import sys

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QFrame,
    QSizePolicy,
    QHeaderView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from hmi.hmi_ros_bridge import HmiRosBridge

try:
    from task_manager import status_codes as Status
except Exception:
    class Status:
        COMMAND_CHECK_WORKSPACE = "check_workspace"
        COMMAND_START_ORGANIZE = "start_organize"
        COMMAND_STOP = "stop"
        COMMAND_SHUTDOWN = "shutdown"


# ============================================================
# YOLO class name -> HMI display mapping
# ============================================================
ITEM_CLASS_MAP = {
    "hammer": {"display_name": "망치", "zone": "A", "category": "Tools", "quantity": 1},
    "screwdriver": {"display_name": "드라이버", "zone": "A", "category": "Tools", "quantity": 1},
    "bolt": {"display_name": "볼트", "zone": "B", "category": "Parts", "quantity": 1},
    "tape": {"display_name": "테이프", "zone": "B", "category": "Parts", "quantity": 1},
    "green_apple": {"display_name": "사과", "zone": "C", "category": "Fruits", "quantity": 1},
    "pineapple": {"display_name": "파인애플", "zone": "C", "category": "Fruits", "quantity": 1},
    "pocari": {"display_name": "포카리", "zone": "D", "category": "Drinks", "quantity": 1},
    "gatorade": {"display_name": "게토레이", "zone": "D", "category": "Drinks", "quantity": 1},
}

ZONE_INFO = {
    "A": {"title": "A Zone", "rule": "Rule: Tools", "class_names": ["hammer", "screwdriver"]},
    "B": {"title": "B Zone", "rule": "Rule: Parts", "class_names": ["bolt", "tape"]},
    "C": {"title": "C Zone", "rule": "Rule: Fruits", "class_names": ["green_apple", "pineapple"]},
    "D": {"title": "D Zone", "rule": "Rule: Drinks", "class_names": ["pocari", "gatorade"]},
}

ITEM_ORDER = [
    "hammer",
    "screwdriver",
    "bolt",
    "tape",
    "green_apple",
    "pineapple",
    "pocari",
    "gatorade",
]


def get_item_display_name(class_name):
    item = ITEM_CLASS_MAP.get(class_name)
    if item is None:
        return class_name
    return item["display_name"]


def get_item_target_zone(class_name):
    item = ITEM_CLASS_MAP.get(class_name)
    if item is None:
        return "-"
    return item["zone"]


class StatusCard(QFrame):
    def __init__(self, title, status, color):
        super().__init__()
        self.setObjectName("StatusCard")

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("StatusTitle")

        self.status_label = QLabel(f"● {status}")
        self.status_label.setObjectName("StatusValue")
        self.status_label.setStyleSheet(f"color: {color};")

        layout.addWidget(title_label)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def set_status(self, status, color="#38BDF8"):
        self.status_label.setText(f"● {status}")
        self.status_label.setStyleSheet(f"color: {color};")


class ZoneBox(QFrame):
    def __init__(self, zone_name, rule, objects, state="normal"):
        super().__init__()

        if state == "warning":
            self.setObjectName("ZoneBoxWarning")
        else:
            self.setObjectName("ZoneBox")

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        title = QLabel(zone_name)
        title.setObjectName("ZoneTitle")

        rule_label = QLabel(rule)
        rule_label.setObjectName("ZoneRule")

        object_label = QLabel(objects)
        object_label.setObjectName("ZoneObjects")
        object_label.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(rule_label)
        layout.addStretch()
        layout.addWidget(object_label)

        self.setLayout(layout)


class SortingRobotHMI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("A4 Cobot2 HMI Interface")
        self.resize(1450, 900)

        self.ros_bridge = None
        self.status_cards = {}
        self.last_notice = (
            "HMI가 시작되었습니다.\n"
            "작업공간 확인을 시작하려면 SCAN WORKSPACE 버튼을 누르세요."
        )
        self.voice_enabled = True

        self.init_ui()
        self.init_ros_bridge()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        # =========================================================
        # Header
        # =========================================================
        header_layout = QHBoxLayout()

        title_box = QVBoxLayout()
        title_box.setSpacing(4)

        title = QLabel("A4 Cobot2 HMI Interface")
        title.setObjectName("MainTitle")

        subtitle = QLabel(
            "Voice-based Workspace Organization | Task Manager + Object Detection + Robot Arm + Safety"
        )
        subtitle.setObjectName("SubTitle")

        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header_layout.addLayout(title_box)
        header_layout.addStretch()

        self.header_status = QLabel("● HMI STARTING")
        self.header_status.setObjectName("HeaderStatus")
        header_layout.addWidget(self.header_status)

        main_layout.addLayout(header_layout)

        # =========================================================
        # Status Cards
        # =========================================================
        status_layout = QHBoxLayout()
        status_layout.setSpacing(12)

        self.status_cards["robot"] = StatusCard("ROBOT", "STANDBY", "#22C55E")
        self.status_cards["camera"] = StatusCard("CAMERA", "WAITING", "#FACC15")
        self.status_cards["yolo"] = StatusCard("YOLO", "WAITING", "#FACC15")
        self.status_cards["voice"] = StatusCard("VOICE", "READY", "#38BDF8")
        self.status_cards["task"] = StatusCard("TASK", "READY", "#A78BFA")
        self.status_cards["safety"] = StatusCard("SAFETY", "UNKNOWN", "#FACC15")

        for card in self.status_cards.values():
            status_layout.addWidget(card)

        main_layout.addLayout(status_layout)

        # =========================================================
        # Center Area: Camera + Zone Map
        # =========================================================
        center_layout = QHBoxLayout()
        center_layout.setSpacing(14)

        # Camera Panel
        camera_panel = QFrame()
        camera_panel.setObjectName("Panel")

        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(16, 16, 16, 16)
        camera_layout.setSpacing(12)

        camera_title = QLabel("Camera / YOLO View")
        camera_title.setObjectName("PanelTitle")

        self.camera_view = QLabel()
        self.camera_view.setObjectName("CameraView")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setText(
            "CAMERA VIEW\n\n"
            "현재는 UI 노드 연동 1차 버전입니다.\n\n"
            "추후 연결 예정:\n"
            "/camera/camera/color/image_raw\n"
            "또는 YOLO result image topic\n\n"
            "현재 HMI 연결:\n"
            "/task_command, /task_status, /user_notice, /safety_state"
        )
        self.camera_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        camera_layout.addWidget(camera_title)
        camera_layout.addWidget(self.camera_view)

        camera_panel.setLayout(camera_layout)
        center_layout.addWidget(camera_panel, 6)

        # Zone Panel
        zone_panel = QFrame()
        zone_panel.setObjectName("Panel")

        zone_layout = QVBoxLayout()
        zone_layout.setContentsMargins(16, 16, 16, 16)
        zone_layout.setSpacing(12)

        zone_title = QLabel("Workspace Zone Map")
        zone_title.setObjectName("PanelTitle")

        zone_grid = QGridLayout()
        zone_grid.setSpacing(12)

        zone_positions = {
            "A": (0, 0),
            "B": (0, 1),
            "C": (1, 0),
            "D": (1, 1),
        }

        for zone_name, position in zone_positions.items():
            zone = ZONE_INFO[zone_name]
            object_names = ", ".join(
                get_item_display_name(class_name)
                for class_name in zone["class_names"]
            )
            zone_grid.addWidget(
                ZoneBox(zone["title"], zone["rule"], object_names, "normal"),
                position[0],
                position[1],
            )

        zone_layout.addWidget(zone_title)
        zone_layout.addLayout(zone_grid)

        zone_panel.setLayout(zone_layout)
        center_layout.addWidget(zone_panel, 4)

        main_layout.addLayout(center_layout, 5)

        # =========================================================
        # Bottom Area: Object Table + TTS
        # =========================================================
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(14)

        # Object Table Panel
        table_panel = QFrame()
        table_panel.setObjectName("Panel")

        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(16, 16, 16, 16)
        table_layout.setSpacing(12)

        table_title = QLabel("Detected Object List")
        table_title.setObjectName("PanelTitle")

        self.table = QTableWidget()
        self.table.setObjectName("ObjectTable")
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Object", "YOLO Class", "Current Zone", "Target Zone", "Quantity", "Status"]
        )

        self.set_default_object_table()

        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        table_layout.addWidget(table_title)
        table_layout.addWidget(self.table)

        table_panel.setLayout(table_layout)
        bottom_layout.addWidget(table_panel, 6)

        # TTS / User Notice Panel
        voice_panel = QFrame()
        voice_panel.setObjectName("Panel")

        voice_layout = QVBoxLayout()
        voice_layout.setContentsMargins(16, 16, 16, 16)
        voice_layout.setSpacing(12)

        voice_title = QLabel("Robot Voice / User Notice")
        voice_title.setObjectName("PanelTitle")

        self.voice_bubble = QLabel(f"🤖  {self.last_notice}")
        self.voice_bubble.setObjectName("VoiceBubble")
        self.voice_bubble.setWordWrap(True)

        button_layout = QHBoxLayout()
        replay_btn = QPushButton("REPLAY VOICE")
        self.mute_btn = QPushButton("VOICE OFF")
        self.mute_btn.setEnabled(False)  # 실제 TTS 제어 미연결 → 비활성화(무시)

        button_layout.addWidget(replay_btn)
        button_layout.addWidget(self.mute_btn)

        voice_layout.addWidget(voice_title)
        voice_layout.addWidget(self.voice_bubble)
        voice_layout.addLayout(button_layout)

        voice_panel.setLayout(voice_layout)
        bottom_layout.addWidget(voice_panel, 4)

        main_layout.addLayout(bottom_layout, 3)

        # =========================================================
        # Control Buttons
        # =========================================================
        control_panel = QFrame()
        control_panel.setObjectName("Panel")

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(16, 14, 16, 14)
        control_layout.setSpacing(12)

        scan_btn = QPushButton("SCAN WORKSPACE")
        sort_btn = QPushButton("START SORTING")
        recheck_btn = QPushButton("RECHECK")
        stop_btn = QPushButton("EMERGENCY STOP")
        reset_btn = QPushButton("RESET")
        reset_btn.setEnabled(False)  # ROS2 reset 명령 미연결 → 비활성화(무시)

        stop_btn.setObjectName("StopButton")

        control_layout.addWidget(scan_btn)
        control_layout.addWidget(sort_btn)
        control_layout.addWidget(recheck_btn)
        control_layout.addWidget(stop_btn)
        control_layout.addWidget(reset_btn)

        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)

        # =========================================================
        # System Log
        # =========================================================
        log_panel = QFrame()
        log_panel.setObjectName("Panel")

        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(16, 12, 16, 16)
        log_layout.setSpacing(10)

        log_title = QLabel("System Log")
        log_title.setObjectName("PanelTitle")

        self.log_box = QTextEdit()
        self.log_box.setObjectName("LogBox")
        self.log_box.setReadOnly(True)
        self.log_box.setText(
            "> HMI initialized\n"
            "> Waiting for ROS bridge\n"
            "> Available commands: check_workspace, start_organize, stop\n"
            "> YOLO classes: hammer, screwdriver, bolt, tape, green_apple, pineapple, pocari, gatorade\n"
        )

        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log_box)

        log_panel.setLayout(log_layout)
        main_layout.addWidget(log_panel, 2)

        self.setLayout(main_layout)

        # Button events
        scan_btn.clicked.connect(self.on_scan_clicked)
        sort_btn.clicked.connect(self.on_sort_clicked)
        recheck_btn.clicked.connect(self.on_recheck_clicked)
        stop_btn.clicked.connect(self.on_stop_clicked)
        reset_btn.clicked.connect(self.on_reset_clicked)
        replay_btn.clicked.connect(self.on_replay_voice_clicked)
        self.mute_btn.clicked.connect(self.on_voice_toggle_clicked)

    def init_ros_bridge(self):
        self.ros_bridge = HmiRosBridge()

        self.ros_bridge.connected_signal.connect(self.on_ros_connected)
        self.ros_bridge.task_status_signal.connect(self.on_task_status_received)
        self.ros_bridge.user_notice_signal.connect(self.on_user_notice_received)
        self.ros_bridge.safety_state_signal.connect(self.on_safety_state_received)
        self.ros_bridge.detection_image_signal.connect(self.on_detection_image_received)
        self.ros_bridge.log_signal.connect(self.add_log)

        self.ros_bridge.start()

    # object_detection_node의 YOLO 인식 이미지를 camera_view에 표시하는 슬롯
    def on_detection_image_received(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        self.camera_view.setPixmap(
            pixmap.scaled(
                self.camera_view.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def add_log(self, text):
        self.log_box.append(f"> {text}")

    def set_voice_notice(self, text):
        self.last_notice = text
        self.voice_bubble.setText(f"🤖  {text}")

    def on_ros_connected(self, connected):
        if connected:
            self.header_status.setText("● ROS CONNECTED")
            self.header_status.setStyleSheet(
                """
                font-size: 18px;
                font-weight: bold;
                color: #22C55E;
                padding: 10px 18px;
                border: 1px solid #22C55E;
                border-radius: 12px;
                background-color: rgba(34, 197, 94, 0.10);
                """
            )
            self.add_log("ROS bridge connected")
        else:
            self.header_status.setText("● ROS ERROR")
            self.add_log("ROS bridge connection failed")

    def publish_task_command(self, command):
        if self.ros_bridge is None:
            self.add_log(f"ROS bridge not ready. Command ignored: {command}")
            return

        self.ros_bridge.publish_command(command)

    def on_scan_clicked(self):
        self.add_log("SCAN WORKSPACE button clicked")
        self.set_voice_notice(
            "작업공간 확인 명령을 전송했습니다.\n"
            "/task_command: check_workspace"
        )
        self.status_cards["task"].set_status("SCANNING", "#FACC15")
        self.publish_task_command(Status.COMMAND_CHECK_WORKSPACE)

    def on_sort_clicked(self):
        self.add_log("START SORTING button clicked")
        self.set_voice_notice(
            "정리 시작 명령을 전송했습니다.\n"
            "/task_command: start_organize"
        )
        self.status_cards["task"].set_status("ORGANIZING", "#FACC15")
        self.publish_task_command(Status.COMMAND_START_ORGANIZE)

    def on_recheck_clicked(self):
        self.add_log("RECHECK button clicked")
        self.set_voice_notice(
            "작업공간 재확인 명령을 전송했습니다.\n"
            "/task_command: check_workspace"
        )
        self.status_cards["task"].set_status("RECHECKING", "#FACC15")
        self.publish_task_command(Status.COMMAND_CHECK_WORKSPACE)

    def on_stop_clicked(self):
        self.add_log("EMERGENCY STOP button clicked")
        self.set_voice_notice(
            "긴급 정지 명령을 전송했습니다.\n"
            "/task_command: stop"
        )
        self.status_cards["task"].set_status("STOP", "#EF4444")
        self.publish_task_command(Status.COMMAND_STOP)

    def on_reset_clicked(self):
        self.add_log("RESET button clicked")
        self.set_voice_notice(
            "HMI 화면 상태를 초기화했습니다.\n"
            "ROS2 reset 명령은 아직 연결하지 않았습니다."
        )
        self.status_cards["robot"].set_status("STANDBY", "#22C55E")
        self.status_cards["camera"].set_status("WAITING", "#FACC15")
        self.status_cards["yolo"].set_status("WAITING", "#FACC15")
        self.status_cards["voice"].set_status("READY", "#38BDF8")
        self.status_cards["task"].set_status("READY", "#A78BFA")
        self.status_cards["safety"].set_status("UNKNOWN", "#FACC15")
        self.set_default_object_table()

    def on_replay_voice_clicked(self):
        self.add_log("REPLAY VOICE button clicked")
        self.voice_bubble.setText(f"🤖  {self.last_notice}")

    def on_voice_toggle_clicked(self):
        self.voice_enabled = not self.voice_enabled

        if self.voice_enabled:
            self.mute_btn.setText("VOICE OFF")
            self.status_cards["voice"].set_status("READY", "#38BDF8")
            self.set_voice_notice(
                "음성 안내 표시를 켰습니다.\n"
                "TTS 서비스가 연결되면 실제 음성 출력 ON/OFF와 연동할 수 있습니다."
            )
            self.add_log("VOICE display enabled")
        else:
            self.mute_btn.setText("VOICE ON")
            self.status_cards["voice"].set_status("MUTED", "#FACC15")
            self.set_voice_notice(
                "음성 안내 표시를 껐습니다.\n"
                "현재는 UI 표시 전용이며, 실제 TTS 제어는 아직 연결하지 않았습니다."
            )
            self.add_log("VOICE display muted")

    def set_default_object_table(self):
        table_data = []

        for class_name in ITEM_ORDER:
            item = ITEM_CLASS_MAP[class_name]
            target_zone = item["zone"]

            table_data.append(
                [
                    item["display_name"],
                    class_name,
                    target_zone,
                    target_zone,
                    str(item["quantity"]),
                    "READY",
                ]
            )

        self.update_object_table(table_data)

    def update_object_table(self, table_data):
        self.table.setRowCount(len(table_data))

        for row, data in enumerate(table_data):
            for col, value in enumerate(data):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)

                if col == 5:
                    item.setForeground(Qt.cyan)

                self.table.setItem(row, col, item)

    def update_detected_objects_from_classes(self, class_names):
        """
        추후 object_detection_node 또는 task_manager_node가
        YOLO class name 목록을 HMI로 전달할 때 사용할 수 있는 함수입니다.

        예:
            class_names = ["hammer", "green_apple", "pocari"]
        """
        table_data = []

        for class_name in class_names:
            mapping = ITEM_CLASS_MAP.get(class_name)

            if mapping is None:
                table_data.append([class_name, class_name, "-", "-", "1", "UNKNOWN"])
                continue

            target_zone = mapping["zone"]

            table_data.append(
                [
                    mapping["display_name"],
                    class_name,
                    target_zone,
                    target_zone,
                    str(mapping["quantity"]),
                    "DETECTED",
                ]
            )

        self.update_object_table(table_data)

    def on_task_status_received(self, status):
        self.add_log(f"/task_status: {status}")

        display, color = self.convert_task_status(status)
        self.status_cards["task"].set_status(display, color)

    def on_user_notice_received(self, notice):
        self.add_log(f"/user_notice: {notice}")
        self.set_voice_notice(notice)

    def on_safety_state_received(self, safety_state):
        self.add_log(f"/safety_state: {safety_state}")

        display = safety_state.upper()
        color = "#22C55E"

        if "stop" in safety_state.lower() or "emergency" in safety_state.lower():
            color = "#EF4444"
        elif "warning" in safety_state.lower():
            color = "#FACC15"

        self.status_cards["safety"].set_status(display, color)

    def convert_task_status(self, status):
        status_lower = status.lower()

        if "ready" in status_lower or status_lower == "idle":
            return "READY", "#22C55E"

        if "checking" in status_lower or "scan" in status_lower:
            return "SCANNING", "#FACC15"

        if "recheck" in status_lower:
            return "RECHECKING", "#FACC15"

        if "judging" in status_lower or "judgement" in status_lower:
            return "JUDGING", "#38BDF8"

        if "misplaced" in status_lower:
            return "MISPLACED", "#EF4444"

        if "all_clear" in status_lower or "clear" in status_lower:
            return "ALL CLEAR", "#22C55E"

        if "organize" in status_lower or "organizing" in status_lower:
            return "ORGANIZING", "#FACC15"

        if "finished" in status_lower or "done" in status_lower or "complete" in status_lower:
            return "FINISHED", "#22C55E"

        if "failed" in status_lower or "error" in status_lower or "unavailable" in status_lower:
            return "ERROR", "#EF4444"

        if "busy" in status_lower:
            return "BUSY", "#FACC15"

        if "stop" in status_lower:
            return "STOPPED", "#EF4444"

        return status.upper(), "#A78BFA"

    def closeEvent(self, event):
        if self.ros_bridge is not None:
            self.ros_bridge.stop_bridge()
            self.ros_bridge.wait(1000)

        event.accept()


def main(args=None):
    app = QApplication(sys.argv)

    app.setStyleSheet(
        """
        QWidget {
            background-color: #0F172A;
            color: #E5E7EB;
            font-family: Arial;
            font-size: 14px;
        }

        #MainTitle {
            font-size: 30px;
            font-weight: bold;
            color: #F8FAFC;
        }

        #SubTitle {
            font-size: 13px;
            color: #94A3B8;
        }

        #HeaderStatus {
            font-size: 18px;
            font-weight: bold;
            color: #FACC15;
            padding: 10px 18px;
            border: 1px solid #FACC15;
            border-radius: 12px;
            background-color: rgba(250, 204, 21, 0.10);
        }

        #Panel {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 16px;
        }

        #PanelTitle {
            font-size: 18px;
            font-weight: bold;
            color: #38BDF8;
            margin-bottom: 8px;
        }

        #StatusCard {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 14px;
        }

        #StatusTitle {
            color: #94A3B8;
            font-size: 12px;
            font-weight: bold;
        }

        #StatusValue {
            font-size: 15px;
            font-weight: bold;
        }

        #CameraView {
            background-color: #020617;
            border: 2px dashed #334155;
            border-radius: 14px;
            color: #94A3B8;
            font-size: 18px;
            font-weight: bold;
            line-height: 150%;
        }

        #ZoneBox {
            background-color: #0F172A;
            border: 1px solid #334155;
            border-radius: 14px;
            min-height: 140px;
        }

        #ZoneBoxWarning {
            background-color: rgba(239, 68, 68, 0.12);
            border: 2px solid #EF4444;
            border-radius: 14px;
            min-height: 140px;
        }

        #ZoneTitle {
            font-size: 20px;
            font-weight: bold;
            color: #F8FAFC;
        }

        #ZoneRule {
            font-size: 13px;
            color: #38BDF8;
        }

        #ZoneObjects {
            font-size: 16px;
            color: #E5E7EB;
        }

        QTableWidget {
            background-color: #0F172A;
            border: 1px solid #334155;
            border-radius: 10px;
            gridline-color: #334155;
            color: #E5E7EB;
            selection-background-color: #2563EB;
        }

        QTableWidget::item {
            padding: 6px;
        }

        QHeaderView::section {
            background-color: #334155;
            color: #F8FAFC;
            padding: 8px;
            border: none;
            font-weight: bold;
        }

        #VoiceBubble {
            background-color: #020617;
            border: 1px solid #38BDF8;
            border-radius: 16px;
            padding: 18px;
            color: #E0F2FE;
            font-size: 17px;
            line-height: 150%;
        }

        QPushButton {
            background-color: #2563EB;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 16px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #1D4ED8;
        }

        QPushButton:pressed {
            background-color: #1E40AF;
        }

        #StopButton {
            background-color: #DC2626;
        }

        #StopButton:hover {
            background-color: #B91C1C;
        }

        #LogBox {
            background-color: #020617;
            border: 1px solid #334155;
            border-radius: 10px;
            color: #CBD5E1;
            font-family: Consolas;
            font-size: 13px;
        }
        """
    )

    window = SortingRobotHMI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
