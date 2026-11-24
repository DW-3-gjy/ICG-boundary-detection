"""Local, read-only progress window; uses the existing LabelMe PySide6 runtime."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QProgressBar,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from training_progress_state import snapshot

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = REPO / 'model/logs/patient_boundary_20261105_01'


def duration(value):
    if value is None:
        return '估算中'
    minutes = max(0, round(value / 60))
    hours, minutes = divmod(minutes, 60)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f'{days}天 {hours}小时'
    if hours:
        return f'{hours}小时 {minutes:02d}分'
    return f'{minutes}分钟'


def label(text='', name=None):
    widget = QLabel(text)
    widget.setTextFormat(Qt.PlainText)
    if name:
        widget.setObjectName(name)
    return widget


def card():
    widget = QFrame()
    widget.setObjectName('card')
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(9)
    return widget, layout


def progress_bar(height=12):
    bar = QProgressBar()
    bar.setRange(0, 10000)
    bar.setTextVisible(False)
    bar.setFixedHeight(height)
    return bar


class ProgressWindow(QMainWindow):
    def __init__(self, study):
        super().__init__()
        self.study = study
        self.setWindowTitle('ICG 实时训练进度')
        icon = QPixmap(64, 64)
        icon.fill(Qt.transparent)
        painter = QPainter(icon)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor('#257C70'))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, 64, 64, 15, 15)
        painter.setPen(QColor('white'))
        painter.setFont(QFont('Segoe UI', 21, QFont.Bold))
        painter.drawText(icon.rect(), Qt.AlignCenter, 'ICG')
        painter.end()
        self.setWindowIcon(QIcon(icon))
        self.setStyleSheet('''
            QMainWindow, QScrollArea, QWidget#body { background: #F3F6FA; }
            QWidget { font-family: 'Microsoft YaHei UI', 'Microsoft YaHei'; font-size: 13px; color: #20384D; }
            QFrame#card { background: white; border: 1px solid #E1E8EF; border-radius: 12px; }
            QLabel#title { font-size: 24px; font-weight: 700; }
            QLabel#muted { color: #6C7E8F; font-size: 12px; }
            QLabel#metric { font-size: 24px; font-weight: 700; color: #1C514F; }
            QLabel#section { font-size: 15px; font-weight: 700; }
            QLabel#percent { font-size: 21px; font-weight: 700; color: #257C70; }
            QProgressBar { border: none; border-radius: 6px; background: #E8EEF3; }
            QProgressBar::chunk { background: #2B8B7C; border-radius: 6px; }
            QPushButton { background: white; border: 1px solid #D5DFE7; border-radius: 7px; padding: 7px 12px; }
            QPushButton:hover { background: #E8F2EF; border-color: #8EB8AD; }
            QPushButton:pressed { background: #D6E9E1; }
        ''')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body.setObjectName('body')
        outer = QVBoxLayout(body)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(12)
        scroll.setWidget(body)
        self.setCentralWidget(scroll)

        heading = QHBoxLayout()
        heading.addWidget(label('ICG 训练进度', 'title'))
        heading.addStretch()
        self.status = label('读取中')
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setMinimumWidth(100)
        self.status.setMaximumWidth(390)
        self.status.setWordWrap(True)
        heading.addWidget(self.status)
        outer.addLayout(heading)
        outer.addWidget(label('五组对照 · 三折交叉验证 · 三个随机种子', 'muted'))

        metrics = QHBoxLayout()
        self.metric_labels = []
        for title in ['已完成训练运行', '有效运行耗时', '预计训练剩余']:
            box, box_layout = card()
            box_layout.addWidget(label(title, 'muted'))
            metric = label('—', 'metric')
            box_layout.addWidget(metric)
            self.metric_labels.append(metric)
            metrics.addWidget(box, 1)
        outer.addLayout(metrics)

        box, layout = card()
        row = QHBoxLayout()
        row.addWidget(label('整体训练进度', 'section'))
        row.addStretch()
        self.overall_percent = label('0.0%', 'percent')
        row.addWidget(self.overall_percent)
        layout.addLayout(row)
        self.overall_bar = progress_bar(15)
        layout.addWidget(self.overall_bar)
        self.phases = label('', 'muted')
        self.phases.setWordWrap(True)
        layout.addWidget(self.phases)
        outer.addWidget(box)

        box, layout = card()
        self.current = label('当前运行', 'section')
        self.current.setWordWrap(True)
        layout.addWidget(self.current)
        row = QHBoxLayout()
        self.steps = label('—')
        row.addWidget(self.steps)
        row.addStretch()
        self.current_percent = label('—')
        row.addWidget(self.current_percent)
        layout.addLayout(row)
        self.current_bar = progress_bar()
        layout.addWidget(self.current_bar)
        self.values = label('', 'muted')
        self.values.setWordWrap(True)
        layout.addWidget(self.values)
        self.updated = label('', 'muted')
        self.updated.setWordWrap(True)
        layout.addWidget(self.updated)
        outer.addWidget(box)

        box, layout = card()
        layout.addWidget(label('各组完成情况', 'section'))
        self.group_rows = {}
        for group in 'ABCDE':
            row = QHBoxLayout()
            title = label(group)
            title.setMinimumWidth(210)
            bar = progress_bar(9)
            count = label('0 / 12')
            count.setMinimumWidth(148)
            count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(title)
            row.addWidget(bar, 1)
            row.addWidget(count)
            layout.addLayout(row)
            self.group_rows[group] = (title, bar, count)
        outer.addWidget(box)
        self.note = label('', 'muted')
        self.note.setWordWrap(True)
        outer.addWidget(self.note)
        footer = QHBoxLayout()
        close_note = label('仅查看日志；关闭此窗口不会停止训练。', 'muted')
        close_note.setWordWrap(True)
        footer.addWidget(close_note, 1)
        refresh = QPushButton('立即刷新')
        refresh.clicked.connect(self.refresh)
        footer.addWidget(refresh)
        report = QPushButton('打开报告目录')
        report.clicked.connect(self.open_reports)
        footer.addWidget(report)
        outer.addLayout(footer)
        outer.addStretch()
        available = QApplication.primaryScreen().availableGeometry()
        self.resize(min(930, available.width() - 60), min(820, available.height() - 60))
        self.setMinimumSize(700, 480)
        self.move(available.center() - self.rect().center())
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2000)
        self.refresh()

    def open_reports(self):
        reports = REPO.parent / 'output/pdf'
        if reports.is_dir():
            os.startfile(str(reports))

    def refresh(self):
        try:
            state = snapshot(self.study)
            self.render(state)
        except Exception as error:
            self.status.setText('暂无法读取')
            self.status.setStyleSheet('background:#FBECE5;color:#965035;border-radius:8px;padding:6px 12px;')
            self.note.setText(f'日志可能正在写入，将自动重试。{type(error).__name__}: {error}')

    def render(self, state):
        self.status.setText(state['status_text'])
        kind = state['status']
        bg, fg = ('#E0F1EB', '#246D59')
        if kind in ('paused', 'stale', 'waiting'):
            bg, fg = '#FFF2D9', '#90652B'
        if kind == 'error':
            bg, fg = '#FAE7E4', '#A04B42'
        self.status.setStyleSheet(f'background:{bg};color:{fg};border-radius:8px;padding:6px 12px;font-weight:600;')
        self.metric_labels[0].setText(f"{state['completed_total']} / {state['total_runs']}")
        self.metric_labels[1].setText(duration(state['active_elapsed_seconds']))
        self.metric_labels[2].setText('已暂停' if kind == 'paused' else duration(state['eta_seconds']))
        fraction = min(1., max(0., state['overall_fraction']))
        self.overall_bar.setValue(round(fraction * 10000))
        self.overall_percent.setText(f'{fraction:.1%}')
        self.phases.setText(f"开发验证 {state['dev_completed']}/{state['dev_total']}　·　最终重训 {state['refit_completed']}/{state['refit_total']}　｜按运行次数计量，非耗时占比")
        group = next((g for g in state['groups'] if g['id'] == state['group']), None)
        name = group['description'] if group else state['current_run']
        seed = f" · 种子 {state['seed']}" if state['seed'] is not None else ''
        self.current.setText(f"{state['group']}组 · {name} · {state['fold_label']}{seed}" if group else name or '等待训练日志')
        self.steps.setText(f"{state['update']:,} / {state['max_updates']:,} 次更新")
        self.current_percent.setText(f"{state['current_fraction']:.1%}")
        self.current_bar.setValue(round(state['current_fraction'] * 10000))
        best = state.get('best_validation')
        if state['fold_label'] == '最终重训':
            metric = '最终重训沿用开发阶段确定的轮次与阈值'
        else:
            metric = f'本次最佳验证 Boundary F1：{best:.3f}' if best is not None else '本次验证结果待写入'
        loss = state.get('training_loss')
        if loss is not None:
            metric += f'　｜最近一次更新损失：{loss:.4f}'
        self.values.setText(metric + '　（均非测试成绩）')
        test = '已完成' if state['final_results_available'] else '统一评估中' if state['test_started'] else '尚未开启'
        self.updated.setText(f"日志时间：{state['updated_at'] or '等待写入'}　｜测试集：{test}　｜窗口每2秒刷新，步数随40步日志推进")
        for g in state['groups']:
            title, bar, count = self.group_rows[g['id']]
            title.setText(f"{g['id']}　{g['description']}")
            bar.setValue(round(g['fraction'] * 10000))
            count.setText(f"开发 {g['dev_completed']}/9 · 重训 {g['refit_completed']}/3")
            title.setStyleSheet('font-weight:700;color:#257C70;' if g['id'] == state['group'] else '')
        notes = [state.get('eta_note', '')]
        notes += state.get('warnings', [])
        if state['report_ready']:
            notes.append('最终报告已生成，可从“打开报告目录”查看。')
        self.note.setText('\n'.join(n for n in notes if n))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    if args.screenshot:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    app = QApplication(sys.argv[:1])
    app.setApplicationName('ICG Training Progress')
    # Also support Qt's offscreen platform, which does not enumerate Windows fonts.
    for filename in ['msyh.ttc', 'msyhbd.ttc', 'segoeui.ttf']:
        font_path = Path('C:/Windows/Fonts') / filename
        if font_path.is_file():
            QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont('Microsoft YaHei UI', 10))
    window = ProgressWindow(args.study.resolve())
    window.show()
    if args.screenshot:
        def capture():
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            ok = window.grab().save(str(args.screenshot))
            print(json.dumps({'screenshot': str(args.screenshot), 'saved': ok}, ensure_ascii=False), flush=True)
            app.exit(0 if ok else 1)
        QTimer.singleShot(500, capture)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
