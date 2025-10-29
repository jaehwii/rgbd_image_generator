import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class RenderSummary:
    frames_total: int = 0
    scene_id: str = ''
    scene_root: str = ''
    engine_name: str = '?'
    device_name: str = '?'
    t0: float = field(default_factory=time.perf_counter)
    per_frame_times: List[float] = field(default_factory=list)
    _t_frame_start: float | None = None

    def start_frame_timer(self) -> None:
        self._t_frame_start = time.perf_counter()

    def stop_frame_timer(self) -> float:
        if self._t_frame_start is None:
            dt = 0.0
        else:
            dt = time.perf_counter() - self._t_frame_start
        self.per_frame_times.append(dt)
        self._t_frame_start = None
        return dt

    def add_frame_num(self, n: int = 1) -> None:
        self.frames_total += n

    def print(self) -> None:
        elapsed = time.perf_counter() - self.t0
        hh = int(elapsed // 3600)
        mm = int((elapsed % 3600) // 60)
        ss = elapsed % 60
        avg_t = (
            (sum(self.per_frame_times) / len(self.per_frame_times))
            if self.per_frame_times
            else 0.0
        )
        max_t = max(self.per_frame_times) if self.per_frame_times else 0.0
        print('\n' + '=' * 72)
        print('[SUMMARY]')
        print(f'  Scene ID         : {self.scene_id}')
        print(f'  Output Dir       : {self.scene_root}')
        print(f'  Engine/Device    : {self.engine_name}/{self.device_name}')
        print('  Sequences        : 1')
        print(f'  Frames           : {self.frames_total}')
        print(f'  Total time       : {hh:02d}:{mm:02d}:{ss:05.2f}  ({elapsed:.2f} s)')
        print(f'  Avg per frame    : {avg_t:.2f} s')
        print(f'  Max per frame    : {max_t:.2f} s')
        print('=' * 72)
