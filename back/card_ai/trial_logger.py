from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

# 终局名次标准计分字典 (第一名 +20, 第二名 +10, 第三名 0, 第四名 -20)
BASE_RANK_SCORES = {
    1: 20.0,
    2: 10.0,
    3: 0.0,
    4: -20.0,
}

# 预置并列区间计算表 (均值分摊)
RANK_INTERVAL_OPTIONS = {
    "1-1": {"label": "独占第 1 名 (+20分)", "min_rank": 1, "max_rank": 1, "score": 20.0},
    "2-2": {"label": "独占第 2 名 (+10分)", "min_rank": 2, "max_rank": 2, "score": 10.0},
    "3-3": {"label": "独占第 3 名 (0分)", "min_rank": 3, "max_rank": 3, "score": 0.0},
    "4-4": {"label": "独占第 4 名 (-20分)", "min_rank": 4, "max_rank": 4, "score": -20.0},
    "1-2": {"label": "并列第 1~2 名 (+15分)", "min_rank": 1, "max_rank": 2, "score": 15.0},
    "2-3": {"label": "并列第 2~3 名 (+5分)", "min_rank": 2, "max_rank": 3, "score": 5.0},
    "3-4": {"label": "并列第 3~4 名 (-10分)", "min_rank": 3, "max_rank": 4, "score": -10.0},
    "1-3": {"label": "并列第 1~3 名 (+10分)", "min_rank": 1, "max_rank": 3, "score": 10.0},
    "2-4": {"label": "并列第 2~4 名 (-3.33分)", "min_rank": 2, "max_rank": 4, "score": -3.3333},
    "1-4": {"label": "四人并列第 1~4 名 (+2.5分)", "min_rank": 1, "max_rank": 4, "score": 2.5},
}


def compute_interval_score(min_rank: int, max_rank: int) -> float:
    """按标准规则计算并列名次区间分摊得分。"""
    ranks = list(range(min_rank, max_rank + 1))
    return sum(BASE_RANK_SCORES[r] for r in ranks) / len(ranks)


class TrialLogger:
    """负责人工监督试用期间的决策级 JSONL 增量追加与整场聚合结算。"""

    def __init__(self, log_dir: str | Path | None = None) -> None:
        base = Path(log_dir or Path(__file__).resolve().parent.parent / "runs_deep_cfr")
        base.mkdir(parents=True, exist_ok=True)
        self.events_file = base / "human_supervised_trial_events.jsonl"
        self.games_file = base / "human_supervised_trial_games.json"

    def log_decision_event(
        self,
        *,
        game_id: str,
        round_index: int,
        turn_index: int,
        is_test: bool,
        model_meta: dict[str, Any],
        state_snapshot: dict[str, Any],
        interaction_target: dict[str, Any],
        sampled_action: str,
        sampled_prob: float,
        action_selection_probs: list[dict[str, Any]],
        human_actual_action: str,
        is_deviated: bool,
        deviation_reason: str,
    ) -> dict[str, Any]:
        """记录单次决策事件并立即增量追加落盘至 JSONL。"""
        decision_id = f"dec_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
        event_record = {
            "event_type": "decision_event",
            "decision_id": decision_id,
            "game_id": game_id,
            "round_index": round_index,
            "turn_index": turn_index,
            "is_test": is_test,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_meta": model_meta,
            "state_snapshot": state_snapshot,
            "interaction_target": interaction_target,
            "model_recommendation": {
                "sampled_action": sampled_action,
                "sampled_prob": sampled_prob,
                "action_selection_probs": action_selection_probs,
            },
            "human_execution": {
                "human_actual_action": human_actual_action,
                "is_deviated": is_deviated,
                "deviation_reason": deviation_reason,
            },
        }

        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_record, ensure_ascii=False) + "\n")

        return event_record

    def log_game_summary(
        self,
        *,
        game_id: str,
        is_test: bool,
        model_meta: dict[str, Any],
        min_rank: int,
        max_rank: int,
        notes: str = "",
    ) -> dict[str, Any]:
        """整场结束时追加结算事件，并同步更新汇总 JSON。"""
        calculated_score = compute_interval_score(min_rank, max_rank)
        interval_key = f"{min_rank}-{max_rank}"
        interval_label = RANK_INTERVAL_OPTIONS.get(interval_key, {}).get(
            "label", f"第 {min_rank}~{max_rank} 名 ({calculated_score:+.2f}分)"
        )

        summary_record = {
            "event_type": "game_summary_event",
            "game_id": game_id,
            "is_test": is_test,
            "end_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_meta": model_meta,
            "rank_result": {
                "min_rank": min_rank,
                "max_rank": max_rank,
                "interval_key": interval_key,
                "interval_label": interval_label,
                "final_score": calculated_score,
            },
            "notes": notes,
        }

        # 1. 增量写 JSONL
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

        # 2. 更新整场汇总 JSON
        games_list = []
        if self.games_file.is_file():
            try:
                games_list = json.loads(self.games_file.read_text(encoding="utf-8"))
            except Exception:
                games_list = []

        # 排除同 game_id 重复提交
        games_list = [g for g in games_list if g.get("game_id") != game_id]
        games_list.append(summary_record)
        self.games_file.write_text(json.dumps(games_list, indent=2, ensure_ascii=False), encoding="utf-8")

        return summary_record

    def get_trial_statistics(self) -> dict[str, Any]:
        """统计当前正式试用场次与累计指标 (自动排除 is_test=True 的对局)。"""
        if not self.games_file.is_file():
            return {
                "formal_games_count": 0,
                "test_games_count": 0,
                "mean_score": 0.0,
                "win_count": 0,
                "rank4_count": 0,
            }

        try:
            games = json.loads(self.games_file.read_text(encoding="utf-8"))
        except Exception:
            games = []

        formal_games = [g for g in games if not g.get("is_test", False)]
        test_games = [g for g in games if g.get("is_test", False)]

        if not formal_games:
            return {
                "formal_games_count": 0,
                "test_games_count": len(test_games),
                "mean_score": 0.0,
                "win_count": 0,
                "rank4_count": 0,
            }

        scores = [g["rank_result"]["final_score"] for g in formal_games]
        wins = [g for g in formal_games if g["rank_result"]["min_rank"] == 1]
        last = [g for g in formal_games if g["rank_result"]["max_rank"] == 4]

        return {
            "formal_games_count": len(formal_games),
            "test_games_count": len(test_games),
            "mean_score": round(sum(scores) / len(scores), 4),
            "win_rate": round(len(wins) / len(formal_games) * 100, 2),
            "rank4_rate": round(len(last) / len(formal_games) * 100, 2),
        }
