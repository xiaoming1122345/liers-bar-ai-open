# Liar's Bar AI (愚人牌 AI)

基于 Deep CFR 与 PPO 强化学习的逆水寒大逃杀（愚人牌）AI 与桌面悬浮窗对局助手。

## 快速启动

### 1. 环境准备
```bash
pip install -r back/requirements.txt
cd front && npm install && cd ..
```

### 2. 启动桌面助手
```bash
# 启动后端推断服务（默认加载生产主力 PPO 模型）
python back/run_desktop_assistant.py --model release/ppo_c_long_v29.pt

# 启动前端悬浮窗 HUD
cd front && npm start
```

### 3. 控制台人机对战测试
```bash
python back/run_human_vs_ai.py --model_path release/ppo_c_long_v29.pt
```

## 预置模型
- `release/ppo_c_long_v29.pt`: 现役实战生产主力（PPO 强化学习）
- `release/deep_cfr_v21_baseline.pt`: 官方防守基线（Deep CFR 纳什均衡）
