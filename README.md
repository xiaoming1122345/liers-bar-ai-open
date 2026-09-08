# 骗子酒馆ai

## 启动

```bash
# 安装依赖
pip install -r back/requirements.txt
cd front && npm install && cd ..

# 启动桌面助手
python back/run_desktop_assistant.py --model release/ppo_c_long_v29.pt
cd front && npm start

# 控制台对战
python back/run_human_vs_ai.py --model_path release/ppo_c_long_v29.pt
```

## 模型说明
- `release/ppo_c_long_v29.pt`: 主力模型 (PPO)
- `release/deep_cfr_v21_baseline.pt`: 基线模型 (CFR)
