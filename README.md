# 骗子酒馆 AI

## 启动

```bash
# 安装依赖
pip install -r back/requirements.txt
cd front && npm install && cd ..

# 启动桌面助手（默认 v54 混合路由）
python back/run_desktop_assistant.py
cd front && npm start

# 控制台对战
python back/run_human_vs_ai.py --model_path release/ppo_c_long_v29.pt
```

## 模型说明
- `release/ppo_c_long_v29.pt`: 主力模型 (PPO)
- `release/deep_cfr_v21_baseline.pt`: 基线模型 (CFR)
- `release/v54_B_seed1_front.pt`: v54 B_seed1 前半场保底模型
- `release/v54_A_seed1_front.pt`: v54 A_seed1 强手桌前半场模型
- `release/v51_B_seed2_duel.pt`: v51 B_seed2 两人局冻结专家
- `release/v43_B10_front.pt`: v43_B10 前半场兼容基线

桌面助手默认使用 `v54_B_seed1_front → v51_B_seed2_duel`：三至四人阶段使用 B1，
在两人存活且质疑、开枪、换轮全部结算完成后切换 B2。设置中可选择强手桌的
`v54_A_seed1_front → v51_B_seed2_duel` 路由，或切回固定模型。

最新 v54 完整对局考卷（每个配置 1,440 局，三桌等权）结果：

| 配置 | 第一名率 | 前二率 | 第四名率 | 原标尺场均分 |
| --- | ---: | ---: | ---: | ---: |
| A_seed1 → B2 | 50.833% | 67.604% | 16.979% | 6.818 |
| B_seed1 → B2（默认） | **51.667%** | 67.326% | **16.181%** | **7.003** |
| B_seed2 → B2 | 50.069% | 64.815% | 16.898% | 6.242 |

详细权重哈希和评测口径见 `release/v54_route_manifest.json`。
