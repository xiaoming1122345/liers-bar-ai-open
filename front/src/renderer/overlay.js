let state = null;
let runtimeAdvice = null;
let analysisTimer = null;
let analysisRevision = 0;

let trialManifest = null;
let trialStats = null;
let currentTrialGameId = `game_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
let currentTurnDecisionCount = 0;
let isTestMode = false;
let lastRecordedStateFingerprint = "";

const elements = {};

function $(id) {
  return document.getElementById(id);
}

function formatPercentFromUnit(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Math.round(Number(value) * 100)}%`;
}

function formatScore(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(2);
}

function seatAnalysisMap() {
  const map = new Map();
  for (const seat of runtimeAdvice?.seat_analysis || []) {
    map.set(Number(seat.seat), seat);
  }
  return map;
}

function seatCardHtml(seat, currentTurnSeat, analysis) {
  const current = seat.seat === currentTurnSeat ? "current" : "";
  const metrics = seat.metrics || null;
  const modeled = analysis || {};
  return `
    <div class="seat-card seat-${seat.seat} ${current}" data-seat-card="${seat.seat}">
      <div class="seat-topline">
        <button class="seat-turn-btn" data-seat-turn="${seat.seat}">#${seat.seat}</button>
        <label class="seat-slot">
          <span>ID</span>
          <input data-seat-id="${seat.seat}" type="text" value="${seat.playerId || ""}" placeholder="player id" />
        </label>
        <div class="seat-slot">
          <span>状态</span>
          <select data-seat-status="${seat.seat}">
            <option value="alive" ${seat.status === "alive" ? "selected" : ""}>alive</option>
            <option value="eliminated" ${seat.status === "eliminated" ? "selected" : ""}>eliminated</option>
            <option value="escaped" ${seat.status === "escaped" ? "selected" : ""}>escaped</option>
          </select>
        </div>
      </div>
      <div class="seat-stats">
        <label class="seat-slot">
          <span>枪位</span>
          <input data-seat-shots="${seat.seat}" type="number" min="0" max="5" value="${seat.shotsTaken}" />
        </label>
        <label class="seat-slot">
          <span>手牌</span>
          <input data-seat-hand="${seat.seat}" type="number" min="0" max="5" value="${seat.handCount}" />
        </label>
        <div class="seat-slot">
          <span>显示</span>
          <strong>${seat.displayName}</strong>
          <small>余 ${seat.safeSlotsLeft}</small>
        </div>
      </div>
      <label class="seat-slot seat-last-action">
        <span>上一动作</span>
        <input data-seat-last-action="${seat.seat}" type="text" value="${seat.lastAction || ""}" placeholder="上一动作" />
      </label>
      <div class="seat-metrics">
        <div class="metric-pill"><span>风格</span><strong>${modeled.public_style_cluster || metrics?.publicStyleCluster || "-"}</strong></div>
        <div class="metric-pill"><span>画像</span><strong>${modeled.inferred_profile_name || metrics?.inferredProfileName || "-"}</strong></div>
        <div class="metric-pill"><span>受压</span><strong>${modeled.pressure_response_label || metrics?.pressureResponseLabel || "-"}</strong></div>
        <div class="metric-pill"><span>威胁</span><strong>${formatPercentFromUnit(modeled.threat_score)}</strong></div>
        <div class="metric-pill"><span>质疑</span><strong>${formatPercentFromUnit(modeled.challenge_pressure)}</strong></div>
        <div class="metric-pill"><span>诈牌</span><strong>${formatPercentFromUnit(modeled.bluff_pressure ?? metrics?.bluffRate)}</strong></div>
      </div>
      <div class="seat-metrics">
        <div class="metric-pill"><span>目标估计</span><strong>${formatScore(modeled.target_card_estimate)}</strong></div>
        <div class="metric-pill"><span>危险率</span><strong>${formatPercentFromUnit(modeled.elimination_hazard)}</strong></div>
        <div class="metric-pill"><span>手型</span><strong>${modeled.likely_bucket || "-"}</strong></div>
      </div>
      <div class="seat-beliefs">
        <label class="belief-box">
          <span>target</span>
          <input data-seat-belief="${seat.seat}" data-belief-key="target" type="number" min="0" max="100" value="${seat.manualBeliefs.target}" />
        </label>
        <label class="belief-box">
          <span>non</span>
          <input data-seat-belief="${seat.seat}" data-belief-key="nonTarget" type="number" min="0" max="100" value="${seat.manualBeliefs.nonTarget}" />
        </label>
        <label class="belief-box">
          <span>ghost</span>
          <input data-seat-belief="${seat.seat}" data-belief-key="ghost" type="number" min="0" max="100" value="${seat.manualBeliefs.ghost}" />
        </label>
        <label class="belief-box">
          <span>wild</span>
          <input data-seat-belief="${seat.seat}" data-belief-key="wild" type="number" min="0" max="100" value="${seat.manualBeliefs.wild}" />
        </label>
      </div>
    </div>
  `;
}

function renderHistory() {
  const history = state?.session?.history || [];
  elements.historyCount.textContent = String(history.length);
  elements.historyList.innerHTML = history.length === 0
    ? `<div class="history-entry"><strong>暂无记录</strong><span>从下方开始录入。</span></div>`
    : history.slice(0, 30).map((item) => `
        <div class="history-entry">
          <strong>${item.summary}</strong>
          <span>${item.createdAt}</span>
        </div>
      `).join("");
}

function renderAdvice() {
  const recommendation = runtimeAdvice?.recommended_action || null;
  const factors = runtimeAdvice?.table_factors || {};
  elements.adviceConfidence.textContent = recommendation
    ? `${Math.round((recommendation.confidence || 0) * 100)}%`
    : "-";
  elements.adviceLabel.textContent = recommendation?.label || "等待输入";
  elements.adviceMeta.textContent = runtimeAdvice
    ? `建模度 ${Math.round((factors.modeling_degree || 0) * 100)}% · 下家 ${factors.next_responder_seat ?? "-"}`
    : "录入桌面状态后会自动刷新建议";
  elements.adviceReasons.innerHTML = (recommendation?.reasons || ["录入 seat / 目标 / 手里抽象牌数后开始分析。"])
    .slice(0, 4)
    .map((reason) => `<div class="advice-reason">${reason}</div>`)
    .join("");
  elements.adviceOptions.innerHTML = (runtimeAdvice?.action_options || [])
    .slice(0, 4)
    .map((option) => `
      <div class="advice-option">
        <strong>${option.label}</strong>
        <div>score ${formatScore(option.score)} · conf ${Math.round((option.confidence || 0) * 100)}%</div>
      </div>
    `)
    .join("");

  // 1. 动作交互目标卡片 (实际接牌者 vs 质疑对象)
  const targetInfo = runtimeAdvice?.interaction_target || null;
  if (targetInfo && elements.interactionTargetBox) {
    const isChallenge = targetInfo.role === "质疑对象";
    const roleIcon = isChallenge ? "⚔️" : "🎯";
    elements.interactionTargetBox.innerHTML = `
      <div class="target-head">
        <span>${roleIcon} ${targetInfo.role}</span>
        <span>${targetInfo.relative_pos || "未知"}</span>
      </div>
      <div class="target-body">
        ${targetInfo.display_name || "-"} (Seat ${targetInfo.seat ?? "-"})
        ${targetInfo.hand_count != null ? `<span style="font-size:11px;font-weight:normal;color:var(--muted);margin-left:8px;">手牌 ${targetInfo.hand_count} · 已开枪 ${targetInfo.shots_taken ?? 0}</span>` : ""}
      </div>
      <div class="target-desc">${targetInfo.status_description || ""}</div>
    `;
  } else if (elements.interactionTargetBox) {
    elements.interactionTargetBox.innerHTML = `<div class="target-desc">等待分析推导接牌或质疑目标...</div>`;
  }

  // 2. 神经网络决策展示 (明确动作选择概率与幂等单次采样选择)
  const neural = runtimeAdvice?.neural_advice || null;
  if (neural && elements.neuralAdviceBox) {
    if (!neural.is_valid) {
      elements.neuralAdviceBox.innerHTML = `
        <div class="neural-head" style="color:var(--danger)">⚠️ 生产模型校验失败</div>
        <div style="font-size:12px;color:var(--muted);margin-top:4px;">${neural.validation_message || "模型哈希或路径不匹配"}</div>
      `;
    } else {
      const probsList = (neural.action_selection_probs || []).map((item) => `
        <div class="neural-row ${item.is_sampled ? 'sampled-hit' : ''}">
          <span class="neural-action" style="${item.is_sampled ? 'color:var(--accent);font-weight:bold;' : ''}">
            ${item.is_sampled ? '★ ' : ''}${item.label}
          </span>
          <span class="neural-prob">${item.prob_percent}</span>
        </div>
      `).join("");

      elements.neuralAdviceBox.innerHTML = `
        <div class="neural-head">
          <span>🧠 模型本次选择 (Sampled Choice)</span>
          <small style="font-size:10px;color:var(--muted);">${neural.short_sha256}</small>
        </div>
        <div class="neural-top">
          <strong>${neural.sampled_action ?? neural.top_action ?? "-"}</strong>
          <span class="neural-prob-main">${Math.round((neural.sampled_prob ?? neural.top_prob ?? 0) * 100)}%</span>
        </div>
        <div class="neural-top5">
          <div style="font-size:10px;color:var(--muted);margin-bottom:4px;">各合法动作选择概率 (模型决策倾向，非胜率):</div>
          ${probsList || "<em>无合法候选动作</em>"}
        </div>
      `;
    }
  } else if (elements.neuralAdviceBox) {
    elements.neuralAdviceBox.innerHTML = `<div class="neural-head">🧠 神经网络决策 <span class="neural-offline">模型未就绪</span></div>`;
  }

  // 3. 人工监督试用打卡抽屉
  renderTrialDrawer();

  // 断层3修复：POMDP 对手推断展示
  const pomdp = runtimeAdvice?.pomdp_inference || null;
  if (pomdp) {
    elements.pomdpBox.innerHTML = `
      <div class="pomdp-head">🔍 POMDP 推断</div>
      <div class="pomdp-grid">
        <div class="metric-pill"><span>诚实</span><strong>${formatPercentFromUnit(pomdp.honest_prob)}</strong></div>
        <div class="metric-pill"><span>诈牌</span><strong>${formatPercentFromUnit(pomdp.bluff_prob)}</strong></div>
        <div class="metric-pill"><span>鬼套</span><strong>${formatPercentFromUnit(pomdp.ghost_trap_prob)}</strong></div>
      </div>
    `;
  } else {
    elements.pomdpBox.innerHTML = "";
  }

  // 断层4修复：前瞻推演摘要展示（取第一个节点）
  const lookahead = (runtimeAdvice?.lookahead_tree || [])[0] || null;
  if (lookahead) {
    elements.lookaheadBox.innerHTML = `
      <div class="lookahead-head">📊 前瞻推演</div>
      <div class="lookahead-grid">
        <div class="metric-pill"><span>即时风险</span><strong>${formatPercentFromUnit(lookahead.immediate_risk)}</strong></div>
        <div class="metric-pill"><span>预期存活</span><strong>${formatPercentFromUnit(lookahead.projected_survival_rate)}</strong></div>
        <div class="metric-pill"><span>期望值</span><strong>${formatScore(lookahead.expected_value)}</strong></div>
      </div>
      ${lookahead.tactical_summary ? `<div class="lookahead-summary">${lookahead.tactical_summary}</div>` : ""}
    `;
  } else {
    elements.lookaheadBox.innerHTML = "";
  }
}

function renderSeats() {
  const session = state?.session;
  if (!session) {
    return;
  }
  const analysisBySeat = seatAnalysisMap();
  elements.seatLayer.innerHTML = session.seats
    .map((seat) => seatCardHtml(seat, session.currentTurnSeat, analysisBySeat.get(seat.seat)))
    .join("");
}

function renderHeader() {
  const session = state?.session;
  if (!session) {
    return;
  }
  elements.overlayTitle.textContent = `第 ${session.gameIndex} 局`;
  elements.overlaySubline.textContent = `轮到 ${session.currentTurnSeat} 号位`;
  elements.centerTargetRank.textContent = session.targetRank;
  elements.centerStackCount.textContent = `叠牌 ${session.stackCount}`;
  elements.dockMySeat.value = String(session.mySeat);
  elements.dockTurnSeat.value = String(session.currentTurnSeat);
  elements.dockTargetRank.value = String(session.targetRank);
  elements.dockStackCount.value = String(session.stackCount);
  elements.heroTargetCount.value = String(session.heroHandView.targetCount);
  elements.heroNonTargetCount.value = String(session.heroHandView.nonTargetCount);
  elements.heroGhostCount.value = String(session.heroHandView.ghostCount);
  elements.heroWildCount.value = String(session.heroHandView.wildCount);
  elements.overlayTopmostBtn.textContent = session.overlayTopmost ? "已顶层" : "顶层";
}

function render() {
  renderHeader();
  renderSeats();
  renderAdvice();
  renderHistory();
}

async function requestAnalysis() {
  if (!state?.session) {
    return;
  }
  const revision = ++analysisRevision;
  try {
    const result = await window.cardApi.analyzeRuntime(
      { session: state.session },
      state.priorFiles?.[0]?.path || "",
    );
    if (revision !== analysisRevision) {
      return;
    }
    runtimeAdvice = result;
    render();
  } catch (error) {
    if (revision !== analysisRevision) {
      return;
    }
    runtimeAdvice = {
      recommended_action: {
        label: "分析失败",
        confidence: 0,
        reasons: [String(error.message || error)],
      },
      action_options: [],
      table_factors: {
        modeling_degree: 0,
        next_responder_seat: null,
      },
    };
    renderAdvice();
  }
}

function scheduleAnalysis() {
  clearTimeout(analysisTimer);
  analysisTimer = setTimeout(requestAnalysis, 260);
}

async function applySeatPatchFromInput(target) {
  const seat = Number(
    target.dataset.seatId ||
      target.dataset.seatStatus ||
      target.dataset.seatShots ||
      target.dataset.seatHand ||
      target.dataset.seatLastAction ||
      target.dataset.seatBelief,
  );
  if (!seat) {
    return;
  }
  if (target.dataset.seatId) {
    await window.cardApi.setSeatPlayerId(seat, target.value);
    return;
  }
  if (target.dataset.seatStatus) {
    await window.cardApi.updateSeat(seat, { status: target.value });
    return;
  }
  if (target.dataset.seatShots) {
    await window.cardApi.updateSeat(seat, { shotsTaken: Number(target.value) });
    return;
  }
  if (target.dataset.seatHand) {
    await window.cardApi.updateSeat(seat, { handCount: Number(target.value) });
    return;
  }
  if (target.dataset.seatLastAction) {
    await window.cardApi.updateSeat(seat, { lastAction: target.value });
    return;
  }
  if (target.dataset.seatBelief) {
    await window.cardApi.updateSeat(seat, {
      manualBeliefs: {
        [target.dataset.beliefKey]: Number(target.value),
      },
    });
  }
}

function eventPayload() {
  return {
    seat: Number(elements.eventSeat.value),
    type: elements.eventType.value,
    count: Number(elements.eventCount.value),
    claimRank: elements.eventClaimRank.value,
    targetSeat: elements.eventTargetSeat.value ? Number(elements.eventTargetSeat.value) : null,
    statusValue: elements.eventStatusValue.value,
    handCount: Number(elements.eventHandCount.value),
    composition: elements.eventComposition.value,
    note: elements.eventComposition.value,
    nextTurnSeat: elements.eventNextSeat.value ? Number(elements.eventNextSeat.value) : null,
  };
}

function bindEvents() {
  elements.overlayTopmostBtn.addEventListener("click", () => {
    window.cardApi.setOverlayTopmost(!state.session.overlayTopmost);
  });
  elements.overlayUndoBtn.addEventListener("click", () => window.cardApi.undo());
  elements.overlayRedoBtn.addEventListener("click", () => window.cardApi.redo());
  elements.overlayNextGameBtn.addEventListener("click", () => window.cardApi.nextGame());
  elements.overlayHideBtn.addEventListener("click", () => window.cardApi.setOverlayVisible(false));
  elements.dockMySeat.addEventListener("change", (event) => window.cardApi.setMySeat(Number(event.target.value)));
  elements.dockTurnSeat.addEventListener("change", (event) => window.cardApi.setCurrentTurnSeat(Number(event.target.value)));
  elements.dockTargetRank.addEventListener("change", (event) => window.cardApi.updateSessionMeta({ targetRank: event.target.value }));
  elements.dockStackCount.addEventListener("change", (event) => window.cardApi.updateSessionMeta({ stackCount: Number(event.target.value) }));

  for (const input of [
    elements.heroTargetCount,
    elements.heroNonTargetCount,
    elements.heroGhostCount,
    elements.heroWildCount,
  ]) {
    input.addEventListener("change", () => {
      window.cardApi.setHeroHandView({
        targetCount: Number(elements.heroTargetCount.value),
        nonTargetCount: Number(elements.heroNonTargetCount.value),
        ghostCount: Number(elements.heroGhostCount.value),
        wildCount: Number(elements.heroWildCount.value),
      });
    });
  }

  elements.seatLayer.addEventListener("click", (event) => {
    const button = event.target.closest("[data-seat-turn]");
    if (!button) {
      return;
    }
    window.cardApi.setCurrentTurnSeat(Number(button.dataset.seatTurn));
  });

  elements.seatLayer.addEventListener("change", (event) => {
    applySeatPatchFromInput(event.target);
  });

  elements.seatLayer.addEventListener(
    "blur",
    (event) => {
      applySeatPatchFromInput(event.target);
    },
    true,
  );

  elements.eventForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await window.cardApi.appendEvent(eventPayload());
    elements.eventComposition.value = "";
  });
}

async function bootstrap() {
  elements.overlayTitle = $("overlayTitle");
  elements.overlaySubline = $("overlaySubline");
  elements.overlayTopmostBtn = $("overlayTopmostBtn");
  elements.overlayUndoBtn = $("overlayUndoBtn");
  elements.overlayRedoBtn = $("overlayRedoBtn");
  elements.overlayNextGameBtn = $("overlayNextGameBtn");
  elements.overlayHideBtn = $("overlayHideBtn");
  elements.centerTargetRank = $("centerTargetRank");
  elements.centerStackCount = $("centerStackCount");
  elements.seatLayer = $("seatLayer");
  elements.historyCount = $("historyCount");
  elements.historyList = $("historyList");
  elements.adviceConfidence = $("adviceConfidence");
  elements.adviceLabel = $("adviceLabel");
  elements.adviceMeta = $("adviceMeta");
  elements.adviceReasons = $("adviceReasons");
  elements.adviceOptions = $("adviceOptions");
  // 断层2-4修复：新增三个展示区 DOM 引用
  elements.modelBadge = $("modelBadge");
  elements.interactionTargetBox = $("interactionTargetBox");
  elements.neuralAdviceBox = $("neuralAdviceBox");
  elements.trialDrawerBox = $("trialDrawerBox");
  elements.pomdpBox = $("pomdpBox");
  elements.lookaheadBox = $("lookaheadBox");

  // 初始化获取生产模型校验和试用统计
  try {
    if (window.cardApi?.getTrialManifest) {
      const manifestRes = await window.cardApi.getTrialManifest();
      trialManifest = manifestRes;
      if (elements.modelBadge) {
        if (manifestRes.is_valid) {
          elements.modelBadge.className = "model-badge";
          elements.modelBadge.textContent = `${manifestRes.manifest.model_id} [${manifestRes.manifest.short_sha256}] 锁定`;
        } else {
          elements.modelBadge.className = "model-badge invalid";
          elements.modelBadge.textContent = `模型异常: ${manifestRes.message}`;
        }
      }
    }
    if (window.cardApi?.getTrialStats) {
      trialStats = await window.cardApi.getTrialStats();
    }
  } catch (e) {
    console.warn("Trial manifest init error:", e);
  }
  elements.dockMySeat = $("dockMySeat");
  elements.dockTurnSeat = $("dockTurnSeat");
  elements.dockTargetRank = $("dockTargetRank");
  elements.dockStackCount = $("dockStackCount");
  elements.heroTargetCount = $("heroTargetCount");
  elements.heroNonTargetCount = $("heroNonTargetCount");
  elements.heroGhostCount = $("heroGhostCount");
  elements.heroWildCount = $("heroWildCount");
  elements.eventForm = $("eventForm");
  elements.eventSeat = $("eventSeat");
  elements.eventType = $("eventType");
  elements.eventCount = $("eventCount");
  elements.eventClaimRank = $("eventClaimRank");
  elements.eventTargetSeat = $("eventTargetSeat");
  elements.eventStatusValue = $("eventStatusValue");
  elements.eventHandCount = $("eventHandCount");
  elements.eventComposition = $("eventComposition");
  elements.eventNextSeat = $("eventNextSeat");

  state = await window.cardApi.bootstrap();
  bindEvents();
  render();
  scheduleAnalysis();
  window.cardApi.subscribeState((payload) => {
    state = payload;
    render();
    scheduleAnalysis();
  });
}

bootstrap();

function renderTrialDrawer() {
  if (!elements.trialDrawerBox) return;
  const neural = runtimeAdvice?.neural_advice || null;
  const recommendedAction = neural?.sampled_action || runtimeAdvice?.recommended_action?.label || "-";
  const formalCount = trialStats?.formal_games_count ?? 0;
  const meanScore = trialStats?.mean_score != null ? `${trialStats.mean_score > 0 ? '+' : ''}${trialStats.mean_score.toFixed(2)}` : "-";

  elements.trialDrawerBox.innerHTML = `
    <div class="trial-head">
      <span>📋 人工监督试用打卡 (20~30场)</span>
      <span>正式: ${formalCount}/30 · 均分: ${meanScore}</span>
    </div>
    <div class="trial-row">
      <label style="display:flex;align-items:center;gap:4px;cursor:pointer;">
        <input type="checkbox" id="trialTestModeCheckbox" ${isTestMode ? "checked" : ""} />
        <span>测试对局 (不计入30场)</span>
      </label>
      <span style="color:var(--muted);font-size:10px;margin-left:auto;">Game: ${currentTrialGameId.slice(-6)}</span>
    </div>

    <!-- 步骤1: 决策记录 -->
    <div style="background:rgba(0,0,0,0.25);padding:8px;border-radius:6px;margin-top:4px;">
      <div style="font-size:11px;color:var(--accent-2);margin-bottom:6px;font-weight:600;">
        本手推荐: <strong style="color:#fff;">${recommendedAction}</strong>
      </div>
      <div class="trial-row">
        <span>执行:</span>
        <select id="trialExecType">
          <option value="adopt">采纳推荐动作</option>
          <option value="deviate">偏离 (人工自选)</option>
        </select>
        <select id="trialDeviationReason" style="display:none;">
          <option value="认为对手质疑倾向高">认为对手质疑倾向高</option>
          <option value="不同的手牌安排">不同的手牌安排</option>
          <option value="状态录入有误">状态录入有误</option>
          <option value="保守求稳">保守求稳</option>
          <option value="其他">其他</option>
        </select>
      </div>
      <div class="trial-row" style="margin-top:6px;">
        <button id="trialLogDecisionBtn" class="trial-btn primary" style="width:100%;">记录本手决策 (JSONL)</button>
      </div>
    </div>

    <!-- 步骤2: 整场结算 -->
    <div style="background:rgba(0,0,0,0.25);padding:8px;border-radius:6px;margin-top:6px;">
      <div style="font-size:11px;color:var(--accent);margin-bottom:6px;font-weight:600;">整场终局名次结算 (四人积分恒等守恒):</div>
      <div class="trial-row">
        <select id="trialRankInterval" style="width:100%;">
          <option value="1-1">独占第 1 名 (+20分)</option>
          <option value="2-2">独占第 2 名 (+10分)</option>
          <option value="3-3">独占第 3 名 (0分)</option>
          <option value="4-4">独占第 4 名 (-20分)</option>
          <option value="1-2">并列第 1~2 名 (+15分)</option>
          <option value="2-3">并列第 2~3 名 (+5分)</option>
          <option value="3-4">并列第 3~4 名 (-10分)</option>
          <option value="1-3">并列第 1~3 名 (+10分)</option>
          <option value="2-4">并列第 2~4 名 (-3.33分)</option>
          <option value="1-4">四人并列第 1~4 名 (+2.5分)</option>
        </select>
      </div>
      <div class="trial-row" style="margin-top:6px;">
        <button id="trialFinishGameBtn" class="trial-btn finish">提交整场结算并开新局</button>
      </div>
    </div>
  `;

  // 绑定事件
  const testBox = $("trialTestModeCheckbox");
  if (testBox) {
    testBox.onchange = (e) => { isTestMode = e.target.checked; };
  }
  const execTypeSel = $("trialExecType");
  const reasonSel = $("trialDeviationReason");
  if (execTypeSel && reasonSel) {
    execTypeSel.onchange = () => {
      reasonSel.style.display = execTypeSel.value === "deviate" ? "block" : "none";
    };
  }

  const logBtn = $("trialLogDecisionBtn");
  if (logBtn) {
    logBtn.onclick = async () => {
      await submitDecisionEvent();
    };
  }

  const finishBtn = $("trialFinishGameBtn");
  if (finishBtn) {
    finishBtn.onclick = async () => {
      await submitGameSummary();
    };
  }
}

async function submitDecisionEvent() {
  if (!window.cardApi?.logTrialDecision || !runtimeAdvice) return;
  const neural = runtimeAdvice.neural_advice || {};
  const isDeviated = $("trialExecType")?.value === "deviate";
  const reason = isDeviated ? ($("trialDeviationReason")?.value || "") : "";
  const sampledAction = neural.sampled_action || runtimeAdvice.recommended_action?.label || "";
  const actualAction = isDeviated ? `manual_deviate:${reason}` : sampledAction;

  currentTurnDecisionCount += 1;
  const payload = {
    game_id: currentTrialGameId,
    round_index: state?.session?.roundIndex ?? 1,
    turn_index: currentTurnDecisionCount,
    is_test: isTestMode,
    model_meta: {
      model_id: neural.model_id || "v21_rule_rollout_512",
      iteration: neural.iteration || 25,
      full_file_sha256: neural.full_file_sha256 || "",
      tensor_sha256: neural.tensor_sha256 || "",
    },
    state_snapshot: {
      fingerprint: runtimeAdvice.state_fingerprint,
      target_rank: runtimeAdvice.target_rank,
      my_seat: runtimeAdvice.my_seat,
      current_turn_seat: runtimeAdvice.current_turn_seat,
      hero_hand_view: runtimeAdvice.hero_hand_view,
      seats_snapshot: (runtimeAdvice.seat_analysis || []).map(s => ({
        seat: s.seat,
        status: s.status,
        shots_taken: s.shots_taken,
        hand_count: s.hand_count,
      })),
      latest_play: runtimeAdvice.table_factors?.latest_play_seat != null ? {
        seat: runtimeAdvice.table_factors.latest_play_seat,
        claim_rank: runtimeAdvice.table_factors.latest_play_claim_rank,
        count: runtimeAdvice.table_factors.latest_play_count,
      } : null,
    },
    interaction_target: runtimeAdvice.interaction_target || {},
    sampled_action: sampledAction,
    sampled_prob: neural.sampled_prob ?? 0.0,
    action_selection_probs: neural.action_selection_probs || [],
    human_actual_action: actualAction,
    is_deviated: isDeviated,
    deviation_reason: reason,
  };

  try {
    const res = await window.cardApi.logTrialDecision(payload);
    const btn = $("trialLogDecisionBtn");
    if (btn) {
      btn.textContent = `✓ 已记录第 ${currentTurnDecisionCount} 手决策`;
      setTimeout(() => { if (btn) btn.textContent = "记录本手决策 (JSONL)"; }, 2000);
    }
  } catch (err) {
    alert(`记录决策失败: ${err.message}`);
  }
}

async function submitGameSummary() {
  if (!window.cardApi?.logTrialSummary) return;
  const intervalVal = $("trialRankInterval")?.value || "1-1";
  const [minRank, maxRank] = intervalVal.split("-").map(Number);
  const neural = runtimeAdvice?.neural_advice || {};

  const payload = {
    game_id: currentTrialGameId,
    is_test: isTestMode,
    model_meta: {
      model_id: neural.model_id || "v21_rule_rollout_512",
      iteration: neural.iteration || 25,
      full_file_sha256: neural.full_file_sha256 || "",
      tensor_sha256: neural.tensor_sha256 || "",
    },
    min_rank: minRank,
    max_rank: maxRank,
    notes: isTestMode ? "测试对局" : "正式监督试用",
  };

  try {
    const res = await window.cardApi.logTrialSummary(payload);
    alert(`第 ${currentTrialGameId.slice(-6)} 局结算已记录！得分为: ${res.summary?.rank_result?.final_score}`);
    // 开启新整场
    currentTrialGameId = `game_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
    currentTurnDecisionCount = 0;
    if (window.cardApi.getTrialStats) {
      trialStats = await window.cardApi.getTrialStats();
    }
    renderTrialDrawer();
  } catch (err) {
    alert(`结算失败: ${err.message}`);
  }
}
