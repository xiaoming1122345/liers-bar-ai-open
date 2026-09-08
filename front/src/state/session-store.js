const fs = require("fs");
const path = require("path");
const { EventEmitter } = require("events");

const SEAT_COUNT = 4;
const BULLET_SLOTS = 5;
const DEFAULT_HAND_COUNT = 5;
const HISTORY_LIMIT = 120;
const GENERIC_ID_PATTERN = /^(player_\d+|me|hero)$/i;

function clone(value) {
  return structuredClone(value);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function nowStamp() {
  return new Date().toISOString();
}

function eventId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

function normalizeSeat(value) {
  const seat = Number(value);
  if (!Number.isFinite(seat)) {
    return 1;
  }
  return clamp(Math.round(seat), 1, SEAT_COUNT);
}

function normalizeText(value) {
  return String(value ?? "").trim();
}

function normalizePersistentPlayerId(value) {
  const id = normalizeText(value);
  if (!id || GENERIC_ID_PATTERN.test(id)) {
    return null;
  }
  return id;
}

function defaultBeliefs() {
  return {
    target: 50,
    nonTarget: 50,
    ghost: 10,
    wild: 10,
  };
}

function createSeatState(seat, previousSeat = null) {
  return {
    seat,
    playerId: previousSeat?.playerId ?? "",
    displayName: `玩家${seat}`,
    status: previousSeat?.status ?? "alive",
    shotsTaken: clamp(Number(previousSeat?.shotsTaken ?? 0), 0, BULLET_SLOTS),
    handCount: clamp(Number(previousSeat?.handCount ?? DEFAULT_HAND_COUNT), 0, DEFAULT_HAND_COUNT),
    lastAction: previousSeat?.lastAction ?? "",
    manualBeliefs: {
      ...defaultBeliefs(),
      ...(previousSeat?.manualBeliefs ?? {}),
    },
    metrics: null,
  };
}

function metricValue(model, snakeKey, camelKey = null, fallback = 0) {
  if (model == null || typeof model !== "object") {
    return fallback;
  }
  if (snakeKey in model) {
    return Number(model[snakeKey] ?? fallback);
  }
  if (camelKey && camelKey in model) {
    return Number(model[camelKey] ?? fallback);
  }
  return fallback;
}

function stringMetric(model, snakeKey, camelKey = null, fallback = "") {
  if (model == null || typeof model !== "object") {
    return fallback;
  }
  if (snakeKey in model) {
    return String(model[snakeKey] ?? fallback);
  }
  if (camelKey && camelKey in model) {
    return String(model[camelKey] ?? fallback);
  }
  return fallback;
}

function voteWinner(votes, fallback = "unknown") {
  let winner = fallback;
  let best = -1;
  for (const [key, count] of Object.entries(votes || {})) {
    if (count > best) {
      winner = key;
      best = count;
    }
  }
  return winner;
}

function weightedAverage(previousValue, previousWeight, nextValue, nextWeight) {
  const totalWeight = previousWeight + nextWeight;
  if (totalWeight <= 0) {
    return 0;
  }
  return (previousValue * previousWeight + nextValue * nextWeight) / totalWeight;
}

function summarizeProfile(feature) {
  if (!feature) {
    return null;
  }
  return {
    playerId: feature.playerId,
    sampleCount: feature.sampleCount,
    observedActions: feature.observedActions,
    publicStyleCluster: feature.publicStyleCluster,
    inferredProfileName: feature.inferredProfileName,
    pressureResponseLabel: feature.pressureResponseLabel,
    bluffRate: feature.bluffRate,
    challengeRate: feature.challengeRate,
    honestRate: feature.honestRate,
    ghostPlayRate: feature.ghostPlayRate,
    meanPlayCount: feature.meanPlayCount,
    variabilityScore: feature.variabilityScore,
    aggressionScore: feature.aggressionScore,
    postShotBluffDelta: feature.postShotBluffDelta,
    postShotChallengeDelta: feature.postShotChallengeDelta,
    sourceReports: feature.sourceReports,
    lastSeenAt: feature.lastSeenAt,
  };
}

class SessionStore {
  constructor({ dataDir }) {
    this.dataDir = dataDir;
    this.matchesDir = path.join(dataDir, "matches");
    this.featurePoolPath = path.join(dataDir, "feature-pool.json");
    this.sessionPath = path.join(dataDir, "session-cache.json");
    this.events = new EventEmitter();
    fs.mkdirSync(this.dataDir, { recursive: true });
    fs.mkdirSync(this.matchesDir, { recursive: true });

    this.featurePool = this._loadFeaturePool();
    this.state = this._loadSession();
    this.pastStates = [];
    this.futureStates = [];
  }

  subscribe(listener) {
    this.events.on("change", listener);
    return () => {
      this.events.off("change", listener);
    };
  }

  snapshot() {
    return clone({
      session: this.state,
      featurePool: this._featurePoolArray(),
    });
  }

  getState() {
    return this.snapshot();
  }

  setTrainingState(partial) {
    return this._commit((draft) => {
      draft.training = {
        ...draft.training,
        ...partial,
      };
    });
  }

  setSelectedReportPath(reportPath) {
    return this._commit((draft) => {
      draft.selectedReportPath = normalizeText(reportPath);
    });
  }

  setOverlayState({ visible, topmost } = {}) {
    return this._commit((draft) => {
      if (typeof visible === "boolean") {
        draft.overlayVisible = visible;
      }
      if (typeof topmost === "boolean") {
        draft.overlayTopmost = topmost;
      }
    });
  }

  setMySeat(seat) {
    return this._commit((draft) => {
      draft.mySeat = normalizeSeat(seat);
    });
  }

  setCurrentTurnSeat(seat) {
    return this._commit((draft) => {
      draft.currentTurnSeat = normalizeSeat(seat);
    });
  }

  setHeroHandView(patch) {
    return this._commit((draft) => {
      draft.heroHandView = {
        ...draft.heroHandView,
        ...patch,
      };
    });
  }

  updateSessionMeta(patch) {
    return this._commit((draft) => {
      if (patch.targetRank != null) {
        draft.targetRank = String(patch.targetRank);
      }
      if (patch.stackCount != null) {
        draft.stackCount = clamp(Number(patch.stackCount) || 0, 0, 99);
      }
      if (patch.notes != null) {
        draft.notes = String(patch.notes);
      }
    });
  }

  updateSeat(seatNumber, patch) {
    return this._commit((draft) => {
      const seat = this._seat(draft, seatNumber);
      if (!seat) {
        return;
      }
      if (patch.status != null) {
        seat.status = String(patch.status);
      }
      if (patch.shotsTaken != null) {
        seat.shotsTaken = clamp(Number(patch.shotsTaken) || 0, 0, BULLET_SLOTS);
      }
      if (patch.handCount != null) {
        seat.handCount = clamp(Number(patch.handCount) || 0, 0, DEFAULT_HAND_COUNT);
      }
      if (patch.lastAction != null) {
        seat.lastAction = String(patch.lastAction);
      }
      if (patch.manualBeliefs != null) {
        seat.manualBeliefs = {
          ...seat.manualBeliefs,
          ...patch.manualBeliefs,
        };
      }
    });
  }

  setSeatPlayerId(seatNumber, playerId) {
    return this._commit((draft) => {
      const seat = this._seat(draft, seatNumber);
      if (!seat) {
        return;
      }
      seat.playerId = normalizeText(playerId);
    });
  }

  appendEvent(payload) {
    return this._commit((draft) => {
      const actorSeat = normalizeSeat(payload.seat ?? draft.currentTurnSeat);
      const targetSeat = payload.targetSeat ? normalizeSeat(payload.targetSeat) : null;
      const actor = this._seat(draft, actorSeat);
      const target = targetSeat ? this._seat(draft, targetSeat) : actor;
      const type = String(payload.type || "note");
      const count = clamp(Number(payload.count) || 0, 0, 3);
      const claimRank = String(payload.claimRank || draft.targetRank || "A");
      const composition = normalizeText(payload.composition);
      const note = normalizeText(payload.note);
      const nextTurnSeat = payload.nextTurnSeat ? normalizeSeat(payload.nextTurnSeat) : null;
      let summary = note || type;

      if (!actor) {
        return;
      }

      if (type === "play") {
        draft.targetRank = claimRank;
        draft.stackCount = clamp(draft.stackCount + count, 0, 99);
        actor.lastAction = composition
          ? `出${count} ${claimRank} ${composition}`
          : `出${count} ${claimRank}`;
        actor.handCount = clamp(actor.handCount - count, 0, DEFAULT_HAND_COUNT);
        draft.currentTurnSeat = nextTurnSeat || this._nextActiveSeat(draft, actorSeat);
        summary = `${actor.displayName} 出${count}张 ${claimRank}${composition ? ` · ${composition}` : ""}`;
      } else if (type === "challenge") {
        actor.lastAction = target ? `质疑 ${target.displayName}` : "质疑";
        draft.stackCount = 0;
        draft.currentTurnSeat = nextTurnSeat || this._nextActiveSeat(draft, actorSeat);
        summary = target
          ? `${actor.displayName} 质疑 ${target.displayName}`
          : `${actor.displayName} 质疑`;
      } else if (type === "shot") {
        const affected = target || actor;
        affected.shotsTaken = clamp(affected.shotsTaken + 1, 0, BULLET_SLOTS);
        affected.lastAction = `开枪 ${affected.shotsTaken}/${BULLET_SLOTS}`;
        draft.currentTurnSeat = nextTurnSeat || draft.currentTurnSeat;
        summary = `${affected.displayName} 开枪 ${affected.shotsTaken}/${BULLET_SLOTS}`;
      } else if (type === "status") {
        const affected = target || actor;
        affected.status = String(payload.statusValue || affected.status || "alive");
        affected.lastAction = `状态 ${affected.status}`;
        if (affected.status !== "alive" && draft.currentTurnSeat === affected.seat) {
          draft.currentTurnSeat = this._nextActiveSeat(draft, affected.seat);
        }
        summary = `${affected.displayName} -> ${affected.status}`;
      } else if (type === "hand") {
        const affected = target || actor;
        affected.handCount = clamp(Number(payload.handCount) || 0, 0, DEFAULT_HAND_COUNT);
        affected.lastAction = `手牌 ${affected.handCount}`;
        summary = `${affected.displayName} 手牌 ${affected.handCount}`;
      } else {
        actor.lastAction = note || actor.lastAction;
        summary = note || `${actor.displayName} 备注`;
      }

      draft.history.unshift({
        id: eventId(),
        type,
        seat: actorSeat,
        targetSeat,
        count,
        claimRank,
        composition,
        note,
        summary,
        createdAt: nowStamp(),
      });
      if (draft.history.length > HISTORY_LIMIT) {
        draft.history = draft.history.slice(0, HISTORY_LIMIT);
      }
      draft.undoneHistory = [];
    });
  }

  undo() {
    if (this.pastStates.length === 0) {
      return this.snapshot();
    }
    this.futureStates.push(clone(this.state));
    this.state = this._decorateState(this.pastStates.pop());
    this._persistSession();
    this._emit();
    return this.snapshot();
  }

  redo() {
    if (this.futureStates.length === 0) {
      return this.snapshot();
    }
    this.pastStates.push(clone(this.state));
    this.state = this._decorateState(this.futureStates.pop());
    this._persistSession();
    this._emit();
    return this.snapshot();
  }

  nextGame() {
    if (this.state.history.length > 0) {
      this._saveMatchRecord(this.state);
    }
    const previous = clone(this.state);
    this.pastStates = [];
    this.futureStates = [];
    this.state = this._createInitialState(previous);
    this._persistSession();
    this._emit();
    return this.snapshot();
  }

  importReport(reportPath, reportPayload) {
    const report = reportPayload?.summary ? reportPayload : reportPayload?.report;
    const seatModels = report?.summary?.seat_models || report?.summary?.seatModels || [];
    let imported = 0;
    let skipped = 0;
    for (const model of seatModels) {
      const rawId = model.player_id ?? model.playerId ?? "";
      const playerId = normalizePersistentPlayerId(rawId);
      if (!playerId) {
        skipped += 1;
        continue;
      }
      this.featurePool[playerId] = this._mergeFeature(this.featurePool[playerId], playerId, model, reportPath);
      imported += 1;
    }
    this._persistFeaturePool();
    this.state = this._decorateState(this.state);
    this._persistSession();
    this._emit();
    return {
      imported,
      skipped,
      knownIds: this._featurePoolArray().length,
    };
  }

  deleteFeature(playerId) {
    const persistentId = normalizePersistentPlayerId(playerId);
    if (!persistentId || !this.featurePool[persistentId]) {
      return {
        deleted: false,
        knownIds: this._featurePoolArray().length,
      };
    }
    delete this.featurePool[persistentId];
    this._persistFeaturePool();
    this.state = this._decorateState(this.state);
    this._persistSession();
    this._emit();
    return {
      deleted: true,
      knownIds: this._featurePoolArray().length,
    };
  }

  _commit(mutator) {
    this.pastStates.push(clone(this.state));
    if (this.pastStates.length > HISTORY_LIMIT) {
      this.pastStates.shift();
    }
    this.futureStates = [];
    const draft = clone(this.state);
    mutator(draft);
    this.state = this._decorateState(draft);
    this._persistSession();
    this._emit();
    return this.snapshot();
  }

  _emit() {
    this.events.emit("change", this.snapshot());
  }

  _featurePoolArray() {
    return Object.values(this.featurePool)
      .map((feature) => summarizeProfile(feature))
      .sort((left, right) => {
        if (right.sampleCount !== left.sampleCount) {
          return right.sampleCount - left.sampleCount;
        }
        return left.playerId.localeCompare(right.playerId);
      });
  }

  _loadFeaturePool() {
    try {
      const payload = JSON.parse(fs.readFileSync(this.featurePoolPath, "utf-8"));
      return payload.featurePool || {};
    } catch {
      return {};
    }
  }

  _persistFeaturePool() {
    fs.writeFileSync(
      this.featurePoolPath,
      JSON.stringify({ featurePool: this.featurePool }, null, 2),
      "utf-8",
    );
  }

  _loadSession() {
    try {
      const payload = JSON.parse(fs.readFileSync(this.sessionPath, "utf-8"));
      return this._decorateState(payload.session || this._createInitialState());
    } catch {
      return this._createInitialState();
    }
  }

  _persistSession() {
    fs.writeFileSync(
      this.sessionPath,
      JSON.stringify({ session: this.state }, null, 2),
      "utf-8",
    );
  }

  _createInitialState(previous = null) {
    const next = {
      sessionId: `session-${Date.now()}`,
      gameIndex: previous ? Number(previous.gameIndex || 0) + 1 : 1,
      mySeat: previous ? normalizeSeat(previous.mySeat) : 1,
      currentTurnSeat: 1,
      targetRank: "A",
      stackCount: 0,
      overlayVisible: previous?.overlayVisible ?? false,
      overlayTopmost: previous?.overlayTopmost ?? true,
      selectedReportPath: previous?.selectedReportPath ?? "",
      notes: "",
      heroHandView: {
        targetCount: clamp(Number(previous?.heroHandView?.targetCount) || 0, 0, DEFAULT_HAND_COUNT),
        nonTargetCount: clamp(Number(previous?.heroHandView?.nonTargetCount) || 0, 0, DEFAULT_HAND_COUNT),
        ghostCount: clamp(Number(previous?.heroHandView?.ghostCount) || 0, 0, DEFAULT_HAND_COUNT),
        wildCount: clamp(Number(previous?.heroHandView?.wildCount) || 0, 0, DEFAULT_HAND_COUNT),
      },
      training: {
        status: "idle",
        command: "",
        output: "",
        lastExitCode: null,
        lastRunAt: null,
      },
      seats: [],
      history: [],
      undoneHistory: [],
    };

    for (let seat = 1; seat <= SEAT_COUNT; seat += 1) {
      const previousSeat = previous?.seats?.find((item) => item.seat === seat) || null;
      next.seats.push(createSeatState(seat, previousSeat));
    }

    return this._decorateState(next);
  }

  _decorateState(state) {
    const next = clone(state);
    next.mySeat = normalizeSeat(next.mySeat);
    next.currentTurnSeat = normalizeSeat(next.currentTurnSeat);
    next.targetRank = String(next.targetRank || "A");
    next.stackCount = clamp(Number(next.stackCount) || 0, 0, 99);
    next.heroHandView = {
      targetCount: clamp(Number(next.heroHandView?.targetCount) || 0, 0, DEFAULT_HAND_COUNT),
      nonTargetCount: clamp(Number(next.heroHandView?.nonTargetCount) || 0, 0, DEFAULT_HAND_COUNT),
      ghostCount: clamp(Number(next.heroHandView?.ghostCount) || 0, 0, DEFAULT_HAND_COUNT),
      wildCount: clamp(Number(next.heroHandView?.wildCount) || 0, 0, DEFAULT_HAND_COUNT),
    };
    next.overlayVisible = Boolean(next.overlayVisible);
    next.overlayTopmost = Boolean(next.overlayTopmost);
    next.training = {
      status: next.training?.status || "idle",
      command: next.training?.command || "",
      output: next.training?.output || "",
      lastExitCode: next.training?.lastExitCode ?? null,
      lastRunAt: next.training?.lastRunAt || null,
    };
    next.seats = next.seats.map((seatState, index) => {
      const seat = createSeatState(index + 1, seatState);
      seat.playerId = normalizeText(seat.playerId);
      seat.displayName = seat.seat === next.mySeat
        ? "我"
        : seat.playerId || `玩家${seat.seat}`;
      seat.manualBeliefs = {
        target: clamp(Number(seat.manualBeliefs?.target) || 0, 0, 100),
        nonTarget: clamp(Number(seat.manualBeliefs?.nonTarget) || 0, 0, 100),
        ghost: clamp(Number(seat.manualBeliefs?.ghost) || 0, 0, 100),
        wild: clamp(Number(seat.manualBeliefs?.wild) || 0, 0, 100),
      };

      const persistentId = normalizePersistentPlayerId(seat.playerId);
      seat.metrics = persistentId ? summarizeProfile(this.featurePool[persistentId]) : null;
      seat.safeSlotsLeft = BULLET_SLOTS - seat.shotsTaken;
      return seat;
    });
    next.history = Array.isArray(next.history) ? next.history : [];
    next.undoneHistory = Array.isArray(next.undoneHistory) ? next.undoneHistory : [];

    const currentSeat = this._seat(next, next.currentTurnSeat);
    if (!currentSeat || currentSeat.status !== "alive") {
      next.currentTurnSeat = this._nextActiveSeat(next, next.currentTurnSeat);
    }
    return next;
  }

  _seat(state, seatNumber) {
    return state.seats.find((item) => item.seat === normalizeSeat(seatNumber)) || null;
  }

  _nextActiveSeat(state, fromSeat) {
    const start = normalizeSeat(fromSeat);
    for (let offset = 1; offset <= SEAT_COUNT; offset += 1) {
      const seat = ((start - 1 + offset) % SEAT_COUNT) + 1;
      const candidate = this._seat(state, seat);
      if (candidate && candidate.status === "alive") {
        return seat;
      }
    }
    return start;
  }

  _saveMatchRecord(session) {
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filePath = path.join(this.matchesDir, `${stamp}-game-${session.gameIndex}.json`);
    fs.writeFileSync(
      filePath,
      JSON.stringify(
        {
          savedAt: nowStamp(),
          session,
        },
        null,
        2,
      ),
      "utf-8",
    );

    // 历史推理与对局数据容量管理：最多保留最近 50 场对战记录，防止无限增长
    try {
      const files = fs.readdirSync(this.matchesDir)
        .filter((f) => f.endsWith(".json"))
        .sort();
      const MAX_MATCH_HISTORY = 50;
      if (files.length > MAX_MATCH_HISTORY) {
        const toDelete = files.slice(0, files.length - MAX_MATCH_HISTORY);
        for (const oldFile of toDelete) {
          try {
            fs.unlinkSync(path.join(this.matchesDir, oldFile));
          } catch (_) {}
        }
      }
    } catch (_) {}
  }

  _mergeFeature(existing, playerId, model, reportPath) {
    const nextWeight = Math.max(1, metricValue(model, "observed_actions", "observedActions", 1));
    const previousWeight = existing?.weight || 0;
    const sampleCount = Number(existing?.sampleCount || 0) + 1;
    const publicStyleCluster = stringMetric(model, "public_style_cluster", "publicStyleCluster", "unknown");
    const inferredProfileName = stringMetric(model, "inferred_profile_name", "inferredProfileName", "unknown");
    const pressureResponseLabel = stringMetric(model, "pressure_response_label", "pressureResponseLabel", "unknown");

    const next = {
      playerId,
      sampleCount,
      weight: previousWeight + nextWeight,
      observedActions: Number(existing?.observedActions || 0) + nextWeight,
      publicStyleVotes: {
        ...(existing?.publicStyleVotes || {}),
      },
      profileVotes: {
        ...(existing?.profileVotes || {}),
      },
      pressureVotes: {
        ...(existing?.pressureVotes || {}),
      },
      sourceReports: Array.from(new Set([...(existing?.sourceReports || []), reportPath])),
      lastSeenAt: nowStamp(),
      bluffRate: weightedAverage(existing?.bluffRate || 0, previousWeight, metricValue(model, "bluff_rate", "bluffRate", 0), nextWeight),
      challengeRate: weightedAverage(existing?.challengeRate || 0, previousWeight, metricValue(model, "challenge_rate", "challengeRate", 0), nextWeight),
      honestRate: weightedAverage(existing?.honestRate || 0, previousWeight, metricValue(model, "honest_rate", "honestRate", 0), nextWeight),
      ghostPlayRate: weightedAverage(existing?.ghostPlayRate || 0, previousWeight, metricValue(model, "ghost_play_rate", "ghostPlayRate", 0), nextWeight),
      meanPlayCount: weightedAverage(existing?.meanPlayCount || 0, previousWeight, metricValue(model, "mean_play_count", "meanPlayCount", 0), nextWeight),
      variabilityScore: weightedAverage(existing?.variabilityScore || 0, previousWeight, metricValue(model, "variability_score", "variabilityScore", 0), nextWeight),
      aggressionScore: weightedAverage(existing?.aggressionScore || 0, previousWeight, metricValue(model, "aggression_score", "aggressionScore", 0), nextWeight),
      postShotBluffDelta: weightedAverage(existing?.postShotBluffDelta || 0, previousWeight, metricValue(model, "post_shot_bluff_delta", "postShotBluffDelta", 0), nextWeight),
      postShotChallengeDelta: weightedAverage(existing?.postShotChallengeDelta || 0, previousWeight, metricValue(model, "post_shot_challenge_delta", "postShotChallengeDelta", 0), nextWeight),
    };

    next.publicStyleVotes[publicStyleCluster] = (next.publicStyleVotes[publicStyleCluster] || 0) + 1;
    next.profileVotes[inferredProfileName] = (next.profileVotes[inferredProfileName] || 0) + 1;
    next.pressureVotes[pressureResponseLabel] = (next.pressureVotes[pressureResponseLabel] || 0) + 1;

    next.publicStyleCluster = voteWinner(next.publicStyleVotes, publicStyleCluster);
    next.inferredProfileName = voteWinner(next.profileVotes, inferredProfileName);
    next.pressureResponseLabel = voteWinner(next.pressureVotes, pressureResponseLabel);
    return next;
  }
}

module.exports = {
  SessionStore,
  normalizePersistentPlayerId,
};
