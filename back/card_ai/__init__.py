import sys
sys.modules["_wmi"] = None

import multiprocessing as _mp
if _mp.current_process().name != "MainProcess":
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from .artifacts import ArtifactStore, SolverArtifact
from .belief import BeliefTracker
from .confidence import build_confidence_report
from .config import BeliefConfig, DeepCFRConfig, GameRules, ScoreRules, SolverConfig, TrainingRunConfig
from .deep_cfr import DeepCFRTrainer, DeepCFRTrainingResult
from .engine import SurvivalGameEngine
from .evaluation import MatchupEvaluator
from .features import FeatureEncoder, FEATURE_DIM
from .identities import build_default_identities
from .networks import build_advantage_net, build_strategy_net
from .neural_policy import NeuralPolicy
from .opponent_modeling import OnlineOpponentModeler, OpponentStyleProfile
from .opponents import default_opponent_profiles
from .policies import SolverPolicy
from .pomdp import POMDPParticleFilter, POMDPInferenceResult
from .reasoning import LookaheadEngine, ActionLookaheadEvaluation
from .response_model import StyleResponseModel, StyleResponseTrainer
from .rewards import TerminalRewardModel
from .runtime_advice import RuntimeAdvisor
from .self_play import RandomPolicy, SelfPlayRunner
from .solvers import OpponentAwareAdaptiveSolver, TabularMCCFRTrainer, UniformRandomSolver
from .training import TrainingBackend

__all__ = [
    "ActionLookaheadEvaluation",
    "ArtifactStore",
    "BeliefConfig",
    "BeliefTracker",
    "DeepCFRConfig",
    "DeepCFRTrainer",
    "DeepCFRTrainingResult",
    "FeatureEncoder",
    "FEATURE_DIM",
    "GameRules",
    "LookaheadEngine",
    "MatchupEvaluator",
    "NeuralPolicy",
    "OnlineOpponentModeler",
    "OpponentAwareAdaptiveSolver",
    "OpponentStyleProfile",
    "POMDPInferenceResult",
    "POMDPParticleFilter",
    "RandomPolicy",
    "ScoreRules",
    "SelfPlayRunner",
    "SolverConfig",
    "SolverPolicy",
    "SolverArtifact",
    "StyleResponseModel",
    "StyleResponseTrainer",
    "SurvivalGameEngine",
    "RuntimeAdvisor",
    "TabularMCCFRTrainer",
    "TerminalRewardModel",
    "TrainingBackend",
    "TrainingRunConfig",
    "UniformRandomSolver",
    "build_advantage_net",
    "build_default_identities",
    "build_confidence_report",
    "build_strategy_net",
    "default_opponent_profiles",
]

