use cascade_data::{
    Action, GameState, HeuristicEvaluator, Mcts, MctsConfig, NeuralEvaluator,
    NeuralEvaluatorConfig, Player, PositionEvaluator,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Debug, Clone)]
enum CliEvaluator {
    Heuristic(HeuristicEvaluator),
    Neural(NeuralEvaluator),
}

impl PositionEvaluator for CliEvaluator {
    fn evaluate(
        &self,
        state: &cascade_data::GameState,
    ) -> Result<cascade_data::Evaluation, cascade_data::DataCollectionError> {
        match self {
            Self::Heuristic(evaluator) => evaluator.evaluate(state),
            Self::Neural(evaluator) => evaluator.evaluate(state),
        }
    }
}

fn load_evaluator(model_path: &Option<PathBuf>) -> Result<CliEvaluator, Box<dyn Error>> {
    match model_path {
        Some(path) => Ok(CliEvaluator::Neural(NeuralEvaluator::load(
            path,
            NeuralEvaluatorConfig::default(),
        )?)),
        None => Ok(CliEvaluator::Heuristic(HeuristicEvaluator::default())),
    }
}

#[derive(Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Command {
    Init {
        model: Option<String>,
        simulations: Option<u32>,
        c_puct: Option<f32>,
    },
    State,
    Step {
        action: Action,
    },
    AiStep,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Response {
    Ok { status: String },
    Error { status: String, message: String },
    State {
        status: String,
        state: cascade_data::GameStateSnapshot,
        turn: Player,
        legal_actions: Vec<Action>,
        is_terminal: bool,
    },
    ActionTaken {
        status: String,
        action: Action,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    let mut state = GameState::new();
    let mut search: Option<Mcts<CliEvaluator>> = None;

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let cmd: Result<Command, _> = serde_json::from_str(&line);
        let resp = match cmd {
            Ok(Command::Init { model, simulations, c_puct }) => {
                let p = model.filter(|s| !s.is_empty()).map(PathBuf::from);
                match load_evaluator(&p) {
                    Ok(eval) => {
                        let sim_count = simulations.unwrap_or(50);
                        let puct = c_puct.unwrap_or(1.35);
                        search = Some(Mcts::new(
                            MctsConfig::default().with_simulations(sim_count).with_c_puct(puct),
                            eval,
                        ));
                        state = GameState::new();
                        Response::Ok {
                            status: "ok".to_string(),
                        }
                    }
                    Err(e) => Response::Error {
                        status: "error".to_string(),
                        message: format!("Failed to load model: {}", e),
                    },
                }
            }
            Ok(Command::State) => {
                let legal_actions = state.legal_actions();
                Response::State {
                    status: "ok".to_string(),
                    state: state.to_snapshot(),
                    turn: state.turn(),
                    legal_actions,
                    is_terminal: state.terminal_status().is_some(),
                }
            }
            Ok(Command::Step { action }) => {
                if state.legal_actions().contains(&action) {
                    if let Err(e) = state.apply_action(action.clone()) {
                        Response::Error {
                            status: "error".to_string(),
                            message: format!("Invalid move state apply: {:?}", e),
                        }
                    } else {
                        Response::ActionTaken {
                            status: "ok".to_string(),
                            action,
                        }
                    }
                } else {
                    Response::Error {
                        status: "error".to_string(),
                        message: format!("Action is not in legal actions: {:?}", action),
                    }
                }
            }
            Ok(Command::AiStep) => {
                if state.terminal_status().is_some() {
                    Response::Error {
                        status: "error".to_string(),
                        message: "Game is already over".to_string(),
                    }
                } else if let Some(m) = &search {
                    match m.search(&state) {
                        Ok(result) => {
                            let dist = result.action_statistics;
                            if dist.is_empty() {
                                Response::Error {
                                    status: "error".to_string(),
                                    message: "No legal moves found by MCTS".to_string(),
                                }
                            } else {
                                // Action statistics is already sorted by visits descending
                                let best_action = dist[0].action.clone();
                                let _ = state.apply_action(best_action.clone());
                                Response::ActionTaken {
                                    status: "ok".to_string(),
                                    action: best_action,
                                }
                            }
                        }
                        Err(e) => Response::Error {
                            status: "error".to_string(),
                            message: format!("MCTS search failed: {:?}", e),
                        }
                    }
                } else {
                    Response::Error {
                        status: "error".to_string(),
                        message: "MCTS not initialized. Run init first.".to_string(),
                    }
                }
            }
            Err(e) => Response::Error {
                status: "error".to_string(),
                message: format!("Failed to parse command: {}", e),
            },
        };

        let out = serde_json::to_string(&resp).unwrap();
        writeln!(stdout, "{}", out)?;
        stdout.flush()?;
    }
    Ok(())
}
