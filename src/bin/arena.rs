use cascade_data::{
    run_arena, run_parallel_arena, ArenaConfig, HeuristicEvaluator, Mcts, MctsConfig,
    NeuralEvaluator, NeuralEvaluatorConfig, ParallelArenaConfig, PositionEvaluator,
};
use serde_json::json;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone)]
struct CliOptions {
    games: u32,
    simulations: u32,
    c_puct: f32,
    seed: u64,
    opening_temperature_plies: u16,
    opening_temperature: f32,
    log_every_plies: u16,
    slow_search_ms: u64,
    quiet_progress: bool,
    threads: usize,
    player_a_name: String,
    player_b_name: String,
    player_a_model: Option<PathBuf>,
    player_b_model: Option<PathBuf>,
    report_json: Option<PathBuf>,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            games: 12,
            simulations: 32,
            c_puct: 1.35,
            seed: 20260420,
            opening_temperature_plies: 8,
            opening_temperature: 1.0,
            log_every_plies: 8,
            slow_search_ms: 1000,
            quiet_progress: false,
            threads: std::thread::available_parallelism()
                .map(|parallelism| parallelism.get())
                .unwrap_or(1),
            player_a_name: "player_a".to_string(),
            player_b_name: "player_b".to_string(),
            player_a_model: None,
            player_b_model: None,
            report_json: None,
        }
    }
}

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

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_args(env::args().skip(1))?;

    if let Some(report_json) = &options.report_json {
        if let Some(parent) = report_json.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
    }

    let player_a_evaluator = load_evaluator(&options.player_a_model)?;
    let player_b_evaluator = load_evaluator(&options.player_b_model)?;
    let player_a_search = Mcts::new(
        MctsConfig::default()
            .with_simulations(options.simulations)
            .with_c_puct(options.c_puct),
        player_a_evaluator,
    );
    let player_b_search = Mcts::new(
        MctsConfig::default()
            .with_simulations(options.simulations)
            .with_c_puct(options.c_puct),
        player_b_evaluator,
    );
    let arena_config = ArenaConfig::default()
        .with_games(options.games)
        .with_seed(options.seed)
        .with_opening_temperature_plies(options.opening_temperature_plies)
        .with_opening_temperature(options.opening_temperature)
        .with_progress_log_every_plies(if options.quiet_progress { 0 } else { options.log_every_plies })
        .with_slow_search_log_millis(if options.quiet_progress { 0 } else { options.slow_search_ms })
        .with_log_game_summaries(!options.quiet_progress);

    let started_at = Instant::now();
    let summary = if options.threads > 1 {
        run_parallel_arena(
            arena_config.clone(),
            ParallelArenaConfig::default().with_threads(options.threads),
            options.player_a_name.clone(),
            &player_a_search,
            options.player_b_name.clone(),
            &player_b_search,
        )?
    } else {
        run_arena(
            arena_config.clone(),
            options.player_a_name.clone(),
            &player_a_search,
            options.player_b_name.clone(),
            &player_b_search,
        )?
    };
    let elapsed_ms = started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;

    if let Some(report_json) = &options.report_json {
        let report = json!({
            "config": arena_config,
            "simulations": options.simulations,
            "c_puct": options.c_puct,
            "elapsed_ms": elapsed_ms,
            "player_a_model": options.player_a_model.as_ref().map(|path| path.display().to_string()),
            "player_b_model": options.player_b_model.as_ref().map(|path| path.display().to_string()),
            "summary": summary,
        });
        fs::write(report_json, serde_json::to_vec_pretty(&report)?)?;
    }

    println!(
        "player_a={} player_b={} games={} player_a_wins={} player_b_wins={} draws={} total_plies={} player_a_score={:.4} player_b_score={:.4} player_a_avg_ms={:.4} player_b_avg_ms={:.4} player_a_nps={:.2} player_b_nps={:.2} elapsed_ms={} threads={}{}",
        options.player_a_name,
        options.player_b_name,
        summary.games_played,
        summary.player_a.wins,
        summary.player_b.wins,
        summary.draws,
        summary.total_plies,
        summary.player_a_score(),
        summary.player_b_score(),
        summary.player_a_avg_search_millis(),
        summary.player_b_avg_search_millis(),
        summary.player_a_nodes_per_second(),
        summary.player_b_nodes_per_second(),
        elapsed_ms,
        options.threads,
        options
            .report_json
            .as_ref()
            .map(|path| format!(" report_json={}", path.display()))
            .unwrap_or_default(),
    );

    Ok(())
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

fn parse_args<I>(args: I) -> Result<CliOptions, Box<dyn Error>>
where
    I: IntoIterator<Item = String>,
{
    let mut options = CliOptions::default();
    let mut args = args.into_iter();

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--games" => options.games = parse_value(&argument, args.next())?,
            "--simulations" => options.simulations = parse_value(&argument, args.next())?,
            "--c-puct" => options.c_puct = parse_value(&argument, args.next())?,
            "--seed" => options.seed = parse_value(&argument, args.next())?,
            "--opening-temperature-plies" => {
                options.opening_temperature_plies = parse_value(&argument, args.next())?
            }
            "--opening-temperature" => {
                options.opening_temperature = parse_value(&argument, args.next())?
            }
            "--log-every-plies" => options.log_every_plies = parse_value(&argument, args.next())?,
            "--slow-search-ms" => options.slow_search_ms = parse_value(&argument, args.next())?,
            "--quiet-progress" => {
                options.quiet_progress = true;
            }
            "--threads" => options.threads = parse_value(&argument, args.next())?,
            "--player-a-name" => {
                options.player_a_name = args.next().ok_or_else(|| format!("missing value for {argument}"))?;
            }
            "--player-b-name" => {
                options.player_b_name = args.next().ok_or_else(|| format!("missing value for {argument}"))?;
            }
            "--player-a-model" => {
                let value = args.next().ok_or_else(|| format!("missing value for {argument}"))?;
                options.player_a_model = Some(PathBuf::from(value));
            }
            "--player-b-model" => {
                let value = args.next().ok_or_else(|| format!("missing value for {argument}"))?;
                options.player_b_model = Some(PathBuf::from(value));
            }
            "--report-json" => {
                let value = args.next().ok_or_else(|| format!("missing value for {argument}"))?;
                options.report_json = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }

    Ok(options)
}

fn parse_value<T>(flag: &str, value: Option<String>) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    let value = value.ok_or_else(|| format!("missing value for {flag}"))?;
    Ok(value.parse()?)
}

fn print_help() {
    println!("Usage: cargo run --bin arena -- [options]");
    println!("  --games <n>                      Total games to play; color alternates every game");
    println!("  --simulations <n>               MCTS simulations per move for both players");
    println!("  --c-puct <value>                Exploration constant for both players");
    println!("  --seed <n>                      Seed used for paired stochastic openings");
    println!("  --opening-temperature-plies <n> Sample from search policy for the first n plies");
    println!("  --opening-temperature <value>   Sampling temperature during the opening window");
    println!("  --log-every-plies <n>           Keep aggregated progress logging enabled; set 0 to disable");
    println!("  --slow-search-ms <n>            Count searches above n ms inside aggregated progress summaries");
    println!("  --quiet-progress                Disable aggregated progress logging");
    println!("  --threads <n>                   Number of arena worker threads");
    println!("  --player-a-name <name>          Label for player A");
    println!("  --player-b-name <name>          Label for player B");
    println!("  --player-a-model <path>         Rust-exported manifest for player A; omit for heuristic");
    println!("  --player-b-model <path>         Rust-exported manifest for player B; omit for heuristic");
    println!("  --report-json <path>            Optional JSON report path");
}
