use cascade_data::{
    run_parallel_self_play, CollectorConfig, HeuristicEvaluator, JsonlDatasetSink, Mcts,
    MctsConfig, NeuralEvaluator, NeuralEvaluatorConfig, ParallelSelfPlayConfig, PositionEvaluator,
    SelfPlayConfig, SelfPlayRunner, TrainingDataCollector,
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
    temperature_plies: u16,
    temperature: f32,
    root_dirichlet_alpha: f32,
    root_exploration_fraction: f32,
    log_every_plies: u16,
    slow_search_ms: u64,
    quiet_progress: bool,
    threads: usize,
    seed: u64,
    output: PathBuf,
    student_model: Option<PathBuf>,
    model_policy_temperature: f32,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            games: 1,
            simulations: 96,
            c_puct: 1.35,
            temperature_plies: 12,
            temperature: 1.0,
            root_dirichlet_alpha: 10.0,
            root_exploration_fraction: 0.25,
            log_every_plies: 8,
            slow_search_ms: 1000,
            quiet_progress: false,
            threads: std::thread::available_parallelism()
                .map(|parallelism| parallelism.get())
                .unwrap_or(1),
            seed: 20260420,
            output: PathBuf::from("artifacts/local_selfplay.jsonl"),
            student_model: None,
            model_policy_temperature: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
enum CliEvaluator {
    Heuristic(HeuristicEvaluator),
    Neural(NeuralEvaluator),
}

impl PositionEvaluator for CliEvaluator {
    fn evaluate(&self, state: &cascade_data::GameState) -> Result<cascade_data::Evaluation, cascade_data::DataCollectionError> {
        match self {
            Self::Heuristic(evaluator) => evaluator.evaluate(state),
            Self::Neural(evaluator) => evaluator.evaluate(state),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_args(env::args().skip(1))?;

    if let Some(parent) = options.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let sink = JsonlDatasetSink::create(&options.output)?;
    let collector = TrainingDataCollector::new(
        CollectorConfig::new("local-teacher")
            .with_flush_interval(32)
            .with_metadata("generator", json!("local-selfplay-bin")),
        sink,
    );
    let evaluator = if let Some(model_path) = &options.student_model {
        CliEvaluator::Neural(NeuralEvaluator::load(
            model_path,
            NeuralEvaluatorConfig::default().with_policy_temperature(options.model_policy_temperature),
        )?)
    } else {
        CliEvaluator::Heuristic(HeuristicEvaluator::default())
    };
    let search = Mcts::new(
        MctsConfig::default()
            .with_simulations(options.simulations)
            .with_c_puct(options.c_puct),
        evaluator,
    );
    let selfplay_config = SelfPlayConfig::default()
        .with_games(options.games)
        .with_sampling_temperature(options.temperature)
        .with_temperature_drop_ply(options.temperature_plies)
        .with_root_dirichlet_alpha(options.root_dirichlet_alpha)
        .with_root_exploration_fraction(options.root_exploration_fraction)
        .with_progress_log_every_plies(if options.quiet_progress { 0 } else { options.log_every_plies })
        .with_slow_search_log_millis(if options.quiet_progress { 0 } else { options.slow_search_ms })
        .with_log_game_summaries(!options.quiet_progress)
        .with_seed(options.seed);

    let started_at = Instant::now();
    let summary = if options.threads > 1 {
        let (summary, _collector) = run_parallel_self_play(
            selfplay_config,
            ParallelSelfPlayConfig::default().with_threads(options.threads),
            search,
            collector,
        )?;
        summary
    } else {
        let mut runner = SelfPlayRunner::new(selfplay_config, search, collector);
        runner.play_games()?
    };
    let elapsed = started_at.elapsed();

    println!(
        "games={} red_wins={} blue_wins={} draws={} samples={} total_plies={} elapsed_ms={} threads={} output={}",
        summary.games_played,
        summary.red_wins,
        summary.blue_wins,
        summary.draws,
        summary.samples_written,
        summary.total_plies,
        elapsed.as_millis(),
        options.threads,
        options.output.display()
    );

    Ok(())
}

fn parse_args<I>(args: I) -> Result<CliOptions, Box<dyn Error>>
where
    I: IntoIterator<Item = String>,
{
    let mut options = CliOptions::default();
    let mut args = args.into_iter();

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--games" => {
                options.games = parse_value(&argument, args.next())?;
            }
            "--simulations" => {
                options.simulations = parse_value(&argument, args.next())?;
            }
            "--c-puct" => {
                options.c_puct = parse_value(&argument, args.next())?;
            }
            "--temperature-plies" => {
                options.temperature_plies = parse_value(&argument, args.next())?;
            }
            "--temperature" => {
                options.temperature = parse_value(&argument, args.next())?;
            }
            "--root-dirichlet-alpha" => {
                options.root_dirichlet_alpha = parse_value(&argument, args.next())?;
            }
            "--root-exploration-fraction" => {
                options.root_exploration_fraction = parse_value(&argument, args.next())?;
            }
            "--log-every-plies" => {
                options.log_every_plies = parse_value(&argument, args.next())?;
            }
            "--slow-search-ms" => {
                options.slow_search_ms = parse_value(&argument, args.next())?;
            }
            "--quiet-progress" => {
                options.quiet_progress = true;
            }
            "--threads" => {
                options.threads = parse_value(&argument, args.next())?;
            }
            "--seed" => {
                options.seed = parse_value(&argument, args.next())?;
            }
            "--output" => {
                let value = args.next().ok_or_else(|| format!("missing value for {argument}"))?;
                options.output = PathBuf::from(value);
            }
            "--student-model" | "--model" | "--neural-model" => {
                let value = args.next().ok_or_else(|| format!("missing value for {argument}"))?;
                options.student_model = Some(PathBuf::from(value));
            }
            "--model-policy-temperature" => {
                options.model_policy_temperature = parse_value(&argument, args.next())?;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                return Err(format!("unknown argument: {argument}").into());
            }
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
    println!("Usage: cargo run --bin selfplay -- [options]");
    println!("  --games <n>               Number of self-play games to generate");
    println!("  --simulations <n>         MCTS simulations per move");
    println!("  --c-puct <value>         Exploration constant for PUCT");
    println!("  --temperature-plies <n>  Sample from visit counts for the first n plies");
    println!("  --temperature <value>    Sampling temperature during the exploration window");
    println!("  --root-dirichlet-alpha <value>  Dirichlet numerator for dynamic root noise; actual alpha is value / legal_actions");
    println!("  --root-exploration-fraction <value>  Fraction of root prior mass replaced by Dirichlet noise");
    println!("  --log-every-plies <n>    Keep aggregated progress logging enabled; set 0 to disable");
    println!("  --slow-search-ms <n>     Count searches above n ms inside aggregated progress summaries");
    println!("  --quiet-progress         Disable aggregated progress logging");
    println!("  --threads <n>            Number of self-play worker threads");
    println!("  --seed <n>               RNG seed for self-play action sampling");
    println!("  --output <path>          Output JSONL path");
    println!("  --student-model <path>   Rust-exported neural manifest path");
    println!("  --model <path>           Alias for --student-model");
    println!("  --neural-model <path>    Alias for --student-model");
    println!("  --model-policy-temperature <value>  Temperature applied to neural policy logits");
}
