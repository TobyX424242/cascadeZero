use cascade_data::{
    run_league_play, run_parallel_league_play, CollectorConfig, JsonlDatasetSink, LeagueMember,
    LeagueMemberTier, LeaguePlayConfig, Mcts, MctsConfig, NeuralEvaluator,
    NeuralEvaluatorConfig, ParallelLeaguePlayConfig, PositionEvaluator,
    TrainingDataCollector,
};
use serde_json::json;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
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
    candidate_name: String,
    candidate_model: Option<PathBuf>,
    league_models: Vec<PathBuf>,
    elite_models: Vec<PathBuf>,
    model_policy_temperature: f32,
    record_league_moves: bool,
    report_json: Option<PathBuf>,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            games: 2,
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
            seed: 20260505,
            output: PathBuf::from("artifacts/local_leagueplay.jsonl"),
            candidate_name: "candidate".to_string(),
            candidate_model: None,
            league_models: Vec::new(),
            elite_models: Vec::new(),
            model_policy_temperature: 1.0,
            record_league_moves: false,
            report_json: None,
        }
    }
}

#[derive(Debug, Clone)]
enum CliEvaluator {
    Neural(NeuralEvaluator),
}

impl PositionEvaluator for CliEvaluator {
    fn evaluate(
        &self,
        state: &cascade_data::GameState,
    ) -> Result<cascade_data::Evaluation, cascade_data::DataCollectionError> {
        match self {
            Self::Neural(evaluator) => evaluator.evaluate(state),
        }
    }
}

#[derive(Debug)]
struct LoadedLeagueSearch {
    name: String,
    tier: LeagueMemberTier,
    model_path: PathBuf,
    search: Mcts<CliEvaluator>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_args(env::args().skip(1))?;

    if let Some(parent) = options.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    if let Some(report_json) = &options.report_json {
        if let Some(parent) = report_json.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
    }

    let candidate_model = options
        .candidate_model
        .as_ref()
        .ok_or("--candidate-model is required")?;
    let evaluator_config = NeuralEvaluatorConfig::default()
        .with_policy_temperature(options.model_policy_temperature);
    let candidate_search = Mcts::new(
        MctsConfig::default()
            .with_simulations(options.simulations)
            .with_c_puct(options.c_puct),
        CliEvaluator::Neural(NeuralEvaluator::load(candidate_model, evaluator_config.clone())?),
    );

    let mut loaded_league_searches = Vec::new();
    for model_path in &options.league_models {
        loaded_league_searches.push(load_league_search(
            model_path,
            LeagueMemberTier::History,
            options.simulations,
            options.c_puct,
            &evaluator_config,
        )?);
    }
    for model_path in &options.elite_models {
        loaded_league_searches.push(load_league_search(
            model_path,
            LeagueMemberTier::Elite,
            options.simulations,
            options.c_puct,
            &evaluator_config,
        )?);
    }

    if loaded_league_searches.is_empty() {
        return Err("leagueplay requires at least one --league-model or --elite-model".into());
    }

    let league_members: Vec<_> = loaded_league_searches
        .iter()
        .map(|entry| LeagueMember::new(entry.name.clone(), entry.tier, &entry.search))
        .collect();

    let sink = JsonlDatasetSink::create(&options.output)?;
    let collector = TrainingDataCollector::new(
        CollectorConfig::new("local-teacher")
            .with_flush_interval(32)
            .with_metadata("generator", json!("local-leagueplay-bin"))
            .with_metadata("candidate_name", json!(options.candidate_name.as_str())),
        sink,
    );
    let leagueplay_config = LeaguePlayConfig::default()
        .with_games(options.games)
        .with_sampling_temperature(options.temperature)
        .with_temperature_drop_ply(options.temperature_plies)
        .with_root_dirichlet_alpha(options.root_dirichlet_alpha)
        .with_root_exploration_fraction(options.root_exploration_fraction)
        .with_progress_log_every_plies(if options.quiet_progress {
            0
        } else {
            options.log_every_plies
        })
        .with_slow_search_log_millis(if options.quiet_progress {
            0
        } else {
            options.slow_search_ms
        })
        .with_log_game_summaries(!options.quiet_progress)
        .with_record_league_moves(options.record_league_moves)
        .with_seed(options.seed);

    let started_at = Instant::now();
    let (summary, _collector) = if options.threads > 1 {
        run_parallel_league_play(
            leagueplay_config.clone(),
            ParallelLeaguePlayConfig::default().with_threads(options.threads),
            options.candidate_name.clone(),
            &candidate_search,
            &league_members,
            collector,
        )?
    } else {
        run_league_play(
            leagueplay_config.clone(),
            options.candidate_name.clone(),
            &candidate_search,
            &league_members,
            collector,
        )?
    };
    let elapsed_ms = started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;

    if let Some(report_json) = &options.report_json {
        let report = json!({
            "config": leagueplay_config,
            "candidate": {
                "name": options.candidate_name,
                "model": candidate_model.display().to_string(),
            },
            "league_members": loaded_league_searches
                .iter()
                .map(|entry| {
                    json!({
                        "name": entry.name,
                        "tier": entry.tier,
                        "model": entry.model_path.display().to_string(),
                    })
                })
                .collect::<Vec<_>>(),
            "simulations": options.simulations,
            "c_puct": options.c_puct,
            "elapsed_ms": elapsed_ms,
            "summary": summary,
        });
        fs::write(report_json, serde_json::to_vec_pretty(&report)?)?;
    }

    println!(
        "candidate={} games={} candidate_wins={} league_wins={} draws={} samples={} total_plies={} league_models={} elapsed_ms={} threads={} output={}{}",
        options.candidate_name,
        summary.games_played,
        summary.candidate_wins,
        summary.league_wins,
        summary.draws,
        summary.samples_written,
        summary.total_plies,
        summary.opponents.len(),
        elapsed_ms,
        options.threads,
        options.output.display(),
        options
            .report_json
            .as_ref()
            .map(|path| format!(" report_json={}", path.display()))
            .unwrap_or_default(),
    );

    Ok(())
}

fn load_league_search(
    model_path: &Path,
    tier: LeagueMemberTier,
    simulations: u32,
    c_puct: f32,
    evaluator_config: &NeuralEvaluatorConfig,
) -> Result<LoadedLeagueSearch, Box<dyn Error>> {
    let name = model_label(model_path);
    let search = Mcts::new(
        MctsConfig::default()
            .with_simulations(simulations)
            .with_c_puct(c_puct),
        CliEvaluator::Neural(NeuralEvaluator::load(model_path, evaluator_config.clone())?),
    );
    Ok(LoadedLeagueSearch {
        name,
        tier,
        model_path: model_path.to_path_buf(),
        search,
    })
}

fn model_label(path: &Path) -> String {
    path.file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.display().to_string())
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
            "--temperature-plies" => {
                options.temperature_plies = parse_value(&argument, args.next())?
            }
            "--temperature" => options.temperature = parse_value(&argument, args.next())?,
            "--root-dirichlet-alpha" => {
                options.root_dirichlet_alpha = parse_value(&argument, args.next())?
            }
            "--root-exploration-fraction" => {
                options.root_exploration_fraction = parse_value(&argument, args.next())?
            }
            "--log-every-plies" => {
                options.log_every_plies = parse_value(&argument, args.next())?
            }
            "--slow-search-ms" => options.slow_search_ms = parse_value(&argument, args.next())?,
            "--quiet-progress" => options.quiet_progress = true,
            "--threads" => options.threads = parse_value(&argument, args.next())?,
            "--seed" => options.seed = parse_value(&argument, args.next())?,
            "--output" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {argument}"))?;
                options.output = PathBuf::from(value);
            }
            "--candidate-name" => {
                options.candidate_name = args
                    .next()
                    .ok_or_else(|| format!("missing value for {argument}"))?;
            }
            "--candidate-model" | "--model" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {argument}"))?;
                options.candidate_model = Some(PathBuf::from(value));
            }
            "--league-model" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {argument}"))?;
                options.league_models.push(PathBuf::from(value));
            }
            "--elite-model" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {argument}"))?;
                options.elite_models.push(PathBuf::from(value));
            }
            "--model-policy-temperature" => {
                options.model_policy_temperature = parse_value(&argument, args.next())?
            }
            "--record-league-moves" => options.record_league_moves = true,
            "--report-json" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {argument}"))?;
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
    println!("Usage: cargo run --bin leagueplay -- [options]");
    println!("  --games <n>                      Total games to play; rounded up to an even count");
    println!("  --simulations <n>                MCTS simulations per move for candidate and league");
    println!("  --c-puct <value>                 Exploration constant for both sides");
    println!("  --temperature-plies <n>          Sample from visit counts for the first n plies");
    println!("  --temperature <value>            Sampling temperature during the exploration window");
    println!("  --root-dirichlet-alpha <value>   Dirichlet alpha numerator; actual alpha is numerator / legal_action_count");
    println!("  --root-exploration-fraction <value>  Fraction of root prior mass replaced by Dirichlet noise");
    println!("  --log-every-plies <n>            Keep aggregated progress logging enabled; set 0 to disable");
    println!("  --slow-search-ms <n>             Count searches above n ms inside aggregated progress summaries");
    println!("  --quiet-progress                 Disable aggregated progress logging");
    println!("  --threads <n>                    Number of leagueplay worker threads");
    println!("  --seed <n>                       RNG seed for opponent selection and action sampling");
    println!("  --output <path>                  Output JSONL path");
    println!("  --candidate-name <name>          Label for the new model side");
    println!("  --candidate-model <path>         Manifest path for the new model");
    println!("  --model <path>                   Alias for --candidate-model");
    println!("  --league-model <path>            Historical league model; may be repeated");
    println!("  --elite-model <path>             Elite league model that is always included; may be repeated");
    println!("  --model-policy-temperature <value>  Temperature applied to neural policy logits");
    println!("  --record-league-moves            Keep league-side samples in the output dataset");
    println!("  --report-json <path>             Optional JSON report path");
}
